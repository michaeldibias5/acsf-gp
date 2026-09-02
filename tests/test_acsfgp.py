"""Tests: descriptor invariances, gradient correctness, model round trip."""
import numpy as np
import pytest
from ase import Atoms

from acsfgp import ACSF, Surrogate, evaluate, uncertainty_metrics
from acsfgp.data import split_indices


def make_atoms(seed=0, n=16, box=8.0):
    """A small random periodic two-species cell."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0, box, size=(n, 3))
    syms = ["Ag"] * (n // 2) + ["Br"] * (n - n // 2)
    return Atoms(symbols=syms, positions=pos, cell=[box] * 3, pbc=True)


@pytest.fixture(scope="module")
def acsf():
    return ACSF(species=["Ag", "Br"], r_cut=4.0)


def test_shapes(acsf):
    at = make_atoms()
    assert acsf.per_atom(at).shape == (len(at), acsf.n_features_per_atom)
    assert acsf.frame(at).shape == (acsf.n_features,)
    assert acsf.n_features == acsf.n_species * acsf.n_features_per_atom


def test_translation_invariance(acsf):
    at = make_atoms()
    a = acsf.frame(at)
    at2 = at.copy()
    at2.positions += np.array([1.3, -0.7, 2.2])
    assert np.allclose(a, acsf.frame(at2), atol=1e-8)


def test_rotation_invariance():

    rng = np.random.RandomState(1)
    pos = rng.uniform(0, 6, size=(12, 3))
    at = Atoms(symbols=["Ag"] * 6 + ["Br"] * 6, positions=pos, pbc=False)
    acsf = ACSF(species=["Ag", "Br"], r_cut=4.0)
    a = acsf.frame(at)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0],
                  [0, 0, 1]])
    at2 = at.copy()
    at2.positions = at.positions @ R.T
    assert np.allclose(a, acsf.frame(at2), atol=1e-8)


def test_permutation_invariance(acsf):
    at = make_atoms()
    a = acsf.frame(at)
    order = np.concatenate([np.random.RandomState(2).permutation(8),
                            8 + np.random.RandomState(3).permutation(8)])
    assert np.allclose(a, acsf.frame(at[order]), atol=1e-8)


def test_unknown_species_raises(acsf):
    at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[6] * 3, pbc=True)
    with pytest.raises(ValueError):
        acsf.frame(at)


def test_forces_match_finite_differences(acsf):
    """Analytic force vs finite difference of E = w . G(R)."""
    at = make_atoms(seed=5, n=14)
    rng = np.random.RandomState(0)
    w = rng.normal(size=acsf.n_features) * 0.01

    def energy(a):
        return float(w @ acsf.frame(a))

    F = acsf.forces_from_weights(at, w)
    h = 1e-4
    for i in rng.choice(len(at), 3, replace=False):
        for c in range(3):
            p0 = at.get_positions().copy()
            p = p0.copy(); p[i, c] += h; at.set_positions(p); ep = energy(at)
            p = p0.copy(); p[i, c] -= h; at.set_positions(p); em = energy(at)
            at.set_positions(p0)
            fd = -(ep - em) / (2 * h)
            assert F[i, c] == pytest.approx(fd, rel=1e-4, abs=1e-7)


def test_forces_sum_to_zero(acsf):
    """Internal forces must sum to zero."""
    at = make_atoms(seed=7, n=18)
    w = np.random.RandomState(1).normal(size=acsf.n_features) * 0.01
    F = acsf.forces_from_weights(at, w)
    assert np.allclose(F.sum(axis=0), 0.0, atol=1e-8)


@pytest.fixture(scope="module")
def trained():
    rng = np.random.RandomState(0)
    images, energies = [], []
    for k in range(40):
        at = make_atoms(seed=k, n=14)
        images.append(at)

        energies.append(float(np.sin(at.positions.sum()) + 0.01 * k))
    m = Surrogate(species=["Ag", "Br"], r_cut=4.0, model="gp")
    m.attach(images, energies)
    m.train(verbose=False)
    return m, images


def test_train_reports_metrics(trained):
    m, _ = trained
    for key in ("mae", "rmse", "r2", "n_train", "n_test", "n_features"):
        assert key in m.training_info


def test_predict_shapes(trained):
    m, images = trained
    assert isinstance(m.predict(images[0]), float)
    assert m.predict(images[:3]).shape == (3,)
    e, s = m.predict(images[0], return_uncertainty=True)
    assert isinstance(e, float) and s >= 0
    assert m.predict_forces(images[0]).shape == (len(images[0]), 3)
    assert set(m.predict_all(images[0])) == {"energy", "energy_std", "forces"}


def test_predict_before_train_raises():
    m = Surrogate(species=["Ag", "Br"])
    with pytest.raises(RuntimeError):
        m.predict(make_atoms())


def test_save_load_roundtrip(trained, tmp_path):
    m, images = trained
    p = tmp_path / "m.joblib"
    m.save(p)
    m2 = Surrogate.load(p)
    assert np.isclose(m.predict(images[0]), m2.predict(images[0]))
    assert np.allclose(m.predict_forces(images[0]), m2.predict_forces(images[0]))


def test_ridge_backend_has_no_uncertainty():
    images = [make_atoms(seed=k, n=12) for k in range(25)]
    energies = [float(np.cos(a.positions.sum())) for a in images]
    m = Surrogate(species=["Ag", "Br"], r_cut=4.0, model="ridge")
    m.attach(images, energies)
    m.train(verbose=False)
    assert m.predict_forces(images[0]).shape == (12, 3)
    with pytest.raises(ValueError):
        m.predict(images[0], return_uncertainty=True)


def test_split_indices():
    tr, te = split_indices(100, test_fraction=0.2, split="random")
    assert len(tr) == 80 and len(te) == 20 and not set(tr) & set(te)
    tr, te = split_indices(100, test_fraction=0.2, split="drift")
    assert tr.max() < te.min()
    with pytest.raises(ValueError):
        split_indices(10, split="nonsense")


def test_metrics():
    y = np.linspace(0, 1, 50)
    out = evaluate(y, y + 0.01, n_atoms=10)
    assert out["mae"] == pytest.approx(0.01)
    assert out["mae_per_atom_meV"] == pytest.approx(1.0)
    u = uncertainty_metrics(np.arange(50.0), np.arange(50.0))
    assert u["spearman"] == pytest.approx(1.0)
    assert u["auc_tail"] == pytest.approx(1.0)
