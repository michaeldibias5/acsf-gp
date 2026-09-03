"""Surrogate model: ACSF descriptors + a Gaussian process (or ridge)."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from .data import load_trajectory, split_indices
from .descriptors import ACSF
from .metrics import evaluate

__all__ = ["Surrogate"]


class Surrogate:
    """Energy / force / uncertainty surrogate for an MD trajectory.

    model: "gp" for uncertainty, "ridge" for speed. r_cut, g2, g4 go to ACSF.
    max_gp_train caps the GP training set (cost is O(n^3)); None disables it.
    """

    def __init__(self, species, r_cut: float = 5.0, g2=None, g4=None,
                 model: str = "gp", max_gp_train: int | None = 1200,
                 random_state: int = 0):
        if model not in ("gp", "ridge"):
            raise ValueError("model must be 'gp' or 'ridge'")
        self.descriptor = ACSF(species=species, r_cut=r_cut, g2=g2, g4=g4)
        self.model_type = model
        self.max_gp_train = max_gp_train
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._keep: np.ndarray | None = None
        self._model = None
        self._n_atoms_train: int | None = None
        self.training_info: dict = {}


    @classmethod
    def from_trajectory(cls, path, n_frames: int | None = 300, stride: int | None = None,
                        **kwargs) -> "Surrogate":
        """Build a model from a trajectory; species are read from the file."""
        images = load_trajectory(path, n_frames=n_frames, stride=stride)
        species = sorted(set(images[0].get_chemical_symbols()))
        obj = cls(species=species, **kwargs)
        obj.attach(images)
        return obj

    def attach(self, images, energies=None) -> "Surrogate":
        """Attach training structures and, optionally, their energies."""
        self._images = list(images)
        if energies is None:
            energies = [a.get_potential_energy() for a in self._images]
        self._energies = np.asarray(energies, dtype=float)
        self._n_atoms_train = len(self._images[0])
        self._X = None
        return self


    def featurize(self, verbose: bool = False) -> np.ndarray:
        """Compute and cache frame descriptors for the attached structures."""
        if getattr(self, "_X", None) is None:
            t0 = time.time()
            self._X = self.descriptor.frames(self._images)
            if verbose:
                print(f"featurised {len(self._images)} frames "
                      f"({self._X.shape[1]} features) in {time.time()-t0:.1f}s")
        return self._X

    def train(self, test_fraction: float = 0.2, split: str = "random",
              verbose: bool = True) -> dict:
        """Fit the model and return held-out metrics.

        split: "random" shuffles frames, "drift" trains on the earliest frames
        and tests on later ones.
        """
        X = self.featurize(verbose=verbose)
        y = self._energies
        train_idx, test_idx = split_indices(len(X), test_fraction=test_fraction,
                                            split=split, random_state=self.random_state)
        self._train_idx, self._test_idx = train_idx, test_idx

        self._scaler = StandardScaler().fit(X[train_idx])
        self._keep = self._scaler.scale_ > 1e-10
        Xs = self._scaler.transform(X)[:, self._keep]

        alphas = np.logspace(-4, 6, 31)
        self._alpha = float(RidgeCV(alphas=alphas).fit(Xs[train_idx], y[train_idx]).alpha_)

        t0 = time.time()
        if self.model_type == "ridge":
            self._model = Ridge(alpha=self._alpha).fit(Xs[train_idx], y[train_idx])
            fit_idx = train_idx
        else:
            fit_idx = train_idx
            if self.max_gp_train and len(train_idx) > self.max_gp_train:
                rng = np.random.RandomState(self.random_state)
                fit_idx = rng.choice(train_idx, self.max_gp_train, replace=False)
            kernel = (ConstantKernel(100.0) * RBF(np.sqrt(Xs.shape[1]) * 30)
                      + WhiteKernel(0.05, (1e-6, 1e1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    random_state=self.random_state).fit(Xs[fit_idx], y[fit_idx])
        train_time = time.time() - t0

        pred = self._model.predict(Xs[test_idx])
        info = evaluate(y[test_idx], pred, n_atoms=self._n_atoms_train)
        info.update(model=self.model_type, split=split, alpha=self._alpha,
                    n_train=int(len(fit_idx)), n_test=int(len(test_idx)),
                    n_features=int(Xs.shape[1]), train_time_s=train_time)
        self.training_info = info
        if verbose:
            print(f"trained {self.model_type} on {info['n_train']} frames "
                  f"({info['n_features']} features) in {train_time:.2f}s")
            print(f"  held-out MAE  {info['mae_per_atom_meV']:.3f} meV/atom "
                  f"({info['mae']:.4f} eV)   R2 {info['r2']:.4f}")
        return info


    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("model is not trained yet - call .train() first")

    @property
    def test_images(self):
        """Structures held out by the most recent train() call."""
        self._check_fitted()
        return [self._images[i] for i in self._test_idx]

    @property
    def train_images(self):
        """Structures used by the most recent train() call."""
        self._check_fitted()
        return [self._images[i] for i in self._train_idx]

    def _transform(self, atoms_or_list):
        images = [atoms_or_list] if isinstance(atoms_or_list, Atoms) else list(atoms_or_list)
        X = self.descriptor.frames(images)
        return self._scaler.transform(X)[:, self._keep], images

    def predict(self, atoms, return_uncertainty: bool = False):
        """Total energy of one structure or a list. Returns (energy, sigma)
        if return_uncertainty=True, which requires model="gp".
        """
        self._check_fitted()
        Xs, images = self._transform(atoms)
        single = isinstance(atoms, Atoms)
        if return_uncertainty:
            if self.model_type != "gp":
                raise ValueError("uncertainty is only available for model='gp'")
            mu, sd = self._model.predict(Xs, return_std=True)
            return (float(mu[0]), float(sd[0])) if single else (mu, sd)
        mu = self._model.predict(Xs)
        return float(mu[0]) if single else mu

    def predict_energy_per_atom(self, atoms) -> float:
        """Total energy divided by the number of atoms."""
        e = self.predict(atoms)
        n = len(atoms) if isinstance(atoms, Atoms) else len(atoms[0])
        return e / n

    def dE_dfeature(self, Xs_row: np.ndarray) -> np.ndarray:
        """dE/ddescriptor in raw (unscaled) descriptor units."""
        full = np.zeros(self.descriptor.n_features)
        idx = np.where(self._keep)[0]
        scale = self._scaler.scale_[self._keep]
        if self.model_type == "ridge":
            full[idx] = self._model.coef_ / scale
            return full
        gp = self._model
        diff = gp.X_train_ - Xs_row
        amp = gp.kernel_.k1.k1.constant_value
        ell = gp.kernel_.k1.k2.length_scale
        k = amp * np.exp(-0.5 * (diff ** 2).sum(1) / ell ** 2)
        grad = gp._y_train_std * (gp.alpha_.ravel() * k) @ diff / ell ** 2 / scale
        full[idx] = grad
        return full

    def predict_forces(self, atoms: Atoms) -> np.ndarray:
        """Forces as the analytic gradient of the predicted energy,
        shape (n_atoms, 3) in eV/Angstrom.
        """
        self._check_fitted()
        Xs, _ = self._transform(atoms)
        weights = self.dE_dfeature(Xs[0])
        return self.descriptor.forces_from_weights(atoms, weights)

    def predict_all(self, atoms: Atoms) -> dict:
        """Energy, forces and (for a GP) uncertainty in one call."""
        out = {"forces": self.predict_forces(atoms)}
        if self.model_type == "gp":
            e, s = self.predict(atoms, return_uncertainty=True)
            out.update(energy=e, energy_std=s)
        else:
            out.update(energy=self.predict(atoms))
        return out


    def save(self, path) -> None:
        """Save descriptor settings and the fitted estimator."""
        import joblib
        self._check_fitted()
        joblib.dump({
            "descriptor": self.descriptor.to_dict(),
            "model_type": self.model_type,
            "scaler": self._scaler,
            "keep": self._keep,
            "estimator": self._model,
            "alpha": getattr(self, "_alpha", None),
            "n_atoms_train": self._n_atoms_train,
            "training_info": self.training_info,
        }, Path(path))

    @classmethod
    def load(cls, path) -> "Surrogate":
        """Load a model saved with save()."""
        import joblib
        d = joblib.load(Path(path))
        obj = cls(species=d["descriptor"]["species"], model=d["model_type"])
        obj.descriptor = ACSF.from_dict(d["descriptor"])
        obj._scaler = d["scaler"]
        obj._keep = d["keep"]
        obj._model = d["estimator"]
        obj._alpha = d.get("alpha")
        obj._n_atoms_train = d.get("n_atoms_train")
        obj.training_info = d.get("training_info", {})
        return obj

    def __repr__(self) -> str:
        state = "trained" if self._model is not None else "untrained"
        return (f"Surrogate({self.model_type}, {state}, "
                f"species={self.descriptor.species}, "
                f"n_features={self.descriptor.n_features})")
