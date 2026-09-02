"""Train on a trajectory, then predict.

    python examples/quickstart.py path/to/md.traj [n_frames]
"""
import sys

import numpy as np

from acsfgp import Surrogate, force_metrics, uncertainty_metrics


def main(path: str, n_frames: int = 200) -> None:

    print(f"loading {path} ...")
    model = Surrogate.from_trajectory(path, n_frames=n_frames)
    print(model)

    info = model.train(test_fraction=0.2, split="random")
    print(f"\nheld-out energy MAE : {info['mae_per_atom_meV']:.3f} meV/atom")
    print(f"held-out R^2        : {info['r2']:.4f}")


    atoms = model._images[0]
    energy, sigma = model.predict(atoms, return_uncertainty=True)
    forces = model.predict_forces(atoms)
    print(f"\nfirst frame: E = {energy:.4f} +/- {sigma:.4f} eV")
    print(f"             max |F| = {np.abs(forces).max():.4f} eV/A")


    ref_f = np.concatenate([a.get_forces() for a in model._images[:5]])
    pred_f = np.concatenate([model.predict_forces(a) for a in model._images[:5]])
    fm = force_metrics(ref_f, pred_f)
    print(f"\nforces on 5 frames  : MAE {fm['force_mae']:.4f} eV/A, "
          f"R^2 {fm['force_r2']:.4f}")


    idx = np.arange(0, len(model._images), 3)[:60]
    imgs = [model._images[i] for i in idx]
    mu, sd = model.predict(imgs, return_uncertainty=True)
    err = np.abs(mu - np.array([a.get_potential_energy() for a in imgs]))
    um = uncertainty_metrics(sd, err)
    print(f"uncertainty vs error: spearman {um['spearman']:+.3f}, "
          f"tail AUC {um['auc_tail']:.3f}")
    print("  (AUC = chance a badly-predicted frame outranks a good one; 0.5 = no signal)")


    model.save("model.joblib")
    reloaded = Surrogate.load("model.joblib")
    assert np.isclose(reloaded.predict(atoms), energy)
    print("\nsaved to model.joblib and verified reload")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
