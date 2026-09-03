"""Train on a trajectory, then predict.

    python examples/quickstart.py path/to/md.traj [n_frames]

Everything reported below is measured on frames the model did not train on.
"""
import sys

import numpy as np

from acsfgp import Surrogate, force_metrics, uncertainty_metrics


def n_force_frames(n_atoms: int) -> int:
    """Force evaluation cost scales with system size, so scale the sample."""
    if n_atoms < 500:
        return 25
    if n_atoms < 1500:
        return 10
    return 4


def main(path: str, n_frames: int = 300) -> None:
    print(f"loading {path} ...")
    model = Surrogate.from_trajectory(path, n_frames=n_frames)
    print(model)

    info = model.train(test_fraction=0.2, split="random")
    print(f"\nheld-out energy MAE : {info['mae_per_atom_meV']:.3f} meV/atom")
    print(f"held-out R^2        : {info['r2']:.4f}")
    print(f"trained on {info['n_train']} frames, tested on {info['n_test']}")

    held_out = model.test_images
    n_atoms = len(held_out[0])

    atoms = held_out[0]
    energy, sigma = model.predict(atoms, return_uncertainty=True)
    forces = model.predict_forces(atoms)
    print(f"\none held-out frame  : E = {energy:.4f} +/- {sigma:.4f} eV")
    print(f"                      max |F| = {np.abs(forces).max():.4f} eV/A")

    nf = min(n_force_frames(n_atoms), len(held_out))
    sample = held_out[:nf]
    ref_f = np.concatenate([a.get_forces() for a in sample])
    pred_f = np.concatenate([model.predict_forces(a) for a in sample])
    fm = force_metrics(ref_f, pred_f)
    print(f"\nforces ({nf} frames, {ref_f.size} components):")
    print(f"  MAE {fm['force_mae']:.4f} eV/A, R^2 {fm['force_r2']:.4f}")

    mu, sd = model.predict(held_out, return_uncertainty=True)
    err = np.abs(mu - np.array([a.get_potential_energy() for a in held_out]))
    um = uncertainty_metrics(sd, err)
    print(f"\nuncertainty ({len(held_out)} frames):")
    print(f"  spearman {um['spearman']:+.3f}, tail AUC {um['auc_tail']:.3f}")
    print("  (AUC = chance a badly-predicted frame outranks a good one; 0.5 = no signal)")
    if len(held_out) < 100:
        print(f"  note: {len(held_out)} frames is a small sample for a correlation;")
        print("  rerun with more frames if this number matters to you")

    model.save("model.joblib")
    reloaded = Surrogate.load("model.joblib")
    assert np.isclose(reloaded.predict(atoms), energy)
    print("\nsaved to model.joblib and verified reload")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
