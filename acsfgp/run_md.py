"""Train on part of a trajectory, then run NVT MD with the trained model.

    python examples/run_md.py path/to/md.traj [temperature_K] [steps]

The starting structure is a held-out frame, so the model has not seen it.
"""
import sys

import numpy as np

from acsfgp import Surrogate, benchmark


def main(path: str, temperature_K: float = 300.0, steps: int = 500,
         n_frames: int = 300) -> None:
    print(f"loading {path} ...")
    model = Surrogate.from_trajectory(path, n_frames=n_frames)
    info = model.train(test_fraction=0.2, split="random")
    print(f"held-out MAE {info['mae_per_atom_meV']:.3f} meV/atom, "
          f"R^2 {info['r2']:.4f}\n")

    start = model.test_images[0]
    ref_f = start.get_forces()
    pred_f = model.predict_forces(start)
    print(f"starting structure: {len(start)} atoms, "
          f"force MAE {np.abs(pred_f - ref_f).mean():.4f} eV/A\n")

    print("speed check:")
    benchmark(model, start)
    print()

    out = model.run_md(start, temperature_K=temperature_K, steps=steps,
                       timestep_fs=1.0, friction_per_fs=0.01,
                       trajectory="md_out.traj", log_every=max(1, steps // 20))

    print("\nfeasibility on this machine:")
    print(f"  {out['ms_per_step']:.1f} ms/step -> {out['ns_per_day']:.3f} ns/day "
          f"at {len(start)} atoms")
    if out["stopped_because"]:
        print(f"  run stopped early: {out['stopped_because']}")
    else:
        print(f"  completed {out['steps_completed']} steps, mean T "
              f"{out['mean_temperature_K']:.1f} K")
    if "mean_energy_std_eV" in out:
        print(f"  model uncertainty: mean {out['mean_energy_std_eV']:.3f} eV, "
              f"max {out['max_energy_std_eV']:.3f} eV")
        print("  (rising sigma means the dynamics is leaving the training data)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else 300.0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 500)
