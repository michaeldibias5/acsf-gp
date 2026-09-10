"""Command-line interface: acsfgp train / acsfgp predict."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _train(args) -> int:
    from .model import Surrogate
    model = Surrogate.from_trajectory(
        args.trajectory, n_frames=args.n_frames, model=args.model, r_cut=args.r_cut)
    info = model.train(test_fraction=args.test_fraction, split=args.split)
    model.save(args.out)
    print(f"\nsaved -> {args.out}")
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in info.items()}, indent=1))
    return 0


def _predict(args) -> int:
    from ase.io import read
    from .model import Surrogate
    model = Surrogate.load(args.model)
    images = read(args.structure, index=args.index)
    if not isinstance(images, list):
        images = [images]
    for n, atoms in enumerate(images):
        row = {"frame": n, "n_atoms": len(atoms)}
        if model.model_type == "gp":
            e, s = model.predict(atoms, return_uncertainty=True)
            row.update(energy_eV=round(e, 6), sigma_eV=round(s, 6))
        else:
            row.update(energy_eV=round(model.predict(atoms), 6))
        if args.forces:
            f = model.predict_forces(atoms)
            row.update(max_force_eV_per_A=round(float(np.abs(f).max()), 6))
        print(json.dumps(row))
    return 0


def _md(args) -> int:
    from ase.io import read
    from .model import Surrogate
    model = Surrogate.load(args.model)
    atoms = read(args.structure, index=args.index)
    if isinstance(atoms, list):
        atoms = atoms[0]
    out = model.run_md(
        atoms, temperature_K=args.temperature, steps=args.steps,
        timestep_fs=args.timestep, friction_per_fs=args.friction,
        trajectory=args.out, log_every=args.log_every)
    print(json.dumps({k: round(out[k], 4) for k in
                      ("ms_per_step", "steps_per_second", "ns_per_day",
                       "mean_temperature_K", "wall_time_s")}, indent=1))
    return 1 if out["stopped_because"] else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="acsfgp", description="ACSF + Gaussian process surrogate models")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a model on an MD trajectory")
    t.add_argument("trajectory")
    t.add_argument("--out", default="model.joblib")
    t.add_argument("--n-frames", type=int, default=300)
    t.add_argument("--r-cut", type=float, default=5.0)
    t.add_argument("--model", choices=["gp", "ridge"], default="gp")
    t.add_argument("--split", choices=["random", "drift"], default="random")
    t.add_argument("--test-fraction", type=float, default=0.2)
    t.set_defaults(func=_train)

    q = sub.add_parser("predict", help="predict with a saved model")
    q.add_argument("model")
    q.add_argument("structure")
    q.add_argument("--index", default=":", help="ASE index string (default all)")
    q.add_argument("--forces", action="store_true", help="also report max |force|")
    q.set_defaults(func=_predict)

    m = sub.add_parser("md", help="run NVT dynamics with a saved model")
    m.add_argument("model")
    m.add_argument("structure")
    m.add_argument("--index", default="0", help="ASE index of the start frame")
    m.add_argument("--temperature", type=float, default=300.0, help="kelvin")
    m.add_argument("--steps", type=int, default=500)
    m.add_argument("--timestep", type=float, default=1.0, help="femtoseconds")
    m.add_argument("--friction", type=float, default=0.01, help="per fs")
    m.add_argument("--out", default="md_out.traj")
    m.add_argument("--log-every", type=int, default=25)
    m.set_defaults(func=_md)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
