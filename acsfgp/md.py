"""NVT molecular dynamics driven by a trained surrogate.

The surrogate replaces the expensive potential inside the ASE Langevin
integrator, which supplies the thermostat that makes the ensemble NVT.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md import velocitydistribution as _vd

from .calculator import SurrogateCalculator

__all__ = ["run_nvt", "benchmark"]


def benchmark(model, atoms, n_evals: int = 10, timestep_fs: float = 1.0,
              uncertainty: bool = True, verbose: bool = True) -> dict:
    """Time energy+force evaluations to estimate the achievable MD rate.

    One MD step costs one evaluation, so this answers "how long would a
    nanosecond take on this machine" without running the dynamics.
    """
    model._check_fitted()
    atoms = atoms.copy()
    model.energy_and_forces(atoms, return_uncertainty=uncertainty)
    t0 = time.time()
    for _ in range(n_evals):
        model.energy_and_forces(atoms, return_uncertainty=uncertainty)
    per_eval = (time.time() - t0) / n_evals
    out = {
        "n_atoms": len(atoms),
        "ms_per_step": 1000 * per_eval,
        "steps_per_second": 1.0 / per_eval,
        "ns_per_day": (1.0 / per_eval) * 86400 * timestep_fs * 1e-6,
        "hours_per_10ps": 10000 / timestep_fs * per_eval / 3600,
    }
    if verbose:
        print(f"{out['n_atoms']} atoms: {out['ms_per_step']:.1f} ms/step "
              f"({out['steps_per_second']:.1f} steps/s, "
              f"{out['ns_per_day']:.3f} ns/day at {timestep_fs:g} fs)")
        print(f"  10 ps of dynamics would take {out['hours_per_10ps']:.1f} h")
    return out


def _min_distance(atoms, probe: float = 3.5) -> float:
    """Shortest interatomic distance, searching out to `probe` angstrom.

    Returns probe itself if nothing is closer, which is all the stability
    check needs to know.
    """
    from ase.neighborlist import neighbor_list
    d = neighbor_list("d", atoms, probe)
    return float(d.min()) if len(d) else float(probe)


def _thermalize(atoms, temperature_K: float, seed: int) -> None:
    """Draw Maxwell-Boltzmann velocities, then remove the net drift.

    ASE renamed this function; both spellings are handled so the package works
    on older and newer installs.
    """
    rng = np.random.RandomState(seed)
    fn = getattr(_vd, "thermalize_momenta", None) or _vd.MaxwellBoltzmannDistribution
    fn(atoms, temperature_K=temperature_K, rng=rng)
    _vd.Stationary(atoms)


def run_nvt(model, atoms, temperature_K: float, steps: int = 1000,
            timestep_fs: float = 1.0, friction_per_fs: float = 0.01,
            trajectory: str | None = None, log_every: int = 10,
            seed: int = 0, initialize_velocities: bool = True,
            uncertainty: bool = True, max_temperature_K: float | None = None,
            min_distance_A: float = 0.7, sigma_factor: float | None = 10.0,
            verbose: bool = True) -> dict:
    """Run NVT dynamics on `atoms` with forces from `model`.

    temperature_K sets the thermostat target, friction_per_fs how hard it
    couples.

    The run stops early and reports why in `stopped_because` if any of these
    happen: the temperature passes max_temperature_K (default 5x the target),
    two atoms come closer than min_distance_A, the energy stops being finite,
    the predicted uncertainty grows past sigma_factor times its starting value
    (set None to disable), or every force comes back exactly zero. The last
    one is the characteristic GP failure: once a structure is far enough from
    every training point the kernel underflows, the model predicts its prior
    mean everywhere, and the atoms fly apart with nothing holding them.

    Returns a dict with the final structure, the log, and timing in steps/s
    and ns/day.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    atoms = atoms.copy()
    known = set(model.descriptor.species)
    unknown = sorted(set(atoms.get_chemical_symbols()) - known)
    if unknown:
        raise ValueError(f"model was not trained on {unknown}; "
                         f"it knows {sorted(known)}")
    if model._n_atoms_train and len(atoms) != model._n_atoms_train:
        warnings.warn(
            f"model was trained on {model._n_atoms_train}-atom frames but this "
            f"structure has {len(atoms)}; total-energy predictions will be off",
            stacklevel=2)

    calc = SurrogateCalculator(model, uncertainty=uncertainty)
    atoms.calc = calc
    if max_temperature_K is None:
        max_temperature_K = max(5.0 * temperature_K, temperature_K + 1000.0)

    if initialize_velocities:
        _thermalize(atoms, temperature_K, seed)

    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=friction_per_fs / units.fs, fixcm=False,
                   rng=np.random.RandomState(seed + 1))

    writer = None
    if trajectory:
        from ase.io.trajectory import Trajectory
        writer = Trajectory(trajectory, "w", atoms)

    log: list[dict] = []
    stopped_because = None

    def record(step):
        row = {
            "step": int(step),
            "time_fs": float(step * timestep_fs),
            "temperature_K": float(atoms.get_temperature()),
            "potential_energy_eV": float(atoms.get_potential_energy()),
            "kinetic_energy_eV": float(atoms.get_kinetic_energy()),
            "max_force_eV_per_A": float(np.abs(atoms.get_forces()).max()),
            "energy_std_eV": float(calc.results.get("energy_std", float("nan"))),
            "min_distance_A": _min_distance(atoms, max(3.5, 3 * min_distance_A)),
        }
        log.append(row)
        if writer is not None:
            writer.write(atoms)
        return row

    def line(row):
        return (f"  step {row['step']:>6}  {row['time_fs']:>8.1f} fs  "
                f"T {row['temperature_K']:>7.1f} K  "
                f"E {row['potential_energy_eV']:>12.3f} eV  "
                f"sigma {row['energy_std_eV']:.3f} eV  "
                f"dmin {row['min_distance_A']:.2f} A")

    if verbose:
        print(f"NVT: {len(atoms)} atoms, target {temperature_K:g} K, "
              f"{steps} steps x {timestep_fs:g} fs "
              f"= {steps * timestep_fs / 1000:.2f} ps")
    row = record(0)
    if verbose:
        print(line(row))
    sigma_limit = None
    if sigma_factor and np.isfinite(row["energy_std_eV"]) and row["energy_std_eV"] > 0:
        sigma_limit = sigma_factor * row["energy_std_eV"]

    t0 = time.time()
    done = 0
    while done < steps:
        chunk = min(log_every, steps - done)
        dyn.run(chunk)
        done += chunk
        row = record(done)
        if verbose:
            print(line(row))
        if not np.isfinite(row["potential_energy_eV"]) or not np.isfinite(row["temperature_K"]):
            stopped_because = "energy or temperature became non-finite"
        elif row["max_force_eV_per_A"] == 0.0:
            stopped_because = ("every force came back exactly zero - the structure "
                               "left the training data far enough that the model has "
                               "nothing to say about it")
        elif sigma_limit is not None and row["energy_std_eV"] > sigma_limit:
            stopped_because = (f"uncertainty rose to {row['energy_std_eV']:.3g} eV, past "
                               f"{sigma_factor:g}x its starting value ({sigma_limit:.3g} eV)")
        elif row["temperature_K"] > max_temperature_K:
            stopped_because = (f"temperature {row['temperature_K']:.0f} K exceeded "
                               f"max_temperature_K={max_temperature_K:.0f} K")
        elif row["min_distance_A"] < min_distance_A:
            stopped_because = (f"atoms came within {row['min_distance_A']:.2f} A, "
                               f"below min_distance_A={min_distance_A:.2f}")
        if stopped_because:
            break
    wall = time.time() - t0

    if writer is not None:
        writer.close()

    steps_per_s = done / wall if wall > 0 else float("inf")
    result = {
        "atoms": atoms,
        "log": log,
        "steps_completed": int(done),
        "steps_requested": int(steps),
        "wall_time_s": float(wall),
        "steps_per_second": float(steps_per_s),
        "ms_per_step": float(1000 * wall / done) if done else float("nan"),
        "ns_per_day": float(steps_per_s * 86400 * timestep_fs * 1e-6),
        "n_energy_evaluations": int(calc.n_calls),
        "mean_temperature_K": float(np.mean([r["temperature_K"] for r in log[1:]] or [np.nan])),
        "stopped_because": stopped_because,
        "trajectory": trajectory,
    }
    sig = np.array([r["energy_std_eV"] for r in log], dtype=float)
    if np.isfinite(sig).any():
        result["max_energy_std_eV"] = float(np.nanmax(sig))
        result["mean_energy_std_eV"] = float(np.nanmean(sig))

    if verbose:
        print(f"\n{done}/{steps} steps in {wall:.1f} s")
        print(f"  {result['ms_per_step']:.1f} ms/step "
              f"({steps_per_s:.1f} steps/s, {result['ns_per_day']:.3f} ns/day)")
        print(f"  mean T {result['mean_temperature_K']:.1f} K "
              f"(target {temperature_K:g} K)")
        if stopped_because:
            print(f"  STOPPED EARLY: {stopped_because}")
        if trajectory:
            print(f"  trajectory -> {trajectory}")
    return result
