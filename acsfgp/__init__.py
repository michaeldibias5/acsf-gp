"""acsfgp - ACSF descriptors + Gaussian process surrogates for MD trajectories.

Quickstart
----------
>>> from acsfgp import Surrogate
>>> model = Surrogate.from_trajectory("md.traj", n_frames=300)
>>> model.train()
>>> model.predict(atoms)                       # energy (eV)
>>> model.predict(atoms, return_uncertainty=True)   # (energy, sigma)
>>> model.predict_forces(atoms)                # (n_atoms, 3) eV/Angstrom
"""
from .descriptors import ACSF, DEFAULT_G2, DEFAULT_G4
from .model import Surrogate
from .data import load_trajectory, split_indices
from .metrics import evaluate, force_metrics, uncertainty_metrics

__version__ = "0.1.0"
__all__ = [
    "Surrogate", "ACSF", "DEFAULT_G2", "DEFAULT_G4",
    "load_trajectory", "split_indices",
    "evaluate", "force_metrics", "uncertainty_metrics",
]
