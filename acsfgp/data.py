"""Trajectory loading and train/test splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_trajectory", "split_indices"]


def load_trajectory(path, n_frames: int | None = 300, stride: int | None = None,
                    skip_first: int = 0):
    """Load structures from an ASE-readable trajectory.

    n_frames takes that many frames spread evenly across the run; stride is an
    alternative used only when n_frames is None. skip_first drops leading
    frames, useful when a run starts from an unrelaxed structure. Frames stay
    in time order so the "drift" split remains meaningful.
    """
    from ase.io.trajectory import Trajectory

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    traj = Trajectory(str(path))
    total = len(traj)
    if total == 0:
        raise ValueError(f"{path} contains no frames")

    lo = min(skip_first, total - 1)
    if n_frames is not None:
        idx = np.unique(np.linspace(lo, total - 1, min(n_frames, total - lo)).astype(int))
    elif stride is not None:
        idx = np.arange(lo, total, stride)
    else:
        idx = np.arange(lo, total)
    return [traj[int(i)] for i in idx]


def split_indices(n: int, test_fraction: float = 0.2, split: str = "random",
                  random_state: int = 0):
    """Return (train_idx, test_idx).

    "random" shuffles frames; "drift" trains on the earliest frames and tests
    on all later ones.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if split == "random":
        perm = np.random.RandomState(random_state).permutation(n)
        n_test = max(1, int(round(n * test_fraction)))
        return np.sort(perm[n_test:]), np.sort(perm[:n_test])
    if split == "drift":
        n_train = max(1, int(round(n * (1 - test_fraction))))
        return np.arange(n_train), np.arange(n_train, n)
    raise ValueError("split must be 'random' or 'drift'")
