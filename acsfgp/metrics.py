"""Evaluation metrics for energies, forces and uncertainty."""
from __future__ import annotations

import numpy as np

__all__ = ["evaluate", "force_metrics", "uncertainty_metrics"]


def evaluate(y_true, y_pred, n_atoms: int | None = None) -> dict:
    """MAE / RMSE / R^2 for energies, plus per-atom MAE if n_atoms is given."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    denom = ((y_true - y_true.mean()) ** 2).sum()
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / denom) if denom > 0 else float("nan"),
    }
    if n_atoms:
        out["mae_per_atom_meV"] = out["mae"] / n_atoms * 1000
    return out


def force_metrics(f_true, f_pred) -> dict:
    """MAE / RMSE / R^2 over all force components."""
    f_true = np.asarray(f_true, float).ravel()
    f_pred = np.asarray(f_pred, float).ravel()
    err = f_pred - f_true
    denom = ((f_true - f_true.mean()) ** 2).sum()
    return {
        "force_mae": float(np.abs(err).mean()),
        "force_rmse": float(np.sqrt((err ** 2).mean())),
        "force_r2": float(1 - (err ** 2).sum() / denom) if denom > 0 else float("nan"),
    }


def uncertainty_metrics(sigma, abs_error, tail_quantile: float = 0.90) -> dict:
    """How well a predicted uncertainty tracks the true error.

    spearman scores the whole ranking; auc_tail scores only whether the worst
    frames rank above the rest, which is what a trigger needs. Near-zero
    correlation with AUC well above 0.5 is common.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    sigma = np.asarray(sigma, float)
    err = np.asarray(abs_error, float)
    out = {
        "spearman": float(spearmanr(sigma, err).statistic),
        "pearson": float(np.corrcoef(sigma, err)[0, 1]) if sigma.std() > 0 else float("nan"),
        "sigma_cv": float(sigma.std() / abs(sigma.mean())) if sigma.mean() else float("nan"),
    }
    label = (err >= np.quantile(err, tail_quantile)).astype(int)
    out["auc_tail"] = (float(roc_auc_score(label, sigma))
                       if 0 < label.sum() < len(label) else float("nan"))
    return out
