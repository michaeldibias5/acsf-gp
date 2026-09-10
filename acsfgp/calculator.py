"""ASE calculator wrapper so a trained Surrogate can drive MD."""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

__all__ = ["SurrogateCalculator"]


class SurrogateCalculator(Calculator):
    """Expose a trained Surrogate through the ASE calculator interface.

    energy_std is stored in results whenever the backend is a GP, so an MD
    driver can watch the model's own uncertainty as the run proceeds.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, model, uncertainty: bool = True, **kwargs):
        super().__init__(**kwargs)
        model._check_fitted()
        self.model = model
        self.uncertainty = bool(uncertainty) and model.model_type == "gp"
        self.n_calls = 0
        self.n_atoms_train = model._n_atoms_train

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        out = self.model.energy_and_forces(
            self.atoms, return_uncertainty=self.uncertainty)
        self.results["energy"] = out["energy"]
        self.results["free_energy"] = out["energy"]
        self.results["forces"] = np.asarray(out["forces"], dtype=float)
        if "energy_std" in out:
            self.results["energy_std"] = out["energy_std"]
        self.n_calls += 1

    def get_energy_std(self, atoms=None) -> float:
        """Predicted standard deviation of the energy, in eV."""
        if not self.uncertainty:
            return float("nan")
        self.get_potential_energy(atoms)
        return float(self.results.get("energy_std", float("nan")))
