# acsf-gp

You give it an MD trajectory, it learns from that trajectory, then, for any new
atomic structure, it can tell you:

- the **energy**
- the **forces** on every atom
- **how confident it is** in that energy

The idea is to have something much cheaper to run than a full machine-learned
interatomic potential (MLIP), which can stand in for one during long
simulations and say when it isn't sure.

---

## Installing

```bash
git clone https://github.com/<your-username>/acsf-gp.git
cd acsf-gp
pip install -e .
```

Needs Python 3.9 or newer.

---

## Using it

```python
from acsfgp import Surrogate

# train on a trajectory (300 frames spread across the whole run)
model = Surrogate.from_trajectory("md.traj", n_frames=300)
model.train()

# now use it on any structure
energy = model.predict(atoms)
energy, uncertainty = model.predict(atoms, return_uncertainty=True)
forces = model.predict_forces(atoms)

# save it so you don't have to retrain
model.save("model.joblib")
model = Surrogate.load("model.joblib")
```

`model.train()` prints how accurate it is on data it didn't train on.

You can also run it from the terminal without writing any Python:

```bash
acsfgp train md.traj --out model.joblib
acsfgp predict model.joblib structure.traj --forces
```

---

## What the main options do

```python
Surrogate.from_trajectory(
    "md.traj",
    n_frames=300,     # how many frames to learn from
    model="gp",       # "gp" gives uncertainty; "ridge" is faster but doesn't
    r_cut=5.0,        # how far around each atom to look, in Angstroms
)

model.train(
    test_fraction=0.2,   # hold back 20% of frames to check accuracy
    split="random",      # or "drift" - see below
)
```

**`split="random"` vs `split="drift"`** is worth understanding:

- `"random"` shuffles the frames, so the test frames are mixed in among the
  training frames.
- `"drift"` trains on the *early* frames and tests on the *later* ones. This is
  harder and more realistic.

train once with each split and compare; a large gap means the system drifts and will 
need retraining during a run


---

## About the uncertainty

The model reports a `sigma` with each energy. It gets **larger** when the
structure looks unfamiliar compared to the training data.

Something to note from prior tests: when the model is already very accurate, 
the uncertainty doesn't track the error well, because whatever error is left 
isn't caused by missing data. The uncertainty becomes much more useful when 
the model is predicting from new configurations.

There's a helper that measures whether the worst predictions get flagged:

```python
from acsfgp import uncertainty_metrics
uncertainty_metrics(sigma_values, actual_errors)
# -> {"spearman": ..., "auc_tail": ...}
```

`auc_tail` is the chance that a badly-predicted structure gets a higher
uncertainty than a well-predicted one. 0.5 means no signal, and the higher
the number the better.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

14 tests, covering the descriptors, the force calculation, and saving/loading.

---

## Example

`examples/quickstart.py` runs the whole workflow on a trajectory:

```bash
python examples/quickstart.py md.traj
```

---

## License

MIT
