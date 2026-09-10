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

# Running molecular dynamics with it

Once the model is trained it can drive an actual simulation. You give it a
starting structure and a temperature, and it runs NVT dynamics — constant
number of atoms, constant volume, constant temperature — using ASE's Langevin
thermostat with the model supplying the forces.

```python
model = Surrogate.from_trajectory("md.traj", n_frames=300)
model.train()

start = model.test_images[0]        # a frame the model didn't train on

result = model.run_md(
    start,
    temperature_K=250,       # thermostat target
    steps=1000,              # number of MD steps
    timestep_fs=1.0,         # femtoseconds per step
    trajectory="md_out.traj" # where to save the frames
)

print(result["ms_per_step"], result["ns_per_day"])
print(result["stopped_because"])    # None if it ran to the end
```

Or from the terminal:

```bash
acsfgp md model.joblib start.traj --temperature 250 --steps 1000
```

To find out how fast the model runs before committing to a long simulation:

```python
from acsfgp import benchmark
benchmark(model, start)
# 250 atoms: 89.2 ms/step (11.2 steps/s, 0.968 ns/day at 1 fs)
#   10 ps of dynamics would take 0.2 h
```

### When it stops early

A surrogate trained on a few hundred frames does not know about the whole
potential energy surface, and it has no built-in repulsion keeping atoms
apart. If the dynamics wanders somewhere the model has never seen, it will
produce nonsense rather than an error. So `run_md` watches for that and stops
with an explanation in `stopped_because`:

- the uncertainty grows past `sigma_factor` times its starting value (10x by
  default)
- every force comes back exactly zero — the Gaussian process compares each new
  structure to the training ones, and once nothing is close enough to compare
  to, it returns its flat prior and the atoms simply drift apart
- two atoms come within `min_distance_A` of each other
- the temperature passes `max_temperature_K`
- the energy stops being a finite number

The uncertainty is the useful one here. In testing it sat flat around 0.15 eV
while the simulation stayed near the training data, then jumped by more than
two orders of magnitude the moment it left. That makes it a workable trigger
for "stop and ask the real MLIP", which is the whole point of the surrogate.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

20 tests, covering the descriptors, the force calculation, saving/loading, and
the MD driver.

---

## Examples

`examples/quickstart.py` runs the training and prediction workflow:

```bash
python examples/quickstart.py md.traj
```

`examples/run_md.py` trains a model and then runs MD with it, reporting how
fast it went:

```bash
python examples/run_md.py md.traj 250 500     # trajectory, temperature, steps
```

---

## License

MIT
