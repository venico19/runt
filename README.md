# runt

**Reservoir UNcertainty quantification Tool** — a Python framework for ensemble-based data assimilation (EnKF) coupled to reservoir flow simulators.

runt orchestrates the standard EnKF cycle: initialize an ensemble of model parameters, forward-propagate each member through a simulator, compare simulated observations to real data, update the ensemble with a Kalman analysis step, and repeat. It supports parallel ensemble runs and pluggable simulator backends.

> **Note:** This codebase targets **Python 2** (e.g. `print` statements, `print>>f` syntax). External simulators and shell utilities assume a **Unix/Linux** environment.

---

## Architecture

runt is organized into four layers. Each layer defines an abstract base class; concrete implementations are composed at runtime in the example scripts.

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Assimilation (DA)                     │
│  Orchestrates init → forecast → analysis → write loop       │
│  DA_init2current  |  DA_levelset_fracture                   │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┼────────┬────────────────┐
    ▼        ▼        ▼                ▼
┌────────┐ ┌──────────────┐ ┌─────────────────────┐
│Analysis│ │  Ensemble    │ │ Simulator Interface │
│ EnKF   │ │  Generator   │ │ IPARS  |  MFD       │
│EnKF_   │ │  Parallel    │ │                     │
│levelset│ │  Fracture    │ │                     │
└────────┘ └──────────────┘ └─────────────────────┘
```

### 1. Data Assimilation (`runt/assimilation/`)

| File | Role |
|------|------|
| `data_assimilation_class.py` | Abstract base defining the DA workflow skeleton (`DArun`, initialization hooks, observation operator `Dynamic2Data`). |
| `DA_init2current.py` | Generic EnKF loop for static parameters (e.g. permeability). Handles ensemble filtering when simulations fail (`ens_status`). Includes a `forecast()` method for post-assimilation runs. |
| `DA_levelset_fracture.py` | Specialized DA for **level-set fracture parameterization**: converts level-set / ρ / θ parameters into fracture line segments, runs mesher + MFD, and writes fracture statistics. |

**Typical `DArun` loop:**

1. Archive prior `output/` directory and create a fresh one.
2. Initialize parameters and observations.
3. For each assimilation time step:
   - Forward-propagate the ensemble (parallel simulations).
   - Extract observation data via operator **H** (`Dynamic2Data`).
   - Run EnKF analysis to update parameters.
   - Write updated parameters and ensemble status.
4. (Optional) Forecast beyond the last observation time.

### 2. Analysis (`runt/analysis/`)

| File | Role |
|------|------|
| `analysis_class.py` | Abstract base with `create_analysis()`. |
| `enkf.py` | Standard **Ensemble Kalman Filter**. Jointly updates static parameters and dynamic state from observation innovations. |
| `enkf_levelset.py` | EnKF variant for level-set fracture models: zeroes out ρ and θ where level-set ≤ 0, and adjusts ensemble means accordingly before the Kalman update. |

**State vector layout:** `A = [parameters; dynamic variables]`, shape `(ParamSize + DynamicSize) × EnSize`.

### 3. Ensemble Generator (`runt/ensemble_generator/`)

| File | Role |
|------|------|
| `ensemble_generator_class.py` | Abstract base: write input → run simulator → read output. |
| `ensemble_generator_parallel.py` | Manages a pool of up to `parallel_num` concurrent simulator jobs; tracks completion and marks failed members inactive. |
| `ens_gen_fracture_parallel.py` | Two-stage parallel pipeline for fractures: **mesher** (build fracture mesh) then **MFD** (mixed finite difference flow solver). |

### 4. Simulator Interface (`runt/simulator_interface/`)

| File | Role |
|------|------|
| `simulator_class.py` | Abstract base: `write_input_file`, `run_simulator`, `check_finish`, `read_output_file`. |
| `ipars.py` | Adapter for **[IPARS](https://www.csm.oregonstate.edu/)** (Integrated Parallel Accurate Reservoir Simulator). Parses well output files and extracts production/injection rates at specified times. |
| `MFD_experiment.py` | Adapter for a custom **MFD** flow solver + external **mesher**. Writes fracture geometry commands and well boundary conditions. |

---

## Examples

Two end-to-end workflows live under `examples/`. Each script subclasses a DA class and a simulator interface to wire problem-specific inputs.

### Example 1: SPE5 permeability history matching (`examples/spe5_ipars_EnKF.py`)

- **Problem:** Estimate 3 log-permeability parameters for the [SPE5 benchmark](https://www.spe.org/web/csp/datasets/set05.htm) using production data from three wells.
- **Simulator:** IPARS
- **Analysis:** Standard EnKF
- **Ensemble size:** 40 (default), 10 assimilation steps

### Example 2: Level-set fracture characterization (`examples/levelset_EnKF_MFD.py`)

- **Problem:** Characterize discrete fractures on a grid using level-set indicator, length (ρ), and angle (θ) at each node. Observations are well production rates.
- **Simulator:** Custom mesher + MFD
- **Analysis:** EnKF_levelset
- **Ensemble size:** 100 (default), 15 assimilation steps

---

## Prerequisites

### Python packages

```
numpy
scipy
```

Install with pip (Python 2):

```bash
pip install numpy scipy
```

### External simulators

| Example | External tool | Notes |
|---------|---------------|-------|
| SPE5 / IPARS | IPARS v3.1+ | Must be on `$PATH` as `ipars`. Example symlinks a build at a hard-coded path — update `write_input_param` in the example. |
| Level-set / MFD | mesher + MFD | Requires `mesher.py` (from a separate mesher build) and `run_example.py` (copied from `./input/`). Paths in `MFD_experiment.py` are hard-coded and must be updated. |

### Input data

Each example expects an `input/` directory (and for SPE5, a `spe5/` directory) alongside the script:

**Level-set example** (`examples/levelset_EnKF_MFD.py`):

```
examples/
  input/
    node.dat          # fracture node coordinates
    levelset.dat      # initial level-set ensemble
    rho_norm.dat      # initial length ensemble (normal)
    theta_norm.dat    # initial angle ensemble (normal)
    obser.dat         # observation schedule and well data
    mesher_cmd        # mesher configuration
    run_example.py    # MFD driver script
```

**SPE5 example** (`examples/spe5_ipars_EnKF.py`):

```
examples/
  spe5/
    spe5.dat          # IPARS grid/data file
    IPARS.IN          # IPARS control file
    spe5.out          # reference well output (used to build synthetic observations)
```

---

## How to Run

### 1. Clone and set up the package path

runt has no `setup.py`. Make the package importable by adding a symlink or setting `PYTHONPATH`:

```bash
cd /path/to/runt
export PYTHONPATH=/path/to/runt:$PYTHONPATH
```

The example scripts also create a symlink (`ln -s ... runt`) — edit the hard-coded path in each example's `if __name__=='__main__'` block to match your install location.

### 2. Run the SPE5 / IPARS example

```bash
cd examples
# Edit spe5_ipars_EnKF.py:
#   - Update the ipars binary symlink path in IPARS_spe5_const_interface.write_input_param
#   - Ensure ./spe5/ contains spe5.dat, IPARS.IN, and spe5.out

python spe5_ipars_EnKF.py
```

**What it does:**

1. Draws an initial ensemble of 3 log-permeability values.
2. Assimilates synthetic well production data at 10 time steps (365–3650 days).
3. Runs a forecast using the final analyzed ensemble.
4. Writes results to `./output/` (parameters, data, ensemble status per step).

**Key knobs** (bottom of `spe5_ipars_EnKF.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EnSize` | 40 | Number of ensemble members |
| `parallel_num` | 5 | Max concurrent IPARS runs |
| `obser_noise` | 0.05 | Observation error (5%) |

### 3. Run the level-set fracture example

```bash
cd examples
# Edit levelset_EnKF_MFD.py:
#   - Update the runt symlink path
# Edit runt/simulator_interface/MFD_experiment.py:
#   - Update the mesher.py symlink path

python levelset_EnKF_MFD.py
```

**What it does:**

1. Loads initial fracture parameter ensembles from `./input/`.
2. For each of 15 assimilation steps: meshes fractures, runs MFD flow, assimilates well rates.
3. Writes level-set, ρ, θ, fracture counts, and ensemble status to `./output/`.

**Key knobs** (bottom of `levelset_EnKF_MFD.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EnSize` | 100 | Ensemble size |
| `parallel_num` | 10 | Max concurrent mesher/MFD jobs |
| `length_threshold` | 0.2 | Minimum fracture length to keep |
| `obser_noise` | 0.05 | Observation error (5%) |

A **reload mode** (`DArun_reload`) re-reads existing simulation output without re-running the simulator — useful for debugging the analysis step.

---

## Output

Both examples write to `./output/`:

| File pattern | Contents |
|--------------|----------|
| `paramXX.dat` / `levelsetXX.dat`, `rhoXX.dat`, `thetaXX.dat` | Updated parameter ensemble at step `XX` |
| `dataXX.dat` | Simulated observation data at step `XX` |
| `ens_statusXX.dat` | Boolean flags for active ensemble members |
| `stepXX/ensYYY/` | Per-member simulator working directories |
| `observation.dat` | Reference observations (SPE5 example) |
| `FracNumberXX.dat` | Fracture count per ensemble member (level-set example) |

Previous runs are archived as `outputYYMMDDHHMM/` before each new run.

---

## Extending runt

To add a new problem:

1. **Simulator:** Subclass `SimulatorClass` (or `IPARS_interface`) — implement input writing, job launching, and output parsing.
2. **Analysis:** Use `EnKF` or subclass `AnalysisClass` for custom update schemes.
3. **Ensemble:** Use `EnsembleGeneratorParallel` or subclass for custom forward-propagation logic.
4. **DA workflow:** Subclass `DA_init2current` or `DA_levelset_fracture` — override `Obser_initialization()` and any problem-specific hooks.
5. **Driver script:** Compose the four components and call `DArun()` (see the examples for templates).

---

## Project layout

```
runt/
├── README.md
├── runt/
│   ├── assimilation/
│   │   ├── data_assimilation_class.py   # Abstract DA workflow
│   │   ├── DA_init2current.py           # Generic EnKF driver
│   │   └── DA_levelset_fracture.py      # Level-set fracture DA
│   ├── analysis/
│   │   ├── analysis_class.py            # Abstract analysis
│   │   ├── enkf.py                        # Standard EnKF
│   │   └── enkf_levelset.py              # Level-set EnKF
│   ├── ensemble_generator/
│   │   ├── ensemble_generator_class.py  # Abstract forward propagate
│   │   ├── ensemble_generator_parallel.py
│   │   └── ens_gen_fracture_parallel.py
│   └── simulator_interface/
│       ├── simulator_class.py           # Abstract simulator adapter
│       ├── ipars.py                       # IPARS adapter
│       └── MFD_experiment.py             # MFD + mesher adapter
└── examples/
    ├── spe5_ipars_EnKF.py               # SPE5 permeability matching
    └── levelset_EnKF_MFD.py             # Fracture characterization
```

---

## Known limitations

- **Python 2 only** — requires porting for Python 3 (`print` function, `except` syntax, integer division, etc.).
- **Unix shell commands** — uses `mv`, `mkdir`, `ln`, `cp` via `subprocess.call`; not portable to Windows without WSL or refactoring.
- **Hard-coded paths** — simulator binary locations and symlinks are environment-specific and must be edited before running.
- **No packaging** — no `setup.py`, `requirements.txt`, or `__init__.py` files; manual `PYTHONPATH` setup is required.
