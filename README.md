# Active Vibration Control of a Rotary Boring Bar

A modular Python simulation project comparing **No Control**, a **PID controller**, and a **Soft Actor-Critic (SAC)** reinforcement-learning controller for active vibration suppression in a simplified rotary boring-bar model.

The project includes a simulation environment, reinforcement-learning training and evaluation scripts, a FastAPI backend, and an interactive Streamlit dashboard.

> **Scope:** This is an educational simulation using an assumed single-mode oscillator. It is not an experimentally identified boring-bar model, a validated digital model, or a demonstration of chatter suppression on physical machinery.

## Features

- **Three-controller comparison:** No Control, PID, and SAC.
- **Reinforcement learning:** Train and evaluate a SAC policy using Stable-Baselines3.
- **Robustness evaluation:** Compare controllers across a deterministic grid of disturbance amplitudes, frequencies, and phases.
- **Interactive dashboard:** Explore displacement, velocity, disturbance force, control force, vibration metrics, and actuator effort.
- **REST API:** Run individual simulations and compare controllers using FastAPI.
- **Exportable results:** Download simulation time series and controller metrics as CSV files.
- **Reproducible configuration:** Centralized plant, training, disturbance, and evaluation settings.

## Results at a glance

The included `results/robust_evaluation_summary.csv` reports the current 84-condition evaluation.

| Controller | Mean steady-state RMS vibration | Mean reduction vs. No Control | Conditions improved |
|---|---:|---:|---:|
| No Control | 4.962 µm | 0.00% | Baseline |
| PID | 3.761 µm | 4.64% | 42.9% (36/84) |
| SAC | 3.410 µm | 14.10% | 100% (84/84) |

For the 24 conditions with disturbance frequencies outside the SAC training range (205 and 215 Hz), the mean reduction was **6.82% for SAC** and **−6.55% for PID**. Negative reduction means vibration was higher than the No Control baseline.

These results describe the included simulation and evaluation grid. They do not establish real-machine performance. The results use one saved SAC policy and one deterministic test grid; multi-seed and physical validation remain future work. Results can change if the model, configuration, controller, or training process changes.

## System architecture

```text
                    Simulation configuration
                              |
                              v
                   Single-mode plant model
                              |
                 Harmonic disturbance input
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         No Control          PID              SAC
             |                |                |
             +----------------+----------------+
                              |
                              v
                    Simulation time series
                              |
                              v
                   Performance calculations
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                FastAPI API       Streamlit UI
                    |                   |
                    +---------+---------+
                              |
                   Plots, metrics and CSVs
```

The dashboard sends requests to the API. The API calls the shared simulation service, which uses the existing plant and controller implementation. SAC inference loads a saved model; it does not train the policy during dashboard requests.

## Model and assumptions

The plant is a single-degree-of-freedom mass–spring–damper surrogate:

$$
m\ddot{q}+c\dot{q}+kq
=F_{\text{disturbance}}+F_{\text{control}}
$$

Current assumptions:

- Nominal natural frequency: 174 Hz.
- Damping ratio: 0.012.
- Modal mass: 1 kg, used as a normalization assumption rather than a measured modal mass.
- Simulation time step: 0.0001 s, corresponding to a 10 kHz simulation update rate.
- Maximum generalized control force: ±2 N, an assumed limit rather than a measured actuator rating.
- Disturbance: sinusoidal force, not a regenerative chatter or complete cutting-process model.
- Integration: semi-implicit Euler.
- The simulated state coordinate is modal displacement. A calibrated mode-shape factor is needed to infer physical tool-tip displacement.

The 174 Hz natural frequency should be treated as a working assumption until its source and applicability to the target boring bar are verified.

### Current noise and uncertainty limitations

The current baseline simulation does **not** implement:

- Sensor measurement noise.
- Randomized plant parameters within each simulation run.
- A Kalman filter or other state estimator.
- Randomly varying cutting-force dynamics.
- Real actuator bandwidth, delay, slew-rate limits, or force-current calibration.

The dashboard currently plots the deterministic simulated response for the selected operating condition. Noise and uncertainty modelling are planned extensions, not existing capabilities.

## Repository layout

```text
.
├── config.py
├── models.py
├── controllers.py
├── env.py
├── sac_training.py
├── experiments.py
├── run_baseline.py
├── tune_pid.py
├── robust_evaluation.py
├── deployment_service.py
├── api.py
├── app.py
├── requirements.txt
├── README.md
└── results/
    ├── sac_vibration_model.zip
    ├── robust_evaluation_by_condition.csv
    ├── robust_evaluation_summary.csv
    └── figures and other evaluation outputs
```

`results/` may contain additional CSV files, plots, and saved model files depending on which scripts have been run.

### Main modules

| File | Purpose |
|---|---|
| `config.py` | Plant, disturbance, training, and evaluation configuration |
| `models.py` | Single-mode plant and sinusoidal disturbance |
| `controllers.py` | No Control and PID controllers |
| `env.py` | Gymnasium environment used for SAC training |
| `sac_training.py` | SAC model construction and training |
| `experiments.py` | Controller simulation, metrics, and standard experiments |
| `run_baseline.py` | Training and standard comparison entry point |
| `tune_pid.py` | PID gain grid search |
| `robust_evaluation.py` | Deterministic 84-condition evaluation |
| `deployment_service.py` | Shared simulation service for API requests |
| `api.py` | FastAPI endpoints |
| `app.py` | Streamlit interactive dashboard |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or 3.11 recommended.
- Dependencies listed in `requirements.txt`.
- The saved SAC model file is required for SAC inference.
- The API and dashboard run locally by default.

## Installation

Run these commands from the project directory.

### Windows CMD

Create a virtual environment if one does not already exist:

```bat
python -m venv .venv
```

Install the dependencies:

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using `.venv\Scripts\python.exe` explicitly helps ensure packages are installed into and executed from the project's virtual environment.

### macOS and Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Run the project

### 1. Train the SAC policy and run standard experiments

A short smoke test:

```bat
.venv\Scripts\python.exe run_baseline.py --timesteps 10000
```

A longer training run:

```bat
.venv\Scripts\python.exe run_baseline.py --timesteps 100000 --seed 42
```

If `--timesteps` is omitted, `run_baseline.py` defaults to 300,000 timesteps. Training saves the policy to `results/sac_vibration_model.zip` and runs the standard experiments.

**Warning:** Training can overwrite the existing model and standard result files. Back up any results you want to preserve before retraining.

### 2. Evaluate the saved policy over 84 conditions

```bat
.venv\Scripts\python.exe robust_evaluation.py
```

The evaluation compares No Control, PID, and SAC across:

- Disturbance amplitudes: 0.5, 1.0, and 1.5 N.
- Disturbance frequencies: 150, 160, 174, 185, 200, 205, and 215 Hz.
- Disturbance phases: 0, π/2, π, and 3π/2 radians.

This gives 3 × 7 × 4 = 84 conditions per controller.

Each condition runs for 0.2 seconds. Steady-state RMS vibration is calculated over the final 80% of the simulated record. Frequencies from 150–200 Hz are within the configured SAC training range; 205 and 215 Hz are outside it.

Outputs include:

- `results/robust_evaluation_by_condition.csv`
- `results/robust_evaluation_summary.csv`
- `results/robust_evaluation_rms.png`
- `results/robust_evaluation_reduction.png`

### 3. Start the FastAPI backend

In the first terminal:

```bat
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

The default local API address is:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Service and SAC model status |
| `GET` | `/model` | Saved-model availability |
| `POST` | `/simulate` | Simulate one controller |
| `POST` | `/compare` | Compare all three controllers |

The API validates request parameters and returns simulation metrics and time-series data. It is intended for local educational use and is not a safety-certified industrial control API.

### 4. Start the Streamlit dashboard

Keep the FastAPI server running. Open a second terminal in the project directory:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL in the terminal, usually:

`http://localhost:8501`

The dashboard allows you to:

- Set disturbance amplitude, frequency, phase, and duration.
- Run No Control, PID, and SAC under the same selected disturbance condition.
- Compare displacement and velocity time histories.
- Inspect disturbance force and controller force.
- Compare steady-state RMS, overall RMS, peak displacement, RMS force, peak force, and actuator saturation.
- See percentage steady-state RMS reduction relative to No Control.
- Identify whether the selected frequency lies inside or outside the SAC training range.
- Download individual controller traces, all traces, and comparison metrics as CSV files.

The dashboard communicates with the API at `http://127.0.0.1:8000` by default. The API URL can be changed in the dashboard sidebar.

### 5. Tune PID gains (optional)

```bat
.venv\Scripts\python.exe tune_pid.py
```

This runs the configured grid search and writes `results/pid_tuning_results.csv`. It overwrites that file if it already exists.

The robust evaluation uses fixed PID gains (`Kp = 50000`, `Ki = 0`, `Kd = 20`); PID is not retuned separately for each of the 84 evaluation conditions.

### 6. Run syntax checks

```bat
.venv\Scripts\python.exe -m py_compile config.py models.py controllers.py env.py sac_training.py experiments.py run_baseline.py tune_pid.py robust_evaluation.py deployment_service.py api.py app.py
```

## Interpreting the metrics

- **Steady-state RMS vibration:** RMS displacement over the configured steady-state portion of the response.
- **Overall RMS vibration:** RMS over the full simulated response.
- **Peak vibration:** Maximum absolute displacement.
- **RMS control force:** Root-mean-square generalized force commanded by the controller.
- **Peak control force:** Maximum absolute generalized control force.
- **Actuator saturation fraction:** Fraction of samples at the actuator force limit.
- **RMS reduction:** Percentage reduction relative to the matched No Control condition.

\[
\text{RMS reduction}(\%) =
100\left(1-\frac{\mathrm{RMS}_{\mathrm{controller}}}
{\mathrm{RMS}_{\mathrm{no\ control}}}\right)
\]

Positive values indicate lower RMS vibration than No Control; negative values indicate worse performance.

A lower vibration metric does not automatically imply a better practical controller. Control effort, saturation, robustness, and the validity of the plant model must also be considered.

## Limitations and interpretation

- A frequency outside the training range is not equivalent to validation on a different physical machine.
- The 84 conditions form a deterministic grid, not 84 independent physical experiments.
- The reported improved-condition percentage is not a statistical significance test.
- A single training seed is insufficient to establish reproducibility.
- The modular scripts use simulated displacement and velocity states; they do not include a Kalman estimator or noisy sensor model.
- The actuator is an ideal generalized force source. Bandwidth, delay, slew rate, and force-current calibration are not modelled.
- The dashboard shows simulation results only. It does not connect to physical sensors, actuators, or a live boring-bar machine.
- No hardware or real machining tests are included.

## Future work

1. Verify model parameters using measured modal or frequency-response data from the target boring bar.
2. Add sensor noise and randomized plant parameters, with repeatable random seeds.
3. Train and evaluate SAC under realistic observation noise if the intended deployment includes noisy sensors.
4. Add a regenerative chatter model and cutting-process dynamics.
5. Model actuator bandwidth, delays, limits, and force-current calibration.
6. Compare PID and SAC under matched tuning and information budgets.
7. Repeat training across multiple random seeds and report uncertainty in evaluation results.
8. Validate the simulation against experimental measurements before making physical performance claims.

## Reproducibility

The saved model, source files, configuration, and result files represent a particular project snapshot. To reproduce the included robust-evaluation results, use the saved `results/sac_vibration_model.zip` with the matching source and configuration files:

```bat
.venv\Scripts\python.exe robust_evaluation.py
```

Retraining the model may change results. Keep a copy of the model and associated CSV outputs if you need to preserve a specific experiment.

## Contributing

Contributions are welcome. Keep changes relevant to the project, test them before submission, document new dependencies and modelling assumptions, and update evaluation results when changing controller behaviour or training configurations.

Do not present simulation results as experimental validation on physical machinery.
