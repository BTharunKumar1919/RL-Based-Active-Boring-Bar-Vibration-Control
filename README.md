# Active Vibration Control of a Rotary Boring Bar (Simulation)

A modular Python project comparing **no active control**, a **PID controller**, and a **Soft Actor-Critic (SAC)** reinforcement-learning controller on a simplified boring-bar vibration model. The project was developed for an AI in Manufacturing course project.

> **Scope:** this is a simulation study using an assumed single-mode oscillator. It is not a physically identified boring-bar model, a validated digital twin, or an experimental demonstration of chatter suppression.

## Results at a glance

The included `results/robust_evaluation_summary.csv` reports the current 84-condition evaluation:

| Controller | Mean steady-state RMS vibration | Mean reduction vs. no control | Conditions improved |
|---|---:|---:|---:|
| No control | 4.962 µm | 0.00% | Baseline |
| PID | 3.761 µm | 4.64% | 42.9% (36/84) |
| SAC | 3.410 µm | 14.10% | 100% (84/84) |

For the 24 conditions with disturbance frequencies outside the training range (205 and 215 Hz), the mean reduction was **6.82% for SAC** and **−6.55% for PID**. Negative reduction means vibration was higher than the no-control baseline.

These are results from the included model and evaluation grid, not evidence of real-machine performance. They are based on one saved SAC policy and one deterministic test grid; repeat-seed and physical validation remain future work. Results may change if the model is retrained or the scripts/configuration are changed.

## Model and assumptions

The plant is a single-degree-of-freedom mass–spring–damper surrogate:

$$
m\ddot{q} + c\dot{q} + kq
= F_{\text{disturbance}} + F_{\text{control}}
$$

- The nominal natural frequency is set to 174 Hz and damping ratio to 0.012.
- Modal mass is set to 1 kg as a normalization assumption, not a measured modal mass.
- The simulation time step is 0.0001 s (10 kHz update rate).
- Maximum generalized control force is assumed to be ±2 N; this is not a measured actuator rating.
- Disturbance is sinusoidal. It does **not** represent regenerative chatter or a complete cutting-process model.
- The state coordinate is a modal coordinate. A calibrated mode-shape factor would be needed to infer physical tool-tip displacement.
- The code uses semi-implicit Euler integration.
- The controller has access to the simulated displacement and velocity states; the current Python project does not implement a Kalman filter or noisy sensor model.

The 174 Hz value should be treated as a working modelling assumption unless and until its source and applicability to the target boring bar are verified.

## Repository layout

```text
config.py                  # plant, disturbance, training and evaluation settings
models.py                  # single-mode plant and sinusoidal disturbance
controllers.py              # no-control and PID controllers
env.py                      # Gymnasium environment for SAC
sac_training.py              # SAC model construction and training
experiments.py               # nominal/unseen tests, metrics and plots
run_baseline.py              # training + standard comparison entry point
tune_pid.py                  # grid search for PID gains
robust_evaluation.py         # 84-condition controller comparison
requirements.txt             # Python dependencies
results/                     # saved model, CSVs and figures
```

## Requirements

- Python 3.10 or 3.11 recommended
- Packages listed in `requirements.txt`

## Setup (Windows CMD)

Run these commands from the repository root:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the project

### 1. Run a short smoke test

This trains a small model and runs the standard experiments. It overwrites the default saved model and standard result files, so back up results first if you want to keep the current outputs.

```bat
python run_baseline.py --timesteps 10000
```

### 2. Train for 100,000 timesteps

```bat
python run_baseline.py --timesteps 100000 --seed 42
```

`run_baseline.py` defaults to **300,000 timesteps** if `--timesteps` is omitted. Training saves the model to `results/sac_vibration_model.zip`, then runs the standard nominal and unseen-condition experiments. Training again with the same path overwrites the model file.

### 3. Evaluate the saved model across 84 conditions

After training, run:

```bat
python robust_evaluation.py
```

This loads `results/sac_vibration_model.zip` and evaluates No control, PID and SAC over 84 combinations:

- Disturbance amplitudes: 0.5, 1.0 and 1.5 N
- Disturbance frequencies: 150, 160, 174, 185, 200, 205 and 215 Hz
- Disturbance phases: 0, π/2, π and 3π/2 radians

The evaluation duration is 0.2 s per condition. Steady-state RMS vibration is calculated over the final 80% of each simulated record. The frequency groups are labelled within-range (150–200 Hz) and outside-range (205–215 Hz), relative to the training frequency range configured in `config.py`.

Generated outputs:

- `results/robust_evaluation_by_condition.csv` — per-condition metrics
- `results/robust_evaluation_summary.csv` — aggregated metrics
- `results/robust_evaluation_rms.png` — RMS vibration plot
- `results/robust_evaluation_reduction.png` — reduction vs. no-control plot

### 4. Tune PID gains (optional)

```bat
python tune_pid.py
```

This runs a grid search and writes `results/pid_tuning_results.csv`. It overwrites that file if it already exists. The robust evaluation uses fixed gains (`Kp = 50000`, `Ki = 0`, `Kd = 20`) so PID is not retuned on the 84 evaluation conditions.

### 5. Run syntax checks

```bat
python -m py_compile config.py models.py controllers.py env.py sac_training.py experiments.py run_baseline.py tune_pid.py robust_evaluation.py
```

## Main output files

- `results/controller_metrics.csv` — nominal and unseen-condition controller metrics
- `results/controller_timeseries.csv` — time-series data for standard experiments
- `results/pid_tuning_results.csv` — PID grid-search results
- `results/robust_evaluation_by_condition.csv` and `results/robust_evaluation_summary.csv` — robustness evaluation
- `results/*.png` — plots generated by the experiments
- `results/sac_vibration_model.zip` — saved SAC policy used by the included evaluation, if included in the repository

Running training or evaluation may overwrite output files. Copy the `results` directory before a new run if you need to preserve a particular model/result set. The CSVs and plots are snapshots; they will only correspond to a given model if they were generated using that model.

## Interpretation and limitations

- A test condition outside the training frequency range is not the same as testing on a different physical machine or validating against real cutting data.
- The 84 test cases are a deterministic grid, not 84 independent physical experiments.
- The reported percentage improved counts conditions where the controller's steady-state RMS was lower than the no-control result for the same condition. It does not measure statistical significance.
- A single training seed is insufficient to establish that performance is repeatable. Evaluate multiple seeds and report variation before making strong claims.
- PID and SAC use the simulated state variables in this implementation. No Kalman estimator is included in these modular scripts.
- The actuator is an ideal force source; actuator bandwidth, delay, force-current calibration, and slew-rate constraints are not modelled.
- No hardware or real machining tests are included.

## Future work

1. Verify model parameters using measured frequency-response/modal data from the target boring bar.
2. Add a regenerative chatter model and cutting-process dynamics.
3. Model actuator bandwidth, delays, limits and force-current calibration.
4. Add sensor noise and a state estimator if required by the intended control architecture.
5. Compare controllers under matched information and tuning budgets.
6. Repeat training across several random seeds and evaluate on additional held-out conditions.
7. Validate against experimental measurements before making real-machine performance claims.

## Reproducibility note

The scripts, settings, saved policy and result files in this repository represent one snapshot of the project. To reproduce the reported numbers, use the included saved model with `python robust_evaluation.py` and retain the same source/configuration files. Retraining creates a new model and can produce different results.