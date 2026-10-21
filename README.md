# Active Vibration Control of a Rotary Boring Bar

A modular Python simulation project comparing **No Control**, a **PID controller**, and a **Soft Actor-Critic (SAC)** reinforcement-learning controller for active vibration suppression in a simplified rotary boring-bar model.

The project includes a multi-mode plant and actuator model, disturbance/noise modelling, SAC training and evaluation, configurable robust evaluation, a FastAPI backend, and an interactive Streamlit dashboard.

> **Scope:** This is an educational simulation framework using assumed plant, actuator, and disturbance parameters. It is not an experimentally identified boring-bar model, a validated digital twin, or a demonstration of chatter suppression on physical machinery.

## Features

- **Three-controller comparison:** No Control, PID, and SAC.
- **Multi-mode plant:** Six current-driven structural modes and one force-driven mode.
- **Actuator model:** Current-commanded control with actuator bandwidth, current slew-rate, and current saturation limits.
- **Reinforcement learning:** Train and evaluate SAC using Stable-Baselines3.
- **Specified SAC training:** Train a policy for a fixed operating condition.
- **Generalized SAC training:** Randomize selected operating and plant parameters independently across episodes.
- **Harmonic disturbance model:** Fundamental cutting force plus optional 2f–6f harmonics with configurable amplitudes and phases.
- **Noise modelling:** Separate cutting-force noise and plant process noise.
- **Configurable plant variation:** Perturb physical plant and actuator parameters within bounded percentage ranges.
- **Robustness evaluation:** Compare controllers over configurable amplitude, frequency, and phase grids.
- **Interactive dashboard:** Configure plant, disturbance, PID, SAC training, controller selection, simulations, and robust evaluation from Streamlit.
- **Model library:** Automatically detect saved SAC models in `results/`, select an active model, or use a model from another directory.
- **Model metadata:** Save a JSON configuration alongside each trained SAC model.
- **REST API:** Run simulations and controller comparisons through FastAPI.
- **Exportable results:** Download time-series data, controller metrics, and robust-evaluation results as CSV files.
- **Reproducible configuration:** Centralized plant, disturbance, training, reward, and evaluation settings.

## Results at a glance

The project is designed to evaluate vibration suppression together with actuator effort rather than relying on vibration reduction alone.

The robust evaluator reports, for each controller:

- Mean, median, best-case, and worst-case steady-state RMS vibration.
- Mean peak vibration.
- Mean RMS control current.
- Mean actuator saturation fraction.
- Mean, best-case, and worst-case vibration reduction relative to No Control.
- Fraction of tested conditions improved relative to No Control.
- A composite robust score that combines vibration reduction and current usage.

The repository may contain saved result files from a particular training run. These results are **configuration- and model-specific** and should be regenerated after changing the plant, controller, reward function, training distribution, or SAC model.

No numerical result in this README should be interpreted as a fixed project-wide benchmark. The purpose of the framework is to make controller comparisons reproducible for a clearly specified simulation configuration.

## System architecture

```text
                         Configuration
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
      Plant / actuator                  Cutting-force model
             |                         + harmonics + noise
             +----------------+----------------+
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
            No Control       PID           SAC
                |             |             |
                +-------------+-------------+
                              |
                              v
                    Simulation time series
                              |
                              v
                    Performance metrics
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Robust evaluation          FastAPI service
                 |                         |
                 +------------+------------+
                              |
                              v
                       Streamlit dashboard
                              |
                     Plots, metrics, CSVs
```

The dashboard communicates with the FastAPI backend. The backend uses the shared simulation/deployment service so that normal comparisons and robust evaluation use the same controller and plant implementation. SAC inference loads a saved policy; training is performed explicitly through the SAC training controls rather than during normal simulation requests.

## Plant and actuator model

The current plant is a seven-mode model:

- **Six current-driven modes:** control current directly excites the modal dynamics through a mode-specific `k_yI` coefficient.
- **One force-driven mode:** the cutting disturbance enters through a dedicated force-driven structural mode.

Nominal current-driven modes:

| Mode | Natural frequency | Damping ratio | `k_yI` |
|---|---:|---:|---:|
| 1 | 51 Hz | 0.03 | -2.0e6 |
| 2 | 67 Hz | 0.08 | -4.0e5 |
| 3 | 98 Hz | 0.03 | -9.0e6 |
| 4 | 138 Hz | 0.08 | -8.0e5 |
| 5 | 273 Hz | 0.05 | +4.0e6 |
| 6 | 450 Hz | 0.05 | -9.5e5 |

Nominal force-driven mode:

- Natural frequency: **188 Hz**
- Damping ratio: **0.034**
- Stiffness: **2.6e6 N/m**

The plant integrates at **0.0001 s (10 kHz)**.

The control interface is current-based:

- Continuous current limit: **±5 A**
- Peak current limit: **±15 A**
- Current-loop bandwidth: **8000 Hz**
- Effective actuator bandwidth: **800 Hz**
- Current slew-rate limit: **10,000 A/s**

The SAC action is normalized to `[-1, 1]` and mapped to the ±5 A continuous-current command range. The actuator dynamics and slew-rate limit are applied before the current enters the modal plant.

The model parameters are assumptions and should not be interpreted as experimentally identified values.

## Disturbance model

The disturbance model consists of a sinusoidal fundamental cutting force:

$$
F_1(t)=A\sin(2\pi f t+\phi)
$$

Optional Fourier harmonics can be added at 2f through 6f. Harmonic amplitudes are specified as fractions of the fundamental amplitude, with independent harmonic phases.

The dashboard can also add **cutting-force noise**, which is separate from plant process noise.

### Noise models

Two different noise sources are implemented:

- **Cutting-force noise:** additive force noise applied to the cutting-force disturbance.
- **Process noise:** additive force noise applied to the force-driven plant mode.

The two noise sources have independent configuration and seeds. They should not be confused with sensor noise: the current implementation does **not** simulate noisy measurements or a state estimator.

## SAC observation and control timing

The SAC policy does not operate directly on a single instantaneous state. Its observation consists of **eight consecutive observations**.

Each observation contains:

```text
[normalized displacement,
 normalized velocity,
 normalized actual actuator current]
```

The eight observations provide **4 ms of history** because the SAC command is updated every five physical plant steps.

The timing is:

```text
Plant integration:       0.1 ms
SAC action interval:     5 plant steps = 0.5 ms
Observation history:     8 SAC observations = 4 ms
```

The action is held for the configured action-repeat interval while the physical plant continues integrating at 10 kHz.

## SAC training modes

The dashboard supports two main training modes.

### Specified-condition training

The policy is trained using a fixed operating condition. The selected amplitude, frequency, phase, harmonics, noise levels, and plant configuration remain fixed during training unless explicitly changed by the user.

This mode is useful for studying a controller specialized to a particular operating condition.

### Generalized / randomized training

Selected parameters can instead be randomized independently between episodes. Current continuous randomization options include:

- Disturbance amplitude.
- Disturbance frequency.
- Fundamental phase.
- Cutting-force noise level.
- Process-noise level.
- Plant and actuator parameters.

The user can choose which parameters participate in randomization and specify their ranges. Generalized training samples operating conditions rather than evaluating every Cartesian combination on a grid.

Harmonic randomization is also supported by the training configuration. Harmonic count can be randomized explicitly when enabled; otherwise the selected harmonic structure remains fixed.

### Reproducibility

Training uses an explicit seed. The saved model is accompanied by a JSON metadata file containing the training mode, plant variations, disturbance configuration, randomization settings, SAC hyperparameters, timing settings, and other requested training parameters.

## SAC reward

The training reward combines vibration, control effort, and action-change penalties:

$$
R = -w_x E_x - w_u E_u - w_{\Delta u}E_{\Delta u}
$$

where the weights are configured in `TrainingConfig` in `config.py`.

Current defaults are:

- Displacement/vibration weight: **1.0**
- Control-effort weight: **0.10**
- Action-change weight: **0.05**

These weights are not universal optimum values. Changing them changes the trade-off the SAC policy learns between vibration suppression, current usage, and command smoothness. Any model retrained after changing the reward weights should be treated as a different experiment.

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
    ├── *.zip                         Saved SAC models
    ├── *.json                        SAC training metadata
    ├── robust_evaluation_*.csv       Robust evaluation data
    ├── robust_evaluation_*.png       Robust evaluation plots
    └── other figures and CSV outputs
```

`results/` may contain additional saved models, metadata, CSV files, plots, and TensorBoard logs depending on which training and evaluation workflows have been run.

### Main modules

| File | Purpose |
|---|---|
| `config.py` | Central plant, disturbance, training, reward, and evaluation configuration |
| `models.py` | Seven-mode plant, actuator dynamics, disturbance model, noise, and plant-variation construction |
| `controllers.py` | No Control and PID controller implementations |
| `env.py` | Gymnasium environment used for SAC training, including observation history and episode randomization |
| `sac_training.py` | SAC construction, specified/generalized training, saving, and metadata generation |
| `experiments.py` | Controller simulation, time-series generation, and performance metrics |
| `run_baseline.py` | Standard training/comparison entry point |
| `tune_pid.py` | PID gain tuning utility |
| `robust_evaluation.py` | Standalone robust evaluation over the default evaluation grid |
| `deployment_service.py` | Shared simulation and robust-evaluation service used by the API/dashboard |
| `api.py` | FastAPI endpoints for simulation, comparison, training, and model management |
| `app.py` | Streamlit interactive dashboard |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or 3.11 recommended.
- Dependencies listed in `requirements.txt`.
- Stable-Baselines3 and Gymnasium are required for SAC training/inference.
- The FastAPI and Streamlit components are required for the interactive application.
- A saved SAC `.zip` model is required when running SAC inference.

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

### 1. Train SAC from the command line

A short smoke test:

```bat
.venv\Scripts\python.exe run_baseline.py --timesteps 10000
```

A longer training run:

```bat
.venv\Scripts\python.exe run_baseline.py --timesteps 300000 --seed 42
```

The dashboard provides more extensive training controls, including specified/generalized training, plant variation, disturbance randomization, harmonics, noise, SAC hyperparameters, and model naming.

Training saves the policy as a `.zip` file and writes a matching `.json` metadata file when using the configurable SAC training workflow.

> **Warning:** Training can overwrite existing result files or model names. Use descriptive model names and preserve any model/result files that are needed for comparison.

### 2. Evaluate the saved policy over the default robust grid

```bat
.venv\Scripts\python.exe robust_evaluation.py
```

The default standalone evaluation uses:

- Disturbance amplitudes: **0.5, 1.0, 1.5 N**.
- Disturbance frequencies: **150, 160, 174, 185, 200, 205, 215 Hz**.
- Disturbance phases: **0, π/2, π, 3π/2 rad**.

This gives **3 × 7 × 4 = 84 conditions per controller**.

Each condition runs for 0.2 s, with the final 80% used for steady-state metrics.

The Streamlit dashboard additionally allows the amplitude, frequency, and phase ranges and their number of divisions to be changed for custom robust evaluations.

Typical outputs include:

- `results/robust_evaluation_by_condition.csv`
- `results/robust_evaluation_summary.csv`
- Robust RMS and vibration-reduction plots
- Additional case-level and summary outputs depending on the selected configuration

### 3. Start the FastAPI backend

In the first terminal:

```bat
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

The default local API address is:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

The API provides endpoints for service health, model information, simulation, controller comparison, SAC training, and model management. Exact endpoint availability should be checked against the current `api.py` because the application interface evolves with the project.

### 4. Start the Streamlit dashboard

Keep the FastAPI server running. Open a second terminal in the project directory:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL in the terminal, usually:

`http://localhost:8501`

The dashboard supports:

- Plant and actuator parameter variation within bounded percentages.
- Process-noise configuration.
- Fundamental cutting-force amplitude, frequency, and phase.
- 2f–6f harmonics.
- Cutting-force noise.
- PID gain and integral-limit controls.
- SAC specified-condition and generalized training.
- SAC hyperparameter controls.
- Training-model naming and metadata saving.
- Automatic discovery of SAC models in `results/`.
- Selection of a saved model or use of a model from another directory.
- A single active SAC model used by subsequent simulations/evaluations.
- Normal controller comparison.
- Configurable robust evaluation ranges and plots.
- Downloadable simulation and evaluation CSV files.

The active SAC model is displayed in the dashboard together with its resolved file path so that the exact model used for inference can be verified.

## Model selection

The dashboard treats one SAC model as the authoritative active model for inference.

Models saved in the project's `results/` directory are automatically discovered and presented by model name rather than by long file path. A refresh operation can be used after adding new model files.

For a model stored elsewhere, the dashboard provides an external-model selection option. The active model path is displayed separately so the exact file used by the simulation can be verified.

## PID control

The dashboard exposes direct PID parameters:

- `Kp`
- `Ki`
- `Kd`
- Integral error limit

PID gains are applied consistently to normal comparisons and robust evaluation. The PID controller is a benchmark controller; its performance depends on the selected gains and should not be treated as an automatically optimal baseline unless a separate tuning procedure has been performed.

## Interpreting the metrics

The simulation reports vibration and actuator-effort metrics using the actual actuator current where applicable.

- **Steady-state RMS vibration:** RMS displacement over the configured steady-state portion of the response.
- **Overall RMS vibration:** RMS displacement over the full simulated response.
- **Peak vibration:** Maximum absolute displacement.
- **RMS control current:** Root-mean-square actual actuator current.
- **Peak control current:** Maximum absolute actual actuator current.
- **Near-saturation fraction:** Fraction of samples close to the continuous current limit.
- **Hard-saturation fraction:** Fraction of samples at or effectively at the continuous current limit.
- **Vibration reduction:** Percentage reduction relative to the matched No Control condition.

$$
\text{Vibration reduction}(\%) =
100\left(1-\frac{\mathrm{RMS}_{\mathrm{controller}}}
{\mathrm{RMS}_{\mathrm{No\ Control}}}\right)
$$

Positive values indicate lower vibration than No Control. Negative values indicate higher vibration than the No Control baseline.

A lower vibration metric does not automatically imply a better practical controller. Control effort, saturation, robustness, model validity, and reproducibility must also be considered.

## Robust evaluation

The robust evaluator runs the same disturbance condition for each controller so that comparisons use matched operating conditions and noise realizations where noise is enabled.

The dashboard can construct a custom Cartesian evaluation grid from:

- Amplitude minimum/maximum and number of divisions.
- Frequency minimum/maximum and number of divisions.
- Phase minimum/maximum and number of divisions.
- Duration per condition.

If `N` divisions are selected, the range contains `N + 1` equally spaced values including both endpoints.

The dashboard limits the total number of conditions to prevent accidentally launching an excessively large evaluation.

The robust plots include amplitude/frequency heatmaps, frequency trends, and best/mean/worst-case summaries. Heatmaps aggregate across phase when multiple phase values are present.

## Limitations and interpretation

- The model parameters are assumed rather than experimentally identified.
- The seven-mode model is a simplified representation and is not a validated structural model of a particular boring bar.
- The disturbance model is harmonic forcing; it is not a full regenerative chatter or cutting-process model.
- Sensor measurement noise is not currently modelled.
- A Kalman filter or other state estimator is not currently implemented.
- Process and cutting-force noise are simulation disturbances, not measured machine noise.
- The actuator is an assumed current-driven model. Real actuator force-current calibration, electrical dynamics, delays, and hardware-specific limits require experimental identification.
- Robust evaluation uses deterministic test grids unless stochastic noise is explicitly enabled.
- A generalized SAC policy trained over parameter ranges is not automatically guaranteed to generalize outside those ranges.
- A frequency outside the SAC training range is not equivalent to validation on a different physical machine.
- A single training seed is insufficient to establish statistical reproducibility.
- The reported improved-condition percentage is not a statistical significance test.
- Simulation performance does not establish physical-machine performance.
- The dashboard does not connect to physical sensors, actuators, or a live boring-bar machine.

## Future work

1. Verify plant parameters using measured modal or frequency-response data from the target boring bar.
2. Validate the seven-mode model against experimental measurements.
3. Add realistic sensor noise and observation/estimation modelling.
4. Develop a regenerative chatter and cutting-process model.
5. Identify actuator bandwidth, delay, slew-rate, current-force calibration, and saturation from hardware data.
6. Improve PID tuning and compare PID and SAC under matched tuning and information budgets.
7. Study SAC reward-weight selection and the vibration-versus-control-effort trade-off systematically.
8. Train and evaluate SAC across multiple random seeds and report uncertainty.
9. Study generalized training distributions and out-of-distribution operating conditions.
10. Validate simulation predictions with controlled experimental machining tests before making physical performance claims.

## Reproducibility

A trained SAC model should be kept together with its matching metadata JSON and the source/configuration snapshot used to create it.

For a saved robust-evaluation result, preserve:

- The SAC `.zip` model.
- The corresponding SAC `.json` metadata.
- `config.py` and relevant source files.
- Robust-evaluation CSV outputs.
- Any custom plant/disturbance configuration used for the experiment.

Retraining a SAC model can change its behaviour even with the same nominal configuration. Different seeds, training distributions, reward weights, and hyperparameters represent different experiments.

## Contributing

Contributions are welcome. Keep changes relevant to the project, test them before submission, document new dependencies and modelling assumptions, and update evaluation results when changing plant, controller, or training behaviour.

Do not present simulation results as experimental validation on physical machinery.
