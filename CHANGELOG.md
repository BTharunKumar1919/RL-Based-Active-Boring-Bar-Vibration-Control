# Project update notes

## Configurable experiment GUI extension

- Preserved the existing Streamlit + FastAPI architecture and original `/simulate` and `/compare` workflow.
- Added plant presets and per-mode parameter offsets.
- Added sinusoidal, multi-harmonic, modulated, and mixed cutting-force models.
- Added independent cutting-force, process-force, and displacement-sensor noise controls.
- Added a metadata-backed SAC policy registry, individual/generalized training modes, background progress/logging, and cancellation with partial-checkpoint registration.
- Added configurable simulation and training endpoints, CSV export from the GUI, updated setup instructions, and unit tests.

Validation performed in the packaging environment: Python syntax compilation passed; all 5 experiment-engine tests passed. Full FastAPI/SAC end-to-end execution was not run because Stable-Baselines3 is not installed in the packaging environment; install `requirements.txt` before launching the app.
