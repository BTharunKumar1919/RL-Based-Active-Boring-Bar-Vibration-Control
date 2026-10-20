"""Streamlit dashboard for simulated active boring-bar vibration control.

The dashboard communicates with api.py. It does not control physical machinery.
"""

import requests
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Active Boring-Bar Vibration Control",
    page_icon="📉",
    layout="wide",
)


API_BASE = st.sidebar.text_input(
    "API URL",
    value="http://127.0.0.1:8000",
).strip().rstrip("/")


CONTROLLERS = ["No control", "PID", "SAC"]

COLORS = {
    "No control": "#7f8c8d",
    "PID": "#e67e22",
    "SAC": "#2471a3",
}


PLANT_VARIATION_LIMIT_PERCENT = 20.0

NOMINAL_CURRENT_MODES = [
    (51.0, 0.03, -2.0e6),
    (67.0, 0.08, -4.0e5),
    (98.0, 0.03, -9.0e6),
    (138.0, 0.08, -8.0e5),
    (273.0, 0.05, 4.0e6),
    (450.0, 0.05, -9.5e5),
]

NOMINAL_FORCE_MODE = (188.0, 0.034, 2.6e6)
NOMINAL_ACTUATOR = (800.0, 10000.0)


def api_get(path):
    response = requests.get(f"{API_BASE}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload, timeout=60):
    response = requests.post(
        f"{API_BASE}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def metric_value(metrics, key):
    value = metrics.get(key)

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def make_comparison_table(results):
    rows = []

    for result in results:
        rows.append({
            "Controller": result["controller"],

            "Steady RMS (µm)": metric_value(
                result,
                "steady_rms_vibration_um",
            ),

            "Overall RMS (µm)": metric_value(
                result,
                "rms_vibration_um",
            ),

            "Peak vibration (µm)": metric_value(
                result,
                "peak_vibration_um",
            ),

            "RMS current (A)": metric_value(
                result,
                "rms_control_current_a",
            ),

            "Peak current (A)": metric_value(
                result,
                "peak_control_current_a",
            ),

            "Saturation (%)": 100 * metric_value(
                result,
                "actuator_saturation_fraction",
            ),
        })

    return pd.DataFrame(rows)


def make_robust_summary_table(summary):
    rows = []
    for item in summary:
        rows.append({
            "Controller": item["controller"],
            "Mean steady RMS (µm)": item["mean_steady_rms_vibration_um"],
            "Best-case steady RMS (µm)": item["best_case_steady_rms_vibration_um"],
            "Worst-case steady RMS (µm)": item["worst_case_steady_rms_vibration_um"],
            "Mean peak (µm)": item["mean_peak_vibration_um"],
            "Mean RMS current (A)": item["mean_rms_control_current_a"],
            "Mean saturation (%)": item["mean_hard_saturation_percent"],
            "Mean vibration reduction (%)": item["mean_vibration_reduction_percent"],
            "Best vibration reduction (%)": item["best_vibration_reduction_percent"],
            "Worst vibration reduction (%)": item["worst_vibration_reduction_percent"],
            "Success rate (%)": item["success_rate_percent"],
            "Robust score": item["composite_robust_score"],
        })
    return pd.DataFrame(rows)



def make_robust_heatmap(cases, controller, value_column, title, colorbar_title):
    subset = cases[cases["controller"] == controller].copy()
    pivot = subset.pivot_table(
        index="amplitude_n",
        columns="frequency_hz",
        values=value_column,
        aggfunc="mean",
    ).sort_index()

    fig = go.Figure(
        go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.to_numpy(),
            colorbar={"title": colorbar_title},
            hovertemplate=(
                "Frequency: %{x:.1f} Hz<br>"
                "Amplitude: %{y:.3g} N<br>"
                f"{colorbar_title}: %{{z:.4g}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Disturbance frequency (Hz)",
        yaxis_title="Cutting-force amplitude (N)",
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


def make_robust_frequency_plot(cases, value_column, title, y_label):
    grouped = (
        cases.groupby(["controller", "frequency_hz"], as_index=False)[value_column]
        .mean()
        .sort_values("frequency_hz")
    )

    fig = go.Figure()
    for controller in CONTROLLERS:
        data = grouped[grouped["controller"] == controller]
        if data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=data["frequency_hz"],
                y=data[value_column],
                name=controller,
                mode="lines+markers",
                line={"color": COLORS[controller], "width": 2},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Disturbance frequency (Hz)",
        yaxis_title=y_label,
        hovermode="x unified",
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


def make_robust_best_mean_worst_plot(summary, mean_col, best_col, worst_col, title, y_label):
    rows = []
    for item in summary:
        rows.append({
            "Controller": item["controller"],
            "Best": item[best_col],
            "Mean": item[mean_col],
            "Worst": item[worst_col],
        })
    frame = pd.DataFrame(rows)
    long = frame.melt("Controller", var_name="Statistic", value_name="Value")

    fig = px.bar(
        long,
        x="Controller",
        y="Value",
        color="Statistic",
        barmode="group",
        title=title,
        text_auto=".4g",
    )
    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title="Controller",
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


def make_trace_plot(traces, y_column, title, y_label):
    fig = go.Figure()

    for controller in CONTROLLERS:
        if controller not in traces:
            continue

        data = traces[controller]

        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data[y_column],
                name=controller,
                mode="lines",
                line={
                    "color": COLORS[controller],
                    "width": 1.8,
                },
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Controller",
        margin={
            "l": 15,
            "r": 15,
            "t": 60,
            "b": 15,
        },
    )

    return fig


def make_force_disturbance_plot(traces):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Applied disturbance force",
            "Controller current",
        ),
    )

    for controller in CONTROLLERS:
        if controller not in traces:
            continue

        data = traces[controller]
        color = COLORS[controller]

        # External cutting-force disturbance
        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["disturbance_force_n"],
                name=f"{controller} disturbance",
                line={
                    "color": color,
                    "width": 1.5,
                },
                legendgroup=controller,
            ),
            row=1,
            col=1,
        )

        # Actual actuator current
        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["control_current_a"],
                name=f"{controller} control",
                line={
                    "color": color,
                    "width": 1.5,
                },
                legendgroup=controller,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(
        title_text="Time (s)",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Force (N)",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Current (A)",
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin={
            "l": 15,
            "r": 15,
            "t": 70,
            "b": 15,
        },
    )

    return fig


def make_bar_chart(table, column, title, y_label):
    fig = px.bar(
        table,
        x="Controller",
        y=column,
        color="Controller",
        color_discrete_map=COLORS,
        title=title,
        text_auto=".4g",
    )

    fig.update_layout(
        xaxis_title="Controller",
        yaxis_title=y_label,
        showlegend=False,
        margin={
            "l": 15,
            "r": 15,
            "t": 60,
            "b": 15,
        },
    )

    return fig


def build_plant_variations():
    """Read plant-parameter percentage changes from the sidebar widgets."""

    current_modes = []

    with st.sidebar.expander("Current-driven modal parameters", expanded=False):
        st.caption(
            "Each value is a percentage change from the nominal V3 plant. "
            f"Allowed range: +/-{PLANT_VARIATION_LIMIT_PERCENT:.0f}%."
        )

        for index, (frequency, damping, kyI) in enumerate(NOMINAL_CURRENT_MODES, start=1):
            st.markdown(f"**Mode {index}: {frequency:.0f} Hz nominal**")

            frequency_percent = st.slider(
                f"Mode {index} natural frequency (%)",
                -PLANT_VARIATION_LIMIT_PERCENT,
                PLANT_VARIATION_LIMIT_PERCENT,
                0.0,
                1.0,
                key=f"mode_{index}_frequency",
            )

            damping_percent = st.slider(
                f"Mode {index} damping ratio (%)",
                -PLANT_VARIATION_LIMIT_PERCENT,
                PLANT_VARIATION_LIMIT_PERCENT,
                0.0,
                1.0,
                key=f"mode_{index}_damping",
            )

            kyI_percent = st.slider(
                f"Mode {index} k_yI (%)",
                -PLANT_VARIATION_LIMIT_PERCENT,
                PLANT_VARIATION_LIMIT_PERCENT,
                0.0,
                1.0,
                key=f"mode_{index}_kyi",
            )

            current_modes.append({
                "natural_frequency_percent": frequency_percent,
                "damping_ratio_percent": damping_percent,
                "k_yI_percent": kyI_percent,
            })

    with st.sidebar.expander("Force-driven mode", expanded=False):
        force_frequency_percent = st.slider(
            "Force-mode natural frequency (%)",
            -PLANT_VARIATION_LIMIT_PERCENT,
            PLANT_VARIATION_LIMIT_PERCENT,
            0.0,
            1.0,
        )

        force_damping_percent = st.slider(
            "Force-mode damping ratio (%)",
            -PLANT_VARIATION_LIMIT_PERCENT,
            PLANT_VARIATION_LIMIT_PERCENT,
            0.0,
            1.0,
        )

        force_stiffness_percent = st.slider(
            "Force-mode stiffness (%)",
            -PLANT_VARIATION_LIMIT_PERCENT,
            PLANT_VARIATION_LIMIT_PERCENT,
            0.0,
            1.0,
        )

    with st.sidebar.expander("Actuator parameters", expanded=False):
        actuator_bandwidth_percent = st.slider(
            "Actuator bandwidth (%)",
            -PLANT_VARIATION_LIMIT_PERCENT,
            PLANT_VARIATION_LIMIT_PERCENT,
            0.0,
            1.0,
        )

        actuator_slew_percent = st.slider(
            "Current slew rate (%)",
            -PLANT_VARIATION_LIMIT_PERCENT,
            PLANT_VARIATION_LIMIT_PERCENT,
            0.0,
            1.0,
        )

    return {
        "current_modes": current_modes,
        "force_mode": {
            "natural_frequency_percent": force_frequency_percent,
            "damping_ratio_percent": force_damping_percent,
            "stiffness_percent": force_stiffness_percent,
        },
        "actuator": {
            "bandwidth_percent": actuator_bandwidth_percent,
            "slew_rate_percent": actuator_slew_percent,
        },
    }


def plant_is_nominal(plant_variations):
    for mode in plant_variations["current_modes"]:
        if any(abs(float(value)) > 1e-12 for value in mode.values()):
            return False

    if any(
        abs(float(value)) > 1e-12
        for value in plant_variations["force_mode"].values()
    ):
        return False

    if any(
        abs(float(value)) > 1e-12
        for value in plant_variations["actuator"].values()
    ):
        return False

    return True


def to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------
# Header and service status
# ---------------------------------------------------------------------

st.title("Active Boring-Bar Vibration Control")

st.caption(
    "Simulation dashboard | No Control vs PID vs Soft Actor-Critic (SAC)"
)


with st.sidebar:
    st.header("Simulation settings")

    amplitude_n = st.slider(
        "Cutting-force amplitude (N)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    frequency_hz = st.slider(
        "Disturbance frequency (Hz)",
        min_value=0.0,
        max_value=250.0,
        value=174.0,
        step=1.0,
    )

    phase_rad = st.slider(
        "Disturbance phase (rad)",
        min_value=-3.14,
        max_value=3.14,
        value=0.0,
        step=0.1,
    )

    with st.expander("Cutting-force harmonics & noise", expanded=False):
        harmonic_count = st.number_input(
            "Number of harmonics",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="Adds harmonics 2f, 3f, ... as percentages of the fundamental amplitude.",
        )

        harmonic_amplitudes_ratio = []
        harmonic_phase_rad = []

        for order in range(2, int(harmonic_count) + 2):
            amplitude_percent = st.slider(
                f"{order}f amplitude (% of fundamental)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0,
                key=f"harmonic_{order}_amplitude",
            )

            phase = st.slider(
                f"{order}f phase (rad)",
                min_value=-3.14,
                max_value=3.14,
                value=0.0,
                step=0.1,
                key=f"harmonic_{order}_phase",
            )

            harmonic_amplitudes_ratio.append(
                float(amplitude_percent) / 100.0
            )
            harmonic_phase_rad.append(float(phase))

        cutting_force_noise_std_n = st.slider(
            "Cutting-force noise RMS (N)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help=(
                "Adds zero-mean Gaussian noise directly to the cutting-force "
                "disturbance at each 0.1 ms plant step. This is different from "
                "plant process noise."
            ),
        )

        cutting_force_noise_seed = st.number_input(
            "Cutting-force noise seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

        if harmonic_count > 0:
            st.caption(
                "Harmonics are integer multiples of the fundamental frequency "
                "with independently specified amplitude and phase."
            )

        if cutting_force_noise_std_n > 0:
            st.caption(
                f"Cutting-force noise enabled: {cutting_force_noise_std_n:.2f} N RMS "
                f"(seed {int(cutting_force_noise_seed)})."
            )

    duration_s = st.slider(
        "Simulation duration (s)",
        min_value=0.02,
        max_value=0.5,
        value=0.2,
        step=0.02,
    )

    with st.expander("PID controller parameters", expanded=False):
        st.caption("These gains apply to PID in both single-condition and robust evaluation.")

        pid_kp = st.number_input(
            "Kp (A/m)",
            min_value=0.0,
            max_value=1.0e6,
            value=180000.0,
            step=5000.0,
            format="%.1f",
        )
        pid_ki = st.number_input(
            "Ki (A/(m·s))",
            min_value=0.0,
            max_value=1.0e9,
            value=0.0,
            step=10000.0,
            format="%.3g",
        )
        pid_kd = st.number_input(
            "Kd (A/(m/s))",
            min_value=0.0,
            max_value=1.0e6,
            value=0.0,
            step=100.0,
            format="%.3g",
        )
        pid_integral_limit = st.number_input(
            "Integral error limit (m·s)",
            min_value=1.0e-6,
            max_value=1.0e-1,
            value=1.0e-3,
            step=1.0e-4,
            format="%.2e",
        )

    st.divider()

    st.subheader("Plant & actuator model")

    plant_variations = build_plant_variations()

    process_noise_std_n = st.slider(
        "Process-noise force RMS (N)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=(
            "Adds zero-mean Gaussian force noise to the force-driven plant "
            "mode at each 0.1 ms plant step. This is process/model noise, "
            "not sensor noise."
        ),
    )

    process_noise_seed = st.number_input(
        "Process-noise seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )

    if plant_is_nominal(plant_variations):
        st.caption("Plant model: nominal V3")
    else:
        st.warning(
            "A perturbed plant is selected. The saved SAC policy was not "
            "retrained for this plant; this is an out-of-model deployment test."
        )

    if process_noise_std_n > 0:
        st.caption(
            f"Process noise enabled: {process_noise_std_n:.2f} N RMS "
            f"(seed {int(process_noise_seed)})."
        )

    st.divider()

    st.caption(
        "These are simulated disturbance parameters, not live machine inputs."
    )


try:
    health = api_get("/health")

    st.sidebar.success("API connected")

    model_status = health.get("sac_model", {})

    if model_status.get("model_exists"):
        st.sidebar.success("SAC model file found")
    else:
        st.sidebar.warning("SAC model file not found")
        st.sidebar.caption(
            model_status.get("message", "")
        )

except requests.RequestException as exc:
    st.error(
        "Cannot connect to the API. Start the FastAPI server in another "
        "terminal, then rerun this dashboard."
    )

    st.code(
        r".venv\Scripts\python.exe -m uvicorn api:app --reload",
        language="bat",
    )

    st.caption(f"Connection detail: {exc}")

    st.stop()


# ---------------------------------------------------------------------
# SAC model selection
# ---------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("SAC model")

try:
    model_catalog = api_get("/models").get("models", [])
except requests.RequestException:
    model_catalog = []

# Keep exactly one source of truth for the model used by simulations.
current_default_model = model_status.get(
    "model_path",
    "results/sac_vibration_model_7mode.zip",
)

if "selected_sac_model_path" not in st.session_state:
    st.session_state["selected_sac_model_path"] = current_default_model

# Re-read the catalogue on every rerun, so newly created models are picked up.
model_paths = [item["path"] for item in model_catalog]
model_by_path = {item["path"]: item for item in model_catalog}

# If the currently selected model is in results, show only its filename in the
# dropdown. The full path remains available in the selected-model display below.
if model_paths:
    selected_path = st.session_state["selected_sac_model_path"]
    if selected_path not in model_paths:
        # Prefer the default model if it exists; otherwise use the newest result.
        selected_path = (
            current_default_model
            if current_default_model in model_paths
            else model_paths[0]
        )
        st.session_state["selected_sac_model_path"] = selected_path

    def model_label(path):
        return Path(path).name

    selected_catalog_model = st.sidebar.selectbox(
        "Model from results/",
        options=model_paths,
        format_func=model_label,
        index=model_paths.index(st.session_state["selected_sac_model_path"]),
        key="catalog_sac_model",
        help=(
            "Only SAC .zip models in the API results directory are shown. "
            "Hover over the selected-model path below to see the complete path."
        ),
    )

    if st.sidebar.button("Use selected results model", use_container_width=True):
        st.session_state["selected_sac_model_path"] = selected_catalog_model
        st.rerun()
else:
    st.sidebar.info("No SAC .zip models found in the results directory.")

if st.sidebar.button("Refresh model list", use_container_width=True):
    st.rerun()

with st.sidebar.expander("Use a model from another directory", expanded=False):
    st.caption(
        "Choose a SAC .zip file from your computer. It will be imported into "
        "results/external_models and then become the active model."
    )
    external_model = st.file_uploader(
        "Choose SAC model file",
        type=["zip"],
        key="external_sac_model_uploader",
    )

    if st.button("Use chosen external model", use_container_width=True):
        if external_model is None:
            st.warning("Choose a .zip SAC model first.")
        else:
            try:
                files = {
                    "file": (
                        external_model.name,
                        external_model.getvalue(),
                        "application/zip",
                    )
                }
                response = requests.post(
                    f"{API_BASE}/models/import",
                    files=files,
                    timeout=30,
                )
                response.raise_for_status()
                imported = response.json()
                st.session_state["selected_sac_model_path"] = imported["model_path"]
                st.success(f"Imported: {imported['model_name']}")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Could not import model: {exc}")

# Single, authoritative model path used by every SAC simulation/evaluation.
active_model_path = st.session_state["selected_sac_model_path"]
active_model_name = Path(active_model_path).name
active_exists = Path(active_model_path).is_file()

# Streamlit's markdown supports a native browser tooltip through the HTML title
# attribute. This keeps the sidebar compact while exposing the exact path on hover.
st.sidebar.markdown(
    f'<div title="{active_model_path}"><b>Active SAC model:</b> '
    f'{active_model_name}</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Status: " + ("file found" if active_exists else "file not found")
)
st.sidebar.markdown(
    f'<div title="{active_model_path}" style="font-size:0.78rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
    f'{active_model_path}</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------

st.subheader("Run comparison")

st.write(
    "All three controllers are evaluated with the same disturbance "
    "amplitude, frequency, phase, and duration."
)


run_clicked = st.button(
    "Run all three controllers",
    type="primary",
    use_container_width=True,
)


if run_clicked:
    payload = {
        "model_path": st.session_state["selected_sac_model_path"],
        "amplitude_n": amplitude_n,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "duration_s": duration_s,
        "harmonic_amplitudes_ratio": harmonic_amplitudes_ratio,
        "harmonic_phase_rad": harmonic_phase_rad,
        "cutting_force_noise_std_n": float(cutting_force_noise_std_n),
        "cutting_force_noise_seed": int(cutting_force_noise_seed),
        "plant_variations": plant_variations,
        "process_noise_std_n": float(process_noise_std_n),
        "process_noise_seed": int(process_noise_seed),
        "pid_gains": {
            "kp": float(pid_kp),
            "ki": float(pid_ki),
            "kd": float(pid_kd),
            "integral_limit_m_s": float(pid_integral_limit),
        },
    }

    try:
        with st.spinner("Running simulations..."):
            traces = {}
            metrics_results = []
            run_metadata = None

            for controller in CONTROLLERS:
                result = api_post(
                    "/simulate",
                    {
                        "controller": controller,
                        **payload,
                    },
                )

                traces[controller] = pd.DataFrame(
                    result["timeseries"]
                )

                if controller == "SAC":
                    run_metadata = {
                        "training_frequency_range_hz": result.get(
                            "training_frequency_range_hz", [150.0, 200.0]
                        ),
                        "sac_action_repeat": result.get(
                            "sac_action_repeat", 5
                        ),
                        "sac_action_interval_s": result.get(
                            "sac_action_interval_s", 0.0005
                        ),
                        "observation_history_length": result.get(
                            "observation_history_length", 8
                        ),
                        "plant_parameters": result.get(
                            "plant_parameters", {}
                        ),
                    }

                metrics_results.append({
                    "controller": controller,
                    **result["metrics"],
                })

            st.session_state["traces"] = traces
            st.session_state["metrics_results"] = metrics_results
            st.session_state["simulation_inputs"] = payload
            st.session_state["run_metadata"] = run_metadata or {}

    except requests.HTTPError as exc:
        detail = (
            exc.response.text
            if exc.response is not None
            else str(exc)
        )

        st.error(
            f"API returned an error: {detail}"
        )

    except requests.RequestException as exc:
        st.error(
            f"Simulation request failed: {exc}"
        )

    except (KeyError, ValueError, TypeError) as exc:
        st.error(
            f"Unexpected API response: {exc}"
        )



# ---------------------------------------------------------------------
# SAC training
# ---------------------------------------------------------------------

st.divider()
st.subheader("Train SAC")
st.write(
    "Train and save a new SAC policy using the same 8-state observation "
    "history and 5-plant-step action repeat used by the evaluation pipeline."
)

with st.expander("SAC training configuration", expanded=True):
    st.caption(
        "Training uses randomized amplitude, frequency, and phase within the "
        "ranges below. Plant perturbations use the current sidebar plant settings."
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        train_total_timesteps = st.number_input("Total timesteps", min_value=1_000, max_value=20_000_000, value=300_000, step=10_000)
        train_seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        train_episode_duration = st.number_input("Episode duration (s)", min_value=0.01, max_value=2.0, value=0.2, step=0.01, format="%.3f")
    with t2:
        train_amp_min = st.number_input("Amplitude min (N)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        train_amp_max = st.number_input("Amplitude max (N)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        train_freq_min = st.number_input("Frequency min (Hz)", min_value=0.0, max_value=1000.0, value=150.0, step=5.0)
        train_freq_max = st.number_input("Frequency max (Hz)", min_value=0.0, max_value=1000.0, value=200.0, step=5.0)
    with t3:
        train_phase_min = st.number_input("Phase min (rad)", min_value=-100.0, max_value=100.0, value=-np.pi, step=0.1, format="%.3f")
        train_phase_max = st.number_input("Phase max (rad)", min_value=-100.0, max_value=100.0, value=np.pi, step=0.1, format="%.3f")
        train_process_noise = st.number_input("Process noise RMS (N)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        train_process_seed = st.number_input("Process-noise seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    st.markdown("**Cutting-force harmonics and noise**")
    hcols = st.columns(5)
    train_harmonic_ratios = []
    train_harmonic_phases = []
    for i, col in enumerate(hcols, start=2):
        with col:
            enabled = st.checkbox(f"{i}f", value=False, key=f"train_h{i}_enabled")
            ratio = st.number_input(f"{i}f amplitude (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"train_h{i}_ratio")
            phase = st.number_input(f"{i}f phase (rad)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=f"train_h{i}_phase")
            if enabled:
                train_harmonic_ratios.append(float(ratio) / 100.0)
                train_harmonic_phases.append(float(phase))

    tn1, tn2 = st.columns(2)
    with tn1:
        train_cutting_noise = st.number_input("Cutting-force noise RMS (N)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with tn2:
        train_cutting_seed = st.number_input("Cutting-force noise seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    st.markdown("**SAC hyperparameters**")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        train_lr = st.number_input("Learning rate", min_value=1e-8, max_value=1.0, value=3e-4, format="%.2e")
        train_buffer = st.number_input("Buffer size", min_value=1_000, max_value=5_000_000, value=200_000, step=10_000)
        train_learning_starts = st.number_input("Learning starts", min_value=0, max_value=5_000_000, value=5_000, step=500)
    with h2:
        train_batch = st.number_input("Batch size", min_value=1, max_value=8192, value=256, step=32)
        train_tau = st.number_input("Tau", min_value=1e-6, max_value=1.0, value=0.005, format="%.3f")
        train_gamma = st.number_input("Gamma", min_value=1e-6, max_value=1.0, value=0.99, format="%.3f")
    with h3:
        train_freq = st.number_input("Train frequency", min_value=1, max_value=1000, value=1, step=1)
        train_gradient_steps = st.number_input("Gradient steps", min_value=1, max_value=1000, value=1, step=1)
    with h4:
        train_action_repeat = st.number_input("Action repeat", min_value=1, max_value=100, value=5, step=1)
        train_history = st.number_input("Observation history", min_value=1, max_value=64, value=8, step=1)

    train_plant_label = "nominal" if plant_is_nominal(plant_variations) else "perturbed"
    generated_name = (
        f"SAC_plant_{train_plant_label}_"
        f"A{train_amp_min:g}-{train_amp_max:g}_"
        f"F{train_freq_min:g}-{train_freq_max:g}_"
        f"P{train_phase_min:g}-{train_phase_max:g}_"
        f"H{len(train_harmonic_ratios)}_"
        f"CN{train_cutting_noise:g}_PN{train_process_noise:g}_"
        f"T{int(train_total_timesteps)//1000}k_seed{int(train_seed)}"
    )

    if "training_model_name" not in st.session_state:
        st.session_state["training_model_name"] = generated_name

    n1, n2 = st.columns([5, 1])
    with n1:
        training_model_name = st.text_input(
            "Model filename (editable)",
            key="training_model_name",
            help="The generated descriptive name is only a default. You can replace it before training/saving.",
        )
    with n2:
        st.write("")
        st.write("")
        if st.button("Reset name", use_container_width=True):
            st.session_state["training_model_name"] = generated_name
            st.rerun()

    st.caption(f"Generated default: `{generated_name}.zip`")

    if not plant_is_nominal(plant_variations):
        st.warning("The current perturbed plant settings will be used for training. This makes the resulting policy specific to that plant model.")

    train_clicked = st.button("Train & save SAC model", type="primary", use_container_width=True)

    if train_clicked:
        if train_amp_max < train_amp_min:
            st.error("Amplitude max must be >= amplitude min.")
        elif train_freq_max < train_freq_min:
            st.error("Frequency max must be >= frequency min.")
        elif train_phase_max < train_phase_min:
            st.error("Phase max must be >= phase min.")
        elif not training_model_name.strip():
            st.error("Model filename cannot be empty.")
        else:
            training_payload = {
                "model_name": training_model_name.strip(),
                "total_timesteps": int(train_total_timesteps),
                "seed": int(train_seed),
                "episode_duration_s": float(train_episode_duration),
                "amplitude_range_n": [float(train_amp_min), float(train_amp_max)],
                "frequency_range_hz": [float(train_freq_min), float(train_freq_max)],
                "phase_range_rad": [float(train_phase_min), float(train_phase_max)],
                "harmonic_amplitudes_ratio": train_harmonic_ratios,
                "harmonic_phase_rad": train_harmonic_phases,
                "cutting_force_noise_std_n": float(train_cutting_noise),
                "cutting_force_noise_seed": int(train_cutting_seed),
                "process_noise_std_n": float(train_process_noise),
                "process_noise_seed": int(train_process_seed),
                "plant_variations": plant_variations,
                "sac_action_repeat": int(train_action_repeat),
                "observation_history_length": int(train_history),
                "learning_rate": float(train_lr),
                "buffer_size": int(train_buffer),
                "learning_starts": int(train_learning_starts),
                "batch_size": int(train_batch),
                "tau": float(train_tau),
                "gamma": float(train_gamma),
                "train_freq": int(train_freq),
                "gradient_steps": int(train_gradient_steps),
            }
            try:
                with st.spinner("Training SAC. This can take a while for large timestep counts..."):
                    train_result = api_post("/train", training_payload, timeout=3600)
                st.success(f"Training complete. Saved model: `{train_result['model_path']}`")
                st.json(train_result.get("metadata", {}))
                st.info("Refresh the dashboard or the model list to make the new model available for selection.")
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"SAC training failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"SAC training request failed: {exc}")


if "traces" not in st.session_state:
    st.info(
        "Set the parameters in the sidebar and click "
        "Run all three controllers."
    )

    st.stop()


traces = st.session_state["traces"]
metrics_results = st.session_state["metrics_results"]
run_inputs = st.session_state["simulation_inputs"]
run_metadata = st.session_state.get("run_metadata", {})

table = make_comparison_table(metrics_results)


# ---------------------------------------------------------------------
# Context and key performance indicators
# ---------------------------------------------------------------------

st.divider()

st.subheader("Simulation condition")


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Disturbance amplitude",
    f"{run_inputs['amplitude_n']:.2f} N",
)

c2.metric(
    "Frequency",
    f"{run_inputs['frequency_hz']:.1f} Hz",
)

c3.metric(
    "Phase",
    f"{run_inputs['phase_rad']:.2f} rad",
)

c4.metric(
    "Duration",
    f"{run_inputs['duration_s']:.2f} s",
)
st.caption(
    f"PID gains: Kp={run_inputs.get('pid_gains', {}).get('kp', 180000.0):.3g}, "
    f"Ki={run_inputs.get('pid_gains', {}).get('ki', 0.0):.3g}, "
    f"Kd={run_inputs.get('pid_gains', {}).get('kd', 0.0):.3g}"
)


meta1, meta2, meta3, meta4, meta5 = st.columns(5)

if run_metadata:
    meta1.metric(
        "Harmonics",
        f"{len(run_inputs.get('harmonic_amplitudes_ratio', []))}",
    )
    meta2.metric(
        "Cutting-force noise",
        f"{run_inputs.get('cutting_force_noise_std_n', 0.0):.2f} N RMS",
    )
    meta3.metric(
        "Process noise",
        f"{run_inputs.get('process_noise_std_n', 0.0):.2f} N RMS",
    )
    meta4.metric(
        "SAC history",
        f"{run_metadata.get('observation_history_length', 8)} states",
    )
    meta5.metric(
        "SAC action interval",
        f"{run_metadata.get('sac_action_interval_s', 0.0005) * 1e3:.2f} ms",
    )


frequency = run_inputs["frequency_hz"]
training_range = run_metadata.get(
    "training_frequency_range_hz",
    [150.0, 200.0],
)

if training_range[0] <= frequency <= training_range[1]:
    st.success(
        f"Frequency is within the SAC training range "
        f"({training_range[0]:.0f}–{training_range[1]:.0f} Hz)."
    )
else:
    st.warning(
        f"Frequency is outside the SAC training range "
        f"({training_range[0]:.0f}–{training_range[1]:.0f} Hz). "
        "Treat SAC performance here as an out-of-range test."
    )


baseline_rows = table[
    table["Controller"] == "No control"
]

if not baseline_rows.empty:
    baseline_rms = baseline_rows.iloc[0]["Steady RMS (µm)"]

    controller_rows = table[
        table["Controller"] != "No control"
    ]

    if np.isfinite(baseline_rms) and baseline_rms > 0:
        cols = st.columns(
            max(1, len(controller_rows))
        )

        for col, (_, row) in zip(
            cols,
            controller_rows.iterrows(),
        ):
            rms = row["Steady RMS (µm)"]

            reduction = (
                100 * (1 - rms / baseline_rms)
                if np.isfinite(rms)
                else np.nan
            )

            col.metric(
                f"{row['Controller']} steady RMS reduction",
                (
                    f"{reduction:.2f}%"
                    if np.isfinite(reduction)
                    else "N/A"
                ),
                help=(
                    "Positive means lower steady-state RMS than "
                    "No Control; negative means worse."
                ),
            )


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

tab_vibration, tab_force, tab_metrics, tab_data = st.tabs([
    "Vibration response",
    "Force & current response",
    "Performance metrics",
    "Raw data & downloads",
])


with tab_vibration:

    st.plotly_chart(
        make_trace_plot(
            traces,
            "displacement_um",
            "Tool-mode displacement response",
            "Displacement (µm)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        make_trace_plot(
            traces,
            "velocity_mm_s",
            "Vibration velocity response",
            "Velocity (mm/s)",
        ),
        use_container_width=True,
    )

    st.caption(
        "The displacement is the simulated modal coordinate expressed "
        "in micrometres. It is not necessarily physical tool-tip "
        "displacement without a calibrated mode-shape scale."
    )


with tab_force:

    st.plotly_chart(
        make_force_disturbance_plot(traces),
        use_container_width=True,
    )

    st.caption(
        "The upper plot shows the applied cutting-force disturbance. "
        "The lower plot shows the actual actuator current after the "
        "modeled actuator dynamics, including bandwidth, slew-rate "
        "limiting, and current saturation."
    )


with tab_metrics:

    st.subheader("Metric table")

    st.dataframe(
        table.style.format(
            {
                "Steady RMS (µm)": "{:.4f}",
                "Overall RMS (µm)": "{:.4f}",
                "Peak vibration (µm)": "{:.4f}",
                "RMS current (A)": "{:.4f}",
                "Peak current (A)": "{:.4f}",
                "Saturation (%)": "{:.2f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            make_bar_chart(
                table,
                "Steady RMS (µm)",
                "Steady-state RMS vibration",
                "RMS (µm)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_bar_chart(
                table,
                "Peak vibration (µm)",
                "Peak absolute displacement",
                "Peak (µm)",
            ),
            use_container_width=True,
        )

    with right:

        st.plotly_chart(
            make_bar_chart(
                table,
                "RMS current (A)",
                "RMS control effort",
                "RMS current (A)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_bar_chart(
                table,
                "Saturation (%)",
                "Actuator saturation",
                "Saturated samples (%)",
            ),
            use_container_width=True,
        )

    st.info(
        "Lower vibration is desirable, but compare it alongside "
        "control effort and saturation. These metrics characterize "
        "this simulated scenario only; one run is not a robustness study."
    )


with tab_data:

    selected_controller = st.selectbox(
        "Select controller trace",
        CONTROLLERS,
    )

    selected_data = traces[selected_controller]

    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download selected time series (CSV)",
        data=to_csv_bytes(selected_data),
        file_name=(
            f"{selected_controller.lower().replace(' ', '_')}_trace.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        "Download all controller metrics (CSV)",
        data=to_csv_bytes(table),
        file_name="controller_comparison_metrics.csv",
        mime="text/csv",
    )

    combined = []

    for controller, data in traces.items():
        copy = data.copy()
        copy.insert(
            0,
            "controller",
            controller,
        )
        combined.append(copy)

    all_traces = pd.concat(
        combined,
        ignore_index=True,
    )

    st.download_button(
        "Download all time series (CSV)",
        data=to_csv_bytes(all_traces),
        file_name="all_controller_traces.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Model scope and limitations
# ---------------------------------------------------------------------

with st.expander("Model assumptions and limitations"):

    st.markdown(
        """
        - This is a **seven-mode current-driven modal model** with a
          harmonic cutting-force disturbance.
        - The SAC policy is loaded from a previously trained model;
          this dashboard does not train it.
        - PID gains are user-configurable from the dashboard; SAC uses a learned policy.
        - Robust evaluation uses a user-defined amplitude/frequency/phase grid and reports both per-case and aggregate metrics.
        - Sensor noise, Kalman filtering, amplifier dynamics, and
          randomized plant parameters are not included in this phase.
        - The SAC action is mapped to the 5 A continuous current limit
          before passing through the modeled actuator dynamics.
        - The actuator model includes bandwidth, current slew-rate
          limiting, and current saturation.
        - Results do not establish performance on regenerative chatter,
          a physical boring bar, or real actuator hardware.
        - A single manually selected condition is not sufficient to
          claim robustness or general superiority.
        """
    )


st.caption(
    "Educational simulation only — not connected to physical machinery."
)

# ---------------------------------------------------------------------
# Robust evaluation
# ---------------------------------------------------------------------

st.divider()
st.subheader("Robust evaluation")
st.write(
    "Evaluate No Control, PID, and SAC over a user-defined grid of "
    "amplitude × frequency × phase. The same plant and noise realization "
    "is used for all controllers in every condition."
)

with st.expander("Robust evaluation grid", expanded=True):
    r1, r2, r3 = st.columns(3)

    with r1:
        robust_amp_min = st.number_input(
            "Amplitude minimum (N)", 0.0, 10.0, 0.5, 0.1, key="robust_amp_min"
        )
        robust_amp_max = st.number_input(
            "Amplitude maximum (N)", 0.0, 10.0, 1.5, 0.1, key="robust_amp_max"
        )
        robust_amp_divisions = st.number_input(
            "Amplitude divisions", 1, 24, 2, 1, key="robust_amp_divisions",
            help="Number of equal intervals. Endpoints are included, so N divisions produce N+1 values."
        )

    with r2:
        robust_freq_min = st.number_input(
            "Frequency minimum (Hz)", 0.0, 1000.0, 150.0, 1.0, key="robust_freq_min"
        )
        robust_freq_max = st.number_input(
            "Frequency maximum (Hz)", 0.0, 1000.0, 215.0, 1.0, key="robust_freq_max"
        )
        robust_freq_divisions = st.number_input(
            "Frequency divisions", 1, 24, 6, 1, key="robust_freq_divisions",
            help="Number of equal intervals. Endpoints are included, so N divisions produce N+1 values."
        )

    with r3:
        robust_phase_min = st.number_input(
            "Phase minimum (rad)", -10.0, 10.0, 0.0, 0.1, key="robust_phase_min"
        )
        robust_phase_max = st.number_input(
            "Phase maximum (rad)", -10.0, 10.0, float(3.0 * np.pi / 2.0), 0.1, key="robust_phase_max"
        )
        robust_phase_divisions = st.number_input(
            "Phase divisions", 1, 24, 3, 1, key="robust_phase_divisions",
            help="Number of equal intervals. Endpoints are included, so N divisions produce N+1 values."
        )

    robust_duration_s = st.number_input(
        "Robust evaluation duration per case (s)",
        min_value=0.02,
        max_value=0.5,
        value=0.2,
        step=0.02,
        key="robust_duration",
    )

    amplitudes_preview = np.linspace(robust_amp_min, robust_amp_max, int(robust_amp_divisions) + 1)
    frequencies_preview = np.linspace(robust_freq_min, robust_freq_max, int(robust_freq_divisions) + 1)
    phases_preview = np.linspace(robust_phase_min, robust_phase_max, int(robust_phase_divisions) + 1)
    preview_cases = len(amplitudes_preview) * len(frequencies_preview) * len(phases_preview)
    grid_valid = (robust_amp_max >= robust_amp_min and robust_freq_max >= robust_freq_min and robust_phase_max >= robust_phase_min)

    if not grid_valid:
        st.error("For each parameter, the maximum must be greater than or equal to the minimum.")
    if preview_cases > 500:
        st.error(f"This grid contains {preview_cases} conditions. Reduce the number of divisions; the limit is 500 conditions.")

    st.caption(
        f"Grid: {len(amplitudes_preview)} amplitudes × {len(frequencies_preview)} frequencies × "
        f"{len(phases_preview)} phases = **{preview_cases} conditions**. "
        "The backend limit is 500 conditions."
    )

robust_run = st.button(
    f"Run robust evaluation ({preview_cases} cases)",
    use_container_width=True,
    disabled=(not grid_valid or preview_cases > 500),
)

if robust_run:
    robust_payload = {
        "model_path": st.session_state["selected_sac_model_path"],
        "amplitudes_n": [float(x) for x in amplitudes_preview],
        "frequencies_hz": [float(x) for x in frequencies_preview],
        "phases_rad": [float(x) for x in phases_preview],
        "duration_s": float(robust_duration_s),
        "plant_variations": plant_variations,
        "process_noise_std_n": float(process_noise_std_n),
        "process_noise_seed": int(process_noise_seed),
        "harmonic_amplitudes_ratio": harmonic_amplitudes_ratio,
        "harmonic_phase_rad": harmonic_phase_rad,
        "cutting_force_noise_std_n": float(cutting_force_noise_std_n),
        "cutting_force_noise_seed": int(cutting_force_noise_seed),
        "pid_gains": {
            "kp": float(pid_kp),
            "ki": float(pid_ki),
            "kd": float(pid_kd),
            "integral_limit_m_s": float(pid_integral_limit),
        },
    }

    try:
        with st.spinner(f"Running {preview_cases}-condition robust evaluation... this may take a while."):
            robust_result = api_post("/robust-evaluation", robust_payload)
        st.session_state["robust_result"] = robust_result
        st.success("Robust evaluation completed.")
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"API returned an error: {detail}")
    except requests.RequestException as exc:
        st.error(f"Robust evaluation request failed: {exc}")

if "robust_result" in st.session_state:
    robust_result = st.session_state["robust_result"]
    robust_summary = make_robust_summary_table(robust_result["summary"])
    robust_cases = pd.DataFrame(robust_result["cases"])

    st.markdown("**Robust summary**")
    st.dataframe(
        robust_summary.style.format({
            "Mean steady RMS (µm)": "{:.4f}",
            "Best-case steady RMS (µm)": "{:.4f}",
            "Worst-case steady RMS (µm)": "{:.4f}",
            "Mean peak (µm)": "{:.4f}",
            "Mean RMS current (A)": "{:.4f}",
            "Mean saturation (%)": "{:.2f}",
            "Mean vibration reduction (%)": "{:.2f}",
            "Best vibration reduction (%)": "{:.2f}",
            "Worst vibration reduction (%)": "{:.2f}",
            "Success rate (%)": "{:.1f}",
            "Robust score": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Robust score = 70% mean vibration reduction versus No Control + 30% low-RMS-current term. "
        "Saturation is reported separately. It is a project comparison metric, not a physical standard."
    )

    st.markdown("**Robust variation plots**")

    plot_controller = st.selectbox(
        "Controller for amplitude × frequency map",
        CONTROLLERS,
        key="robust_plot_controller",
    )

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(
            make_robust_heatmap(
                robust_cases,
                plot_controller,
                "steady_rms_vibration_um",
                f"Steady RMS across amplitude and frequency — {plot_controller}",
                "Steady RMS (µm)",
            ),
            use_container_width=True,
        )

    with p2:
        st.plotly_chart(
            make_robust_heatmap(
                robust_cases,
                plot_controller,
                "steady_rms_reduction_vs_no_control_percent",
                f"Vibration reduction across amplitude and frequency — {plot_controller}",
                "Reduction (%)",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        make_robust_frequency_plot(
            robust_cases,
            "steady_rms_vibration_um",
            "Mean steady RMS versus disturbance frequency",
            "Mean steady RMS (µm)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        make_robust_frequency_plot(
            robust_cases,
            "steady_rms_reduction_vs_no_control_percent",
            "Mean vibration reduction versus disturbance frequency",
            "Mean reduction vs No Control (%)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        make_robust_best_mean_worst_plot(
            robust_result["summary"],
            "mean_steady_rms_vibration_um",
            "best_case_steady_rms_vibration_um",
            "worst_case_steady_rms_vibration_um",
            "Best, mean, and worst steady RMS",
            "Steady RMS (µm)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        make_robust_best_mean_worst_plot(
            robust_result["summary"],
            "mean_vibration_reduction_percent",
            "best_vibration_reduction_percent",
            "worst_vibration_reduction_percent",
            "Best, mean, and worst vibration reduction",
            "Reduction vs No Control (%)",
        ),
        use_container_width=True,
    )

    st.download_button(
        "Download robust evaluation cases (CSV)",
        data=to_csv_bytes(robust_cases),
        file_name="robust_evaluation_cases.csv",
        mime="text/csv",
        key="download_robust_cases",
    )

