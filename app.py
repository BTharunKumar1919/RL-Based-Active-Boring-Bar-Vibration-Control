"""Streamlit dashboard for simulated active boring-bar vibration control.

The dashboard communicates with api.py. It does not control physical machinery.
"""

import requests

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


def api_post(path, payload):
    response = requests.post(
        f"{API_BASE}{path}",
        json=payload,
        timeout=60,
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
            "Worst-case steady RMS (µm)": item["worst_case_steady_rms_vibration_um"],
            "Mean peak (µm)": item["mean_peak_vibration_um"],
            "Mean RMS current (A)": item["mean_rms_control_current_a"],
            "Mean saturation (%)": item["mean_hard_saturation_percent"],
            "Mean vibration reduction (%)": item["mean_vibration_reduction_percent"],
            "Worst vibration reduction (%)": item["worst_vibration_reduction_percent"],
            "Success rate (%)": item["success_rate_percent"],
            "Robust score": item["composite_robust_score"],
        })
    return pd.DataFrame(rows)


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
# Robust evaluation
# ---------------------------------------------------------------------

st.divider()
st.subheader("Robust evaluation")
st.write(
    "Evaluate No Control, PID, and SAC over the standard 3 × 7 × 4 operating-condition grid "
    "(amplitude × frequency × phase). The same plant and noise realization is used for all controllers."
)

robust_duration_s = st.number_input(
    "Robust evaluation duration per case (s)",
    min_value=0.02,
    max_value=0.5,
    value=0.2,
    step=0.02,
    key="robust_duration",
)

robust_run = st.button(
    "Run standard robust evaluation (84 cases)",
    use_container_width=True,
)

if robust_run:
    robust_payload = {
        "amplitudes_n": [0.5, 1.0, 1.5],
        "frequencies_hz": [150.0, 160.0, 174.0, 185.0, 200.0, 205.0, 215.0],
        "phases_rad": [0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)],
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
        with st.spinner("Running 84-condition robust evaluation... this may take a while."):
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

    st.markdown("**Robust summary**")
    st.dataframe(
        robust_summary.style.format({
            "Mean steady RMS (µm)": "{:.4f}",
            "Worst-case steady RMS (µm)": "{:.4f}",
            "Mean peak (µm)": "{:.4f}",
            "Mean RMS current (A)": "{:.4f}",
            "Mean saturation (%)": "{:.2f}",
            "Mean vibration reduction (%)": "{:.2f}",
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

    robust_cases = pd.DataFrame(robust_result["cases"])
    st.download_button(
        "Download robust evaluation cases (CSV)",
        data=to_csv_bytes(robust_cases),
        file_name="robust_evaluation_cases.csv",
        mime="text/csv",
        key="download_robust_cases",
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
        - Robust evaluation uses the standard amplitude/frequency/phase grid and reports both per-case and aggregate metrics.
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