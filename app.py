
"""Streamlit dashboard for simulated active boring-bar vibration control.

The dashboard communicates with api.py. It does not control physical machinery.
"""

import io
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
                result, "steady_rms_vibration_um"
            ),
            "Overall RMS (µm)": metric_value(
                result, "rms_vibration_um"
            ),
            "Peak vibration (µm)": metric_value(
                result, "peak_vibration_um"
            ),
            "RMS current (A)": metric_value(
                result, "rms_control_current_a"
            ),
            "Peak current (A)": metric_value(
                result, "peak_control_current_a"
            ),
            "Saturation (%)": 100 * metric_value(
                result, "actuator_saturation_fraction"
            ),
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
                line={"color": COLORS[controller], "width": 1.8},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Controller",
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
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

        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["disturbance_force_n"],
                name=f"{controller} disturbance",
                line={"color": color, "width": 1.5},
                legendgroup=controller,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["control_current_a"],
                name=f"{controller} control",
                line={"color": color, "width": 1.5},
                legendgroup=controller,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Force (N)", row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=1)
    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin={"l": 15, "r": 15, "t": 70, "b": 15},
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
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


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

    duration_s = st.slider(
        "Simulation duration (s)",
        min_value=0.02,
        max_value=0.5,
        value=0.2,
        step=0.02,
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
        st.sidebar.caption(model_status.get("message", ""))

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
# Advanced configurable experiment builder / policy registry
# ---------------------------------------------------------------------

st.divider()
st.header("Configurable experiments and SAC policy registry")
st.caption("Choose a trained policy independently from the plant and cutting-force configuration used for evaluation. Mismatched evaluations are allowed and flagged.")

try:
    preset_data = api_get("/plant-presets")
    policy_data = api_get("/policies").get("policies", [])
except requests.RequestException:
    preset_data = {"presets": [{"id":"paper_nominal","label":"Paper nominal (baseline)"}]}
    policy_data = []

preset_options = {x["label"]: x["id"] for x in preset_data.get("presets", [])}
with st.expander("1. Experiment configuration", expanded=True):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        adv_controller = st.selectbox("Controller", ["SAC", "PID", "No control"], key="adv_controller")
        policy_labels = {f"{p.get('name',p.get('id'))} [{p.get('training_mode','unknown')}]" + (" — missing checkpoint" if not p.get("model_exists") else ""): p.get("id") for p in policy_data}
        usable_policy_labels = [k for k,pid in policy_labels.items() if next((p.get("model_exists") for p in policy_data if p.get("id")==pid),False)]
        selected_policy_label = st.selectbox("Trained SAC policy", usable_policy_labels or ["No registered policies yet"], key="adv_policy")
        selected_policy_id = policy_labels.get(selected_policy_label) if usable_policy_labels else None
        adv_preset_label = st.selectbox("Plant preset", list(preset_options.keys()), key="adv_preset")
        adv_preset = preset_options[adv_preset_label]
    with adv2:
        force_kind = st.selectbox("Cutting-force model", ["sinusoidal", "multi_harmonic", "modulated", "mixed"], format_func=lambda x: {"sinusoidal":"Single harmonic","multi_harmonic":"Multiple harmonics","modulated":"Amplitude-modulated harmonic","mixed":"Harmonics + modulation"}[x], key="adv_force_kind")
        force_amp = st.number_input("Fundamental amplitude (N)", 0.0, 10.0, 1.0, 0.1, key="adv_force_amp")
        force_freq = st.number_input("Fundamental frequency (Hz)", 0.0, 1000.0, 174.0, 1.0, key="adv_force_freq")
        force_phase = st.number_input("Fundamental phase (rad)", -6.28, 6.28, 0.0, 0.1, key="adv_force_phase")
    with adv3:
        adv_duration = st.number_input("Duration (s)", 0.01, 2.0, 0.2, 0.01, key="adv_duration")
        adv_seed = st.number_input("Random seed", 0, 2147483647, 42, 1, key="adv_seed")
        process_noise = st.number_input("Process noise std (N)", 0.0, 10.0, 0.0, 0.01, key="adv_process_noise")
        sensor_noise_um = st.number_input("Sensor displacement noise std (µm)", 0.0, 100.0, 0.0, 0.1, key="adv_sensor_noise")
        force_noise = st.number_input("Cutting-force noise std (N)", 0.0, 10.0, 0.0, 0.01, key="adv_force_noise")

    harm1, harm2 = st.columns(2)
    with harm1:
        h2_order = st.number_input("Harmonic 2 order", 2, 20, 2, 1, key="adv_h2_order")
        h2_amp = st.number_input("Harmonic 2 amplitude (N)", 0.0, 10.0, 0.25, 0.05, key="adv_h2_amp")
        h2_phase = st.number_input("Harmonic 2 phase (rad)", -6.28, 6.28, 0.0, 0.1, key="adv_h2_phase")
    with harm2:
        h3_order = st.number_input("Harmonic 3 order", 2, 20, 3, 1, key="adv_h3_order")
        h3_amp = st.number_input("Harmonic 3 amplitude (N)", 0.0, 10.0, 0.10, 0.05, key="adv_h3_amp")
        h3_phase = st.number_input("Harmonic 3 phase (rad)", -6.28, 6.28, 0.0, 0.1, key="adv_h3_phase")
    if force_kind in ("modulated", "mixed"):
        mod1, mod2 = st.columns(2)
        with mod1: mod_depth = st.slider("Modulation depth", 0.0, 1.0, 0.25, 0.05, key="adv_mod_depth")
        with mod2: mod_freq = st.number_input("Modulation frequency (Hz)", 0.0, 500.0, 5.0, 0.5, key="adv_mod_freq")
    else:
        mod_depth, mod_freq = 0.0, 5.0

    with st.expander("Per-mode plant parameter variations", expanded=False):
        st.caption("Percentage offsets are applied to the selected preset. Frequency and damping offsets are relative to each mode's nominal values.")
        variations = {}
        for mode_i in range(1, 7):
            st.markdown(f"**Current-driven mode {mode_i}**")
            vc1, vc2, vc3 = st.columns(3)
            variations[f"mode_{mode_i}"] = {
                "frequency_pct": vc1.slider("Frequency (%)", -20, 20, 0, 1, key=f"adv_m{mode_i}_f"),
                "damping_pct": vc2.slider("Damping (%)", -20, 20, 0, 1, key=f"adv_m{mode_i}_d"),
                "gain_pct": vc3.slider("Input gain (%)", -20, 20, 0, 1, key=f"adv_m{mode_i}_g"),
            }
        st.markdown("**Cutting-force-driven mode**")
        vf1, vf2, vf3 = st.columns(3)
        variations["force_mode"] = {
            "frequency_pct": vf1.slider("Frequency (%)", -20, 20, 0, 1, key="adv_fm_f"),
            "damping_pct": vf2.slider("Damping (%)", -20, 20, 0, 1, key="adv_fm_d"),
            "stiffness_pct": vf3.slider("Stiffness (%)", -20, 20, 0, 1, key="adv_fm_k"),
        }

    cutting_force_cfg = {"kind": force_kind, "amplitude_n": force_amp, "frequency_hz": force_freq,
        "phase_rad": force_phase, "noise_std_n": force_noise,
        "harmonics": [{"order": int(h2_order), "amplitude_n": h2_amp, "phase_rad": h2_phase},
                      {"order": int(h3_order), "amplitude_n": h3_amp, "phase_rad": h3_phase}],
        "modulation_depth": mod_depth, "modulation_frequency_hz": mod_freq}
    advanced_payload = {"controller": adv_controller, "policy_id": selected_policy_id,
        "plant_preset": adv_preset, "variations": variations, "cutting_force": cutting_force_cfg,
        "noise": {"process_std_n": process_noise, "sensor_std_m": sensor_noise_um * 1e-6},
        "duration_s": adv_duration, "seed": int(adv_seed)}
    if st.button("Run configurable experiment", type="primary", key="run_advanced_experiment"):
        try:
            with st.spinner("Running configurable simulation..."):
                adv_result = api_post("/experiment/simulate", advanced_payload)
            st.session_state["advanced_result"] = adv_result
        except requests.HTTPError as exc:
            st.error(f"Experiment failed: {exc.response.text if exc.response is not None else exc}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

if "advanced_result" in st.session_state:
    adv_result = st.session_state["advanced_result"]
    if adv_result.get("warning"):
        st.warning(adv_result["warning"])
    st.subheader("Latest configurable experiment")
    st.json({"controller": adv_result.get("controller"), "policy": adv_result.get("policy"),
             "seed": adv_result.get("seed"), "metrics": adv_result.get("metrics"),
             "plant_variations": adv_result.get("variations"), "noise": adv_result.get("noise")})
    adv_df = pd.DataFrame(adv_result.get("timeseries", []))
    if not adv_df.empty:
        st.plotly_chart(px.line(adv_df, x="time_s", y="displacement_um", title="Configurable experiment: displacement (µm)"), use_container_width=True)
        st.download_button("Download configurable experiment CSV", adv_df.to_csv(index=False).encode("utf-8"), file_name="configurable_experiment.csv", mime="text/csv")

with st.expander("Train and manage SAC policies", expanded=False):
    train_col1, train_col2, train_col3 = st.columns(3)
    with train_col1:
        training_mode = st.selectbox("Training strategy", ["individual", "generalized"], format_func=lambda x: "Individual configuration" if x=="individual" else "Generalized (randomized plant)", key="training_mode")
        training_name = st.text_input("Policy name", value="", key="training_name")
    with train_col2:
        training_steps = st.number_input("Training timesteps", 1000, 5000000, 50000, 1000, key="training_steps")
        training_seed = st.number_input("Training seed", 0, 2147483647, 42, 1, key="training_seed")
    with train_col3:
        training_jitter = st.slider("Generalized plant variation (%)", 0.0, 30.0, 10.0, 1.0, key="training_jitter")
        st.caption("Training uses the selected force model and noise settings above.")
    if st.button("Start SAC training job", key="start_training_job"):
        try:
            job = api_post("/training/jobs", {"name": training_name, "training_mode": training_mode,
                "total_timesteps": int(training_steps), "seed": int(training_seed),
                "plant_variation_pct": training_jitter, "plant_preset": adv_preset, "variations": variations,
                "cutting_force": cutting_force_cfg,
                "noise": {"process_std_n": process_noise, "sensor_std_m": sensor_noise_um*1e-6}})
            st.session_state["training_job_id"] = job["id"]
            st.success(f"Training job started: {job['id']}")
        except requests.HTTPError as exc:
            st.error(f"Could not start training: {exc.response.text if exc.response is not None else exc}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
    if st.session_state.get("training_job_id"):
        job_id = st.session_state["training_job_id"]
        try:
            job = api_get(f"/training/jobs/{job_id}")
            st.write(f"Job **{job['id']}** — status: **{job['status']}**")
            st.progress(float(job.get("progress", 0.0)))
            st.caption(f"{job.get('timesteps',0):,} / {job.get('total_timesteps',0):,} timesteps")
            st.code("\n".join(job.get("logs", [])[-12:]), language="text")
            if job["status"] in ("running", "queued"):
                if st.button("Stop training (save partial checkpoint)", key="cancel_training_job"):
                    api_post(f"/training/jobs/{job_id}/cancel", {})
                    st.warning("Cancellation requested. Refresh status to confirm the partial checkpoint was saved.")
            if job.get("error"): st.error(job["error"])
            if st.button("Refresh training status / policy list", key="refresh_training_status"):
                st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not retrieve training status: {exc}")

st.divider()

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
    }

    try:
        with st.spinner("Running simulations..."):
            traces = {}
            metrics_results = []

            for controller in CONTROLLERS:
                result = api_post(
                    "/simulate",
                    {"controller": controller, **payload},
                )
                traces[controller] = pd.DataFrame(result["timeseries"])
                metrics_results.append({
                    "controller": controller,
                    **result["metrics"],
                })

            st.session_state["traces"] = traces
            st.session_state["metrics_results"] = metrics_results
            st.session_state["simulation_inputs"] = payload

    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"API returned an error: {detail}")
    except requests.RequestException as exc:
        st.error(f"Simulation request failed: {exc}")
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"Unexpected API response: {exc}")


if "traces" not in st.session_state:
    st.info("Set the parameters in the sidebar and click Run all three controllers.")
    st.stop()


traces = st.session_state["traces"]
metrics_results = st.session_state["metrics_results"]
run_inputs = st.session_state["simulation_inputs"]
table = make_comparison_table(metrics_results)

# ---------------------------------------------------------------------
# Context and key performance indicators
# ---------------------------------------------------------------------

st.divider()
st.subheader("Simulation condition")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Disturbance amplitude", f"{run_inputs['amplitude_n']:.2f} N")
c2.metric("Frequency", f"{run_inputs['frequency_hz']:.1f} Hz")
c3.metric("Phase", f"{run_inputs['phase_rad']:.2f} rad")
c4.metric("Duration", f"{run_inputs['duration_s']:.2f} s")

frequency = run_inputs["frequency_hz"]
if 150 <= frequency <= 200:
    st.success("Frequency is within the SAC training range (150–200 Hz).")
else:
    st.warning(
        "Frequency is outside the SAC training range (150–200 Hz). "
        "Treat SAC performance here as an out-of-range test."
    )

baseline_rows = table[table["Controller"] == "No control"]
if not baseline_rows.empty:
    baseline_rms = baseline_rows.iloc[0]["Steady RMS (µm)"]
    controller_rows = table[table["Controller"] != "No control"]

    if np.isfinite(baseline_rms) and baseline_rms > 0:
        cols = st.columns(max(1, len(controller_rows)))
        for col, (_, row) in zip(cols, controller_rows.iterrows()):
            rms = row["Steady RMS (µm)"]
            reduction = 100 * (1 - rms / baseline_rms) if np.isfinite(rms) else np.nan
            col.metric(
                f"{row['Controller']} steady RMS reduction",
                f"{reduction:.2f}%" if np.isfinite(reduction) else "N/A",
                help="Positive means lower steady-state RMS than No Control; negative means worse.",
            )

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

tab_vibration, tab_force, tab_metrics, tab_data = st.tabs([
    "Vibration response",
    "Force response",
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
        "The displacement is the simulated modal coordinate expressed in "
        "micrometres. It is not necessarily physical tool-tip displacement "
        "without a calibrated mode-shape scale."
    )

with tab_force:
    st.plotly_chart(
        make_force_disturbance_plot(traces),
        use_container_width=True,
    )

    st.caption(
        "Control force is the simulated generalized force. It does not model "
        "real actuator bandwidth, delay, or hardware dynamics."
    )

with tab_metrics:
    st.subheader("Metric table")
    st.dataframe(
        table.style.format({
            "Steady RMS (µm)": "{:.4f}",
            "Overall RMS (µm)": "{:.4f}",
            "Peak vibration (µm)": "{:.4f}",
            "RMS force (N)": "{:.4f}",
            "Peak force (N)": "{:.4f}",
            "Saturation (%)": "{:.2f}",
        }, na_rep="N/A"),
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
                "RMS force (N)",
                "RMS control effort",
                "RMS force (N)",
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
        "Lower vibration is desirable, but compare it alongside control effort "
        "and saturation. These metrics characterize this simulated scenario "
        "only; one run is not a robustness study."
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
        file_name=f"{selected_controller.lower().replace(' ', '_')}_trace.csv",
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
        copy.insert(0, "controller", controller)
        combined.append(copy)

    all_traces = pd.concat(combined, ignore_index=True)
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
        - This is a **seven-mode current-driven modal model** with a harmonic cutting-force disturbance.
        - The SAC policy is loaded from a previously trained model; this
          dashboard does not train it.
        - PID uses fixed gains; SAC uses a learned policy.
        - Sensor noise, Kalman filtering, amplifier dynamics, and randomized
          plant parameters are not included in this phase. The SAC action is
          mapped to the 5 A continuous current limit.
        - Results do not establish performance on regenerative chatter,
          a physical boring bar, or real actuator hardware.
        - A single manually selected condition is not sufficient to claim
          robustness or general superiority.
        """
    )

st.caption(
    "Educational simulation only — not connected to physical machinery."
)