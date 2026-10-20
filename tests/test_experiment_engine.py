import numpy as np
from config import PLANT
from experiment_engine import (
    CuttingForce, apply_plant_variations, plant_from_dict, plant_to_dict,
    resolve_plant_preset, simulate_experiment,
)

def test_plant_config_round_trip():
    restored = plant_from_dict(plant_to_dict(PLANT))
    assert len(restored.current_modes) == 6
    assert restored.force_mode.natural_frequency_hz == PLANT.force_mode.natural_frequency_hz

def test_modal_variations_are_applied_relatively():
    changed = apply_plant_variations(PLANT, {"mode_1":{"frequency_pct":10}, "force_mode":{"stiffness_pct":-5}})
    assert np.isclose(changed.current_modes[0].natural_frequency_hz, PLANT.current_modes[0].natural_frequency_hz*1.1)
    assert np.isclose(changed.force_mode.stiffness_n_m, PLANT.force_mode.stiffness_n_m*0.95)

def test_preset_and_custom_variation_merge():
    cfg, merged = resolve_plant_preset("frequencies_plus_5pct", {"mode_1":{"frequency_pct":2}})
    assert np.isclose(cfg.current_modes[0].natural_frequency_hz, PLANT.current_modes[0].natural_frequency_hz*1.07)
    assert merged["mode_1"]["frequency_pct"] == 7

def test_cutting_force_is_seed_reproducible():
    config={"kind":"mixed","amplitude_n":1,"frequency_hz":100,"harmonics":[{"order":2,"amplitude_n":0.2}],"modulation_depth":0.2,"noise_std_n":0.1}
    a=CuttingForce(config,np.random.default_rng(7))(0.003)
    b=CuttingForce(config,np.random.default_rng(7))(0.003)
    assert a == b

def test_short_no_control_experiment_returns_metrics_and_trace():
    result=simulate_experiment({"controller":"No control","duration_s":0.01,"seed":11,
        "cutting_force":{"kind":"sinusoidal","amplitude_n":1,"frequency_hz":174,"phase_rad":0,"noise_std_n":0},
        "noise":{"process_std_n":0,"sensor_std_m":0}})
    assert len(result["timeseries"]) == 100
    assert "steady_rms_vibration_um" in result["metrics"]
