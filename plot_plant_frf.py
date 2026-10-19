"""Plot the open-loop plant FRFs implied by config.py.

1) Cutting force -> displacement, with current = 0
2) Current -> displacement, with cutting force = 0

These are the modal plant FRFs only. The actuator bandwidth and slew-rate
dynamics are not included because this script describes the underlying
mechanical plant.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import PLANT


def second_order_frf(
    freq_hz,
    natural_freq_hz,
    damping_ratio,
    input_gain,
):
    """Complex FRF of:

        input_gain
        -------------------------------
        s^2 + 2*zeta*wn*s + wn^2
    """

    w = 2.0 * np.pi * np.asarray(
        freq_hz,
        dtype=float,
    )

    wn = 2.0 * np.pi * natural_freq_hz

    s = 1j * w

    return input_gain / (
        s**2
        + 2.0 * damping_ratio * wn * s
        + wn**2
    )


def calculate_frfs(freq_hz):
    """Return cutting-force and current plant FRFs."""

    G_yF = np.zeros_like(
        freq_hz,
        dtype=complex,
    )

    G_yI = np.zeros_like(
        freq_hz,
        dtype=complex,
    )

    # Cutting-force-driven mode.
    mode = PLANT.force_mode

    wn = (
        2.0
        * np.pi
        * mode.natural_frequency_hz
    )

    gain_F = (
        wn**2
        / mode.stiffness_n_m
    )

    G_yF += second_order_frf(
        freq_hz,
        mode.natural_frequency_hz,
        mode.damping_ratio,
        gain_F,
    )

    # Current-driven modes.
    for mode in PLANT.current_modes:

        wn = (
            2.0
            * np.pi
            * mode.natural_frequency_hz
        )

        gain_I = (
            wn**2
            / mode.k_yI
        )

        G_yI += second_order_frf(
            freq_hz,
            mode.natural_frequency_hz,
            mode.damping_ratio,
            gain_I,
        )

    return G_yF, G_yI


def plot_frf_linear(
    freq_hz,
    G,
    title,
    magnitude_unit,
):
    """Plot magnitude and phase of an FRF."""

    magnitude = np.abs(G)
    phase_deg = np.angle(
        G,
        deg=True,
    )

    fig, (ax_mag, ax_phase) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    ax_mag.plot(
        freq_hz,
        magnitude,
    )

    ax_mag.set_xlim(
        0,
        600,
    )

    ax_mag.set_ylabel(
        f"Magnitude ({magnitude_unit})"
    )

    ax_mag.set_title(title)
    ax_mag.grid(True)

    ax_phase.plot(
        freq_hz,
        phase_deg,
    )

    ax_phase.set_xlim(
        0,
        600,
    )

    ax_phase.set_xlabel(
        "Frequency (Hz)"
    )

    ax_phase.set_ylabel(
        "Phase (deg)"
    )

    ax_phase.grid(True)

    fig.tight_layout()

    return fig


def main():
    # Linear frequency sweep from 0 to 600 Hz.
    freq = np.linspace(
        0.0,
        600.0,
        12001,
    )

    G_yF, G_yI = calculate_frfs(
        freq
    )

    plot_frf_linear(
        freq,
        G_yF * 1e6,
        "Cutting-force-to-displacement FRF (current = 0)",
        "um/N",
    )

    plot_frf_linear(
        freq,
        G_yI * 1e6,
        "Current-to-displacement FRF (cutting force = 0)",
        "um/A",
    )

    plt.show()


if __name__ == "__main__":
    main()