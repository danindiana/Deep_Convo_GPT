"""
Hormonal Dynamics Model

Simulates the endocrine system's role in sexual arousal, including the
Hypothalamic-Pituitary-Gonadal (HPG) axis and key neurohormones.

Models:
- HPG axis (GnRH → LH/FSH → Testosterone/Estrogen)
- Prolactin dynamics (post-orgasmic inhibition)
- Oxytocin release (arousal and orgasm)
- Cortisol (stress effects)
- Circadian rhythms
- Feedback regulation

References:
- Bancroft, J. (2005). The endocrinology of sexual arousal.
- Krüger, T. H., et al. (2002). Neuroendocrine response to orgasm.

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Tuple, Callable, Optional, Dict
import warnings


class HormonalDynamicsModel:
    """
    Comprehensive model of hormonal regulation of sexual arousal.

    Includes:
    - HPG axis with feedback loops
    - Prolactin dynamics
    - Oxytocin release
    - Cortisol (stress hormone)
    - Circadian variations
    - Sex differences
    """

    def __init__(
        self,
        sex: str = 'male',
        gnrh_pulse_amplitude: float = 1.0,
        gnrh_pulse_frequency: float = 1.5,  # pulses per hour
        testosterone_baseline: float = 15.0,  # nmol/L (male range)
        estrogen_baseline: float = 0.1,  # nmol/L (male range)
        prolactin_sensitivity: float = 1.0,
        stress_level: float = 0.0,  # 0-1 scale
    ):
        """
        Initialize hormonal dynamics model.

        Parameters
        ----------
        sex : str
            'male' or 'female' - affects hormone ranges and dynamics
        gnrh_pulse_amplitude : float
            Amplitude of GnRH pulsatile release
        gnrh_pulse_frequency : float
            Frequency of GnRH pulses (per hour)
        testosterone_baseline : float
            Baseline testosterone level (nmol/L)
        estrogen_baseline : float
            Baseline estrogen level (nmol/L)
        prolactin_sensitivity : float
            Sensitivity to prolactin inhibition (higher = stronger refractory)
        stress_level : float
            Chronic stress level (0-1, affects cortisol)
        """
        self.sex = sex.lower()
        self.gnrh_amplitude = gnrh_pulse_amplitude
        self.gnrh_frequency = gnrh_pulse_frequency
        self.testosterone_baseline = testosterone_baseline
        self.estrogen_baseline = estrogen_baseline
        self.prolactin_sensitivity = prolactin_sensitivity
        self.stress_level = stress_level

        # Sex-specific parameter adjustments
        if self.sex == 'female':
            self.testosterone_baseline *= 0.05  # 20x lower in females
            self.estrogen_baseline *= 100  # 100x higher in females
            self.prolactin_sensitivity *= 0.5  # Shorter refractory period

        # Time constants (in hours)
        self.tau_gnrh = 0.1  # GnRH half-life ~10 minutes
        self.tau_lh = 1.0  # LH half-life ~60 minutes
        self.tau_testosterone = 2.0  # Testosterone half-life ~2 hours
        self.tau_estrogen = 1.5
        self.tau_prolactin = 1.0
        self.tau_oxytocin = 0.05  # Very fast (~3 minutes)
        self.tau_cortisol = 2.0

    def gnrh_pulse(self, t: float) -> float:
        """
        Generate pulsatile GnRH release.

        GnRH is released in pulses every 60-120 minutes.

        Parameters
        ----------
        t : float
            Time in hours

        Returns
        -------
        float
            GnRH pulse magnitude
        """
        # Pulsatile pattern using sine wave
        pulse_period = 1.0 / self.gnrh_frequency  # hours
        phase = (t % pulse_period) / pulse_period

        # Sharp pulses using rectified sine
        if phase < 0.1:  # Pulse duration ~10% of period
            return self.gnrh_amplitude * np.sin(phase * 10 * np.pi) ** 2
        else:
            return 0.01  # Small baseline

    def circadian_modulation(self, t: float) -> float:
        """
        Circadian rhythm modulation (testosterone peaks in morning).

        Parameters
        ----------
        t : float
            Time in hours

        Returns
        -------
        float
            Circadian multiplier (0.7 to 1.3)
        """
        # 24-hour cycle, peak at t=8 (8am), trough at t=20 (8pm)
        return 1.0 + 0.3 * np.cos(2 * np.pi * (t - 8) / 24)

    def derivatives(
        self,
        state: np.ndarray,
        t: float,
        arousal_signal: Callable[[float], float],
        orgasm_times: list
    ) -> np.ndarray:
        """
        Compute derivatives for the hormonal system.

        State variables:
        [0] GnRH - Gonadotropin-Releasing Hormone
        [1] LH - Luteinizing Hormone
        [2] FSH - Follicle-Stimulating Hormone
        [3] Testosterone
        [4] Estrogen
        [5] Prolactin
        [6] Oxytocin
        [7] Cortisol

        Parameters
        ----------
        state : ndarray
            Current hormone levels
        t : float
            Current time (hours)
        arousal_signal : callable
            Function returning arousal level at time t (0-1)
        orgasm_times : list
            Times when orgasm occurred (triggers prolactin surge)

        Returns
        -------
        ndarray
            Derivatives for each state variable
        """
        GnRH, LH, FSH, T, E, Prl, Oxy, Cort = state

        arousal = arousal_signal(t)
        circadian = self.circadian_modulation(t)

        # Check for recent orgasm
        recent_orgasm = False
        for orgasm_t in orgasm_times:
            if 0 < (t - orgasm_t) < 0.05:  # Within 3 minutes
                recent_orgasm = True
                break

        # GnRH dynamics - pulsatile release, inhibited by stress/prolactin
        gnrh_pulse = self.gnrh_pulse(t)
        gnrh_inhibition = 1.0 / (1.0 + Prl / 10.0 + Cort / 5.0)
        dGnRH = gnrh_pulse * gnrh_inhibition - GnRH / self.tau_gnrh

        # LH/FSH dynamics - stimulated by GnRH, negative feedback from T/E
        lh_production = GnRH * 10.0 / (1.0 + T / 20.0 + E / 5.0)
        dLH = lh_production - LH / self.tau_lh

        fsh_production = GnRH * 8.0 / (1.0 + E / 3.0)
        dFSH = fsh_production - FSH / self.tau_lh

        # Testosterone - produced by LH, circadian rhythm, decay
        t_production = LH * 2.0 * circadian
        dT = t_production - T / self.tau_testosterone

        # Estrogen - produced by FSH and aromatization of testosterone
        e_production = FSH * 1.5 + T * 0.1
        dE = e_production - E / self.tau_estrogen

        # Prolactin - baseline low, surge after orgasm
        prl_baseline = 5.0
        if recent_orgasm:
            prl_surge = 15.0  # 3-4x baseline
        else:
            prl_surge = 0.0
        dPrl = prl_surge - (Prl - prl_baseline) / self.tau_prolactin

        # Oxytocin - rises with arousal, surges at orgasm
        oxy_baseline = 1.0
        oxy_arousal = arousal * 0.5
        if recent_orgasm:
            oxy_surge = 4.0
        else:
            oxy_surge = 0.0
        dOxy = (oxy_arousal + oxy_surge) - (Oxy - oxy_baseline) / self.tau_oxytocin

        # Cortisol - stress hormone, inhibits HPG axis
        cort_baseline = 10.0 * (1.0 + self.stress_level)
        cort_arousal_effect = -arousal * 2.0  # Arousal reduces cortisol
        dCort = cort_arousal_effect - (Cort - cort_baseline) / self.tau_cortisol

        return np.array([dGnRH, dLH, dFSH, dT, dE, dPrl, dOxy, dCort])

    def simulate(
        self,
        duration: float = 24.0,  # hours
        dt: float = 0.01,
        arousal_signal: Optional[Callable[[float], float]] = None,
        orgasm_times: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate hormonal dynamics over time.

        Parameters
        ----------
        duration : float
            Simulation duration in hours
        dt : float
            Time step in hours
        arousal_signal : callable, optional
            Function returning arousal level (0-1) at time t
            If None, uses constant baseline
        orgasm_times : list, optional
            Times (in hours) when orgasm occurs

        Returns
        -------
        dict
            Dictionary with time and all hormone trajectories
        """
        if arousal_signal is None:
            arousal_signal = lambda t: 0.0

        if orgasm_times is None:
            orgasm_times = []

        # Initial conditions (baseline levels)
        initial_state = np.array([
            0.5,  # GnRH
            5.0,  # LH
            3.0,  # FSH
            self.testosterone_baseline,
            self.estrogen_baseline,
            5.0,  # Prolactin
            1.0,  # Oxytocin
            10.0 * (1.0 + self.stress_level)  # Cortisol
        ])

        # Time points
        t = np.arange(0, duration, dt)

        # Solve ODE system
        solution = odeint(
            self.derivatives,
            initial_state,
            t,
            args=(arousal_signal, orgasm_times)
        )

        return {
            'time': t,
            'GnRH': solution[:, 0],
            'LH': solution[:, 1],
            'FSH': solution[:, 2],
            'Testosterone': solution[:, 3],
            'Estrogen': solution[:, 4],
            'Prolactin': solution[:, 5],
            'Oxytocin': solution[:, 6],
            'Cortisol': solution[:, 7],
            'arousal': np.array([arousal_signal(ti) for ti in t])
        }

    def plot_hpg_axis(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot HPG axis hormones."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        t = results['time']

        # GnRH and gonadotropins
        axes[0].plot(t, results['GnRH'], 'b-', label='GnRH', linewidth=2)
        axes[0].set_ylabel('GnRH (arbitrary)', fontsize=11)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Hypothalamic-Pituitary-Gonadal (HPG) Axis',
                         fontsize=14, fontweight='bold')

        axes[1].plot(t, results['LH'], 'r-', label='LH', linewidth=2)
        axes[1].plot(t, results['FSH'], 'orange', label='FSH', linewidth=2)
        axes[1].set_ylabel('Gonadotropins (IU/L)', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, results['Testosterone'], 'g-',
                    label='Testosterone', linewidth=2)
        axes[2].plot(t, results['Estrogen'], 'purple',
                    label='Estrogen', linewidth=2)
        axes[2].set_ylabel('Sex Hormones (nmol/L)', fontsize=11)
        axes[2].set_xlabel('Time (hours)', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_arousal_hormones(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot hormones directly involved in arousal/orgasm."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        t = results['time']

        # Arousal signal
        axes[0].plot(t, results['arousal'], 'b-', linewidth=2.5)
        axes[0].fill_between(t, 0, results['arousal'], alpha=0.3)
        axes[0].set_ylabel('Arousal Level', fontsize=11)
        axes[0].set_title('Arousal and Neurohormone Dynamics',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Oxytocin (bonding, orgasm)
        axes[1].plot(t, results['Oxytocin'], 'cyan', linewidth=2,
                    label='Oxytocin')
        axes[1].set_ylabel('Oxytocin (pg/mL)', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Prolactin (refractory period)
        axes[2].plot(t, results['Prolactin'], 'purple', linewidth=2,
                    label='Prolactin')
        axes[2].axhline(y=5, color='gray', linestyle='--', alpha=0.5,
                       label='Baseline')
        axes[2].set_ylabel('Prolactin (ng/mL)', fontsize=11)
        axes[2].set_xlabel('Time (hours)', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_stress_effects(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot stress hormone (cortisol) and its effects on HPG axis."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        t = results['time']

        # Cortisol
        axes[0].plot(t, results['Cortisol'], 'orange', linewidth=2.5)
        axes[0].set_ylabel('Cortisol (μg/dL)', fontsize=11)
        axes[0].set_title('Stress Effects on Arousal System',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Compare testosterone with/without stress
        axes[1].plot(t, results['Testosterone'], 'g-', linewidth=2.5,
                    label='Testosterone')
        axes[1].set_ylabel('Testosterone (nmol/L)', fontsize=11)
        axes[1].set_xlabel('Time (hours)', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def example_arousal_episode():
    """
    Example: Simulate arousal episode with orgasm.

    Demonstrates hormonal changes during sexual activity.
    """
    print("=" * 70)
    print("Hormonal Dynamics During Sexual Arousal Episode")
    print("=" * 70)
    print()

    # Create model (male)
    model = HormonalDynamicsModel(sex='male', stress_level=0.1)

    # Define arousal signal: arousal from hour 1-2, orgasm at hour 1.5
    def arousal_signal(t):
        if 1.0 < t < 2.0:
            # Ramp up, plateau, then orgasm
            if t < 1.5:
                return (t - 1.0) * 2.0  # Ramp to 1.0
            else:
                return 1.0
        else:
            return 0.0

    orgasm_times = [1.5]  # Orgasm at 1.5 hours

    # Simulate 6 hours
    print("Simulating 6-hour period with arousal episode...")
    results = model.simulate(
        duration=6.0,
        dt=0.001,
        arousal_signal=arousal_signal,
        orgasm_times=orgasm_times
    )

    print("✓ Simulation complete")
    print()

    # Find peak values
    print("Peak Hormone Levels:")
    print("-" * 50)
    print(f"Peak Oxytocin:    {results['Oxytocin'].max():.2f} pg/mL " +
          f"(baseline: {results['Oxytocin'][:100].mean():.2f})")
    print(f"Peak Prolactin:   {results['Prolactin'].max():.2f} ng/mL " +
          f"(baseline: {results['Prolactin'][:100].mean():.2f})")
    print(f"Testosterone:     {results['Testosterone'].mean():.2f} nmol/L " +
          f"(range: {results['Testosterone'].min():.2f}-{results['Testosterone'].max():.2f})")
    print()

    # Generate plots
    print("Generating plots...")
    fig1 = model.plot_hpg_axis(results)
    fig1.savefig('hpg_axis_dynamics.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: hpg_axis_dynamics.png")

    fig2 = model.plot_arousal_hormones(results)
    fig2.savefig('arousal_hormones.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: arousal_hormones.png")

    fig3 = model.plot_stress_effects(results)
    fig3.savefig('stress_effects.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: stress_effects.png")

    print()
    print("=" * 70)
    print("Example complete! View generated plots.")
    print("=" * 70)

    return results


def compare_stress_levels():
    """Compare hormonal dynamics under different stress conditions."""
    print()
    print("=" * 70)
    print("Comparing Stress Effects on Hormonal Dynamics")
    print("=" * 70)
    print()

    stress_levels = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stress in enumerate(stress_levels):
        model = HormonalDynamicsModel(sex='male', stress_level=stress)
        results = model.simulate(duration=24.0, dt=0.01)

        t = results['time']
        axes[idx].plot(t, results['Testosterone'], 'g-', linewidth=2,
                      label='Testosterone')
        axes[idx].plot(t, results['Cortisol'], 'orange', linewidth=2,
                      label='Cortisol')
        axes[idx].plot(t, results['LH'], 'r--', linewidth=1.5, label='LH')

        axes[idx].set_title(f'Stress Level: {stress:.1f}', fontweight='bold')
        axes[idx].set_xlabel('Time (hours)')
        axes[idx].set_ylabel('Hormone Levels')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Effect of Chronic Stress on HPG Axis (24-hour period)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('stress_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: stress_comparison.png")
    print()


if __name__ == '__main__':
    # Run example
    results = example_arousal_episode()

    # Compare stress levels
    compare_stress_levels()

    print()
    print("All demonstrations complete!")
    print()

    # Show plots (comment out if running headless)
    # plt.show()
