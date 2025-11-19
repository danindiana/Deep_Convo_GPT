"""
Integrated Arousal Model

Comprehensive multi-level model integrating:
1. Behavioral dynamics (excitation/inhibition)
2. Hormonal regulation (HPG axis)
3. Physiological responses
4. Neural state classification

This model demonstrates the full complexity of sexual arousal across
biological systems and timescales.

References:
- Georgiadis & Kringelbach (2012). The human sexual response cycle.
- Toates, F. (2009). An integrative theoretical framework.

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Tuple, Dict, Optional, Callable
import sys
from pathlib import Path

# Import other models
try:
    from arousal_dynamics import ArousalModel
    from hormonal_dynamics import HormonalDynamicsModel
except ImportError:
    # If running as script, add parent directory
    sys.path.append(str(Path(__file__).parent))
    from arousal_dynamics import ArousalModel
    from hormonal_dynamics import HormonalDynamicsModel


class IntegratedArousalModel:
    """
    Integrated multi-level arousal model.

    Combines:
    - Behavioral level: Dual control (E/I) dynamics
    - Hormonal level: HPG axis and neurohormones
    - Physiological level: Cardiovascular, autonomic
    - Neural level: Brain state and classification

    Demonstrates bi-directional coupling between levels.
    """

    def __init__(
        self,
        sex: str = 'male',
        excitation_sensitivity: float = 1.0,
        inhibition_sensitivity: float = 1.0,
        stress_level: float = 0.0
    ):
        """
        Initialize integrated model.

        Parameters
        ----------
        sex : str
            'male' or 'female'
        excitation_sensitivity : float
            Sensitivity of excitation system (SES)
        inhibition_sensitivity : float
            Sensitivity of inhibition system (SIS)
        stress_level : float
            Chronic stress level (0-1)
        """
        self.sex = sex
        self.excitation_sens = excitation_sensitivity
        self.inhibition_sens = inhibition_sensitivity
        self.stress_level = stress_level

        # Sub-models
        self.behavioral_model = ArousalModel(
            excitation_rate=1.5 * excitation_sensitivity,
            inhibition_rate=0.8 * inhibition_sensitivity
        )

        self.hormonal_model = HormonalDynamicsModel(
            sex=sex,
            stress_level=stress_level
        )

    def compute_physiological_state(
        self,
        arousal: float,
        hormones: Dict[str, float],
        time: float
    ) -> Dict[str, float]:
        """
        Compute physiological variables from arousal and hormones.

        Parameters
        ----------
        arousal : float
            Net arousal level (E - I)
        hormones : dict
            Hormone levels
        time : float
            Current time

        Returns
        -------
        dict
            Physiological measurements
        """
        # Baseline values
        hr_baseline = 70.0
        bp_sys_baseline = 120.0
        bp_dia_baseline = 80.0
        resp_baseline = 14.0
        skin_cond_baseline = 5.0
        pupil_baseline = 3.0

        # Arousal effects (0-1 normalized)
        arousal_norm = np.clip(arousal, 0, 2) / 2.0

        # Heart rate: increases with arousal and sympathetic activation
        hr = hr_baseline + arousal_norm * 90  # Can reach 160 bpm

        # Blood pressure
        bp_sys = bp_sys_baseline + arousal_norm * 50
        bp_dia = bp_dia_baseline + arousal_norm * 20

        # Respiration: increases with arousal
        resp = resp_baseline + arousal_norm * 21  # Can reach 35 br/min

        # Skin conductance: sympathetic activation
        skin_cond = skin_cond_baseline + arousal_norm * 20

        # Pupil dilation: autonomic arousal
        pupil = pupil_baseline + arousal_norm * 4  # Can reach 7mm

        # Genital blood flow: parasympathetic + testosterone/estrogen
        t_effect = hormones.get('Testosterone', 15) / 20.0
        e_effect = hormones.get('Estrogen', 0.1) / 0.5
        hormone_effect = (t_effect + e_effect) / 2.0
        genital_flow = 0.3 + arousal_norm * 0.7 * hormone_effect

        # Prolactin inhibits arousal
        prl_inhibition = 1.0 / (1.0 + hormones.get('Prolactin', 5) / 10.0)

        # Apply prolactin inhibition to physiological responses
        hr *= prl_inhibition
        genital_flow *= prl_inhibition

        return {
            'heart_rate': hr,
            'blood_pressure_sys': bp_sys,
            'blood_pressure_dia': bp_dia,
            'respiration_rate': resp,
            'skin_conductance': skin_cond,
            'pupil_diameter': pupil,
            'genital_blood_flow': genital_flow
        }

    def detect_orgasm_threshold(self, arousal: float, oxytocin: float) -> bool:
        """
        Detect if orgasm threshold is reached.

        Orgasm occurs when arousal is high and oxytocin is elevated.

        Parameters
        ----------
        arousal : float
            Current arousal level
        oxytocin : float
            Current oxytocin level

        Returns
        -------
        bool
            True if orgasm threshold reached
        """
        # Threshold criteria
        arousal_threshold = 1.5
        oxytocin_threshold = 2.0

        return (arousal > arousal_threshold and oxytocin > oxytocin_threshold)

    def simulate(
        self,
        duration: float = 2.0,  # hours
        dt: float = 0.001,
        stimulus_func: Optional[Callable[[float], float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run integrated simulation.

        Parameters
        ----------
        duration : float
            Simulation duration (hours)
        dt : float
            Time step (hours)
        stimulus_func : callable, optional
            Sexual stimulus function of time

        Returns
        -------
        dict
            Complete simulation results across all levels
        """
        if stimulus_func is None:
            # Default: pulse stimulus from 0.5-1.5 hours
            stimulus_func = lambda t: 1.0 if 0.5 < t < 1.5 else 0.0

        # Time array
        time = np.arange(0, duration, dt)
        n_steps = len(time)

        # Initialize storage
        results = {
            'time': time,
            'stimulus': np.array([stimulus_func(t) for t in time]),
            'excitation': np.zeros(n_steps),
            'inhibition': np.zeros(n_steps),
            'arousal': np.zeros(n_steps),
            'GnRH': np.zeros(n_steps),
            'LH': np.zeros(n_steps),
            'Testosterone': np.zeros(n_steps),
            'Estrogen': np.zeros(n_steps),
            'Prolactin': np.zeros(n_steps),
            'Oxytocin': np.zeros(n_steps),
            'Cortisol': np.zeros(n_steps),
            'heart_rate': np.zeros(n_steps),
            'blood_pressure_sys': np.zeros(n_steps),
            'genital_blood_flow': np.zeros(n_steps),
            'orgasm_occurred': np.zeros(n_steps, dtype=bool)
        }

        # Track orgasm times
        orgasm_times = []

        # Initial behavioral state
        E = 0.1
        I = 0.0

        print(f"Running integrated simulation ({n_steps} steps)...")

        # Simulation loop
        for step, t in enumerate(time):
            if step % (n_steps // 10) == 0:
                print(f"  Progress: {100*step/n_steps:.0f}%")

            # Get stimulus
            s = stimulus_func(t)

            # Update behavioral model (small time step)
            alpha = self.behavioral_model.alpha
            beta = self.behavioral_model.beta
            gamma = self.behavioral_model.gamma
            eta = self.behavioral_model.eta

            dE = (alpha * s - gamma * I * E - 0.1 * E +
                  eta * E**2 / (1 + E**2)) * dt
            dI = (beta * E - I) * dt

            E += dE
            I += dI
            E = max(0, E)
            I = max(0, I)

            arousal = E - I

            # Store behavioral
            results['excitation'][step] = E
            results['inhibition'][step] = I
            results['arousal'][step] = arousal

            # Hormonal simulation (only every 100 steps for efficiency)
            if step % 100 == 0 or step == n_steps - 1:
                # Create arousal signal for hormonal model
                def arousal_signal(t_h):
                    # Find closest time index
                    idx = int(t_h / dt)
                    if idx < step:
                        return results['arousal'][idx] / 2.0  # Normalize to 0-1
                    else:
                        return arousal / 2.0

                # Simulate short hormonal period
                h_results = self.hormonal_model.simulate(
                    duration=t + 0.1,
                    dt=0.001,
                    arousal_signal=arousal_signal,
                    orgasm_times=orgasm_times
                )

                # Extract current hormone levels
                h_idx = -1
                results['GnRH'][step] = h_results['GnRH'][h_idx]
                results['LH'][step] = h_results['LH'][h_idx]
                results['Testosterone'][step] = h_results['Testosterone'][h_idx]
                results['Estrogen'][step] = h_results['Estrogen'][h_idx]
                results['Prolactin'][step] = h_results['Prolactin'][h_idx]
                results['Oxytocin'][step] = h_results['Oxytocin'][h_idx]
                results['Cortisol'][step] = h_results['Cortisol'][h_idx]
            else:
                # Interpolate hormones
                results['GnRH'][step] = results['GnRH'][step-1]
                results['LH'][step] = results['LH'][step-1]
                results['Testosterone'][step] = results['Testosterone'][step-1]
                results['Estrogen'][step] = results['Estrogen'][step-1]
                results['Prolactin'][step] = results['Prolactin'][step-1]
                results['Oxytocin'][step] = results['Oxytocin'][step-1]
                results['Cortisol'][step] = results['Cortisol'][step-1]

            # Compute physiological state
            hormones = {
                'Testosterone': results['Testosterone'][step],
                'Estrogen': results['Estrogen'][step],
                'Prolactin': results['Prolactin'][step],
                'Oxytocin': results['Oxytocin'][step]
            }
            physiology = self.compute_physiological_state(arousal, hormones, t)

            results['heart_rate'][step] = physiology['heart_rate']
            results['blood_pressure_sys'][step] = physiology['blood_pressure_sys']
            results['genital_blood_flow'][step] = physiology['genital_blood_flow']

            # Check for orgasm
            if self.detect_orgasm_threshold(arousal, results['Oxytocin'][step]):
                if not results['orgasm_occurred'][:step].any() or \
                   (step - np.where(results['orgasm_occurred'])[0][-1] > 100):
                    # New orgasm (not duplicate detection)
                    results['orgasm_occurred'][step] = True
                    orgasm_times.append(t)
                    print(f"  Orgasm detected at t={t:.3f} hours")

        print("  Progress: 100%")
        print("✓ Simulation complete")

        return results

    def plot_integrated_results(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """
        Plot comprehensive multi-level results.

        Parameters
        ----------
        results : dict
            Simulation results

        Returns
        -------
        fig : Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

        t = results['time']

        # Panel 1: Stimulus and Behavioral
        axes[0].plot(t, results['stimulus'], 'gray', linestyle='--',
                    linewidth=1.5, label='Stimulus', alpha=0.6)
        axes[0].plot(t, results['arousal'], 'b-', linewidth=2.5, label='Net Arousal')
        axes[0].fill_between(t, 0, results['arousal'], alpha=0.2)
        if results['orgasm_occurred'].any():
            orgasm_times = t[results['orgasm_occurred']]
            for ot in orgasm_times:
                axes[0].axvline(ot, color='red', linestyle=':', linewidth=2, alpha=0.7)
        axes[0].set_ylabel('Arousal', fontsize=11)
        axes[0].set_title('Integrated Arousal Dynamics Across Biological Levels',
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Hormones
        axes[1].plot(t, results['Testosterone'], 'g-', linewidth=2, label='Testosterone')
        axes[1].plot(t, results['Prolactin'], 'purple', linewidth=2, label='Prolactin')
        axes[1].plot(t, results['Oxytocin'], 'cyan', linewidth=2, label='Oxytocin')
        axes[1].set_ylabel('Hormones', fontsize=11)
        axes[1].legend(loc='upper right', ncol=3)
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Cardiovascular
        axes[2].plot(t, results['heart_rate'], 'r-', linewidth=2, label='Heart Rate')
        axes[2].set_ylabel('HR (bpm)', fontsize=11)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Panel 4: Genital Response
        axes[3].plot(t, results['genital_blood_flow'], 'm-', linewidth=2.5)
        axes[3].fill_between(t, 0, results['genital_blood_flow'], alpha=0.3, color='m')
        axes[3].set_ylabel('Genital\nBlood Flow', fontsize=11)
        axes[3].grid(True, alpha=0.3)

        # Panel 5: E/I Components
        axes[4].plot(t, results['excitation'], 'g-', linewidth=2, label='Excitation (E)')
        axes[4].plot(t, results['inhibition'], 'r-', linewidth=2, label='Inhibition (I)')
        axes[4].set_ylabel('E / I', fontsize=11)
        axes[4].set_xlabel('Time (hours)', fontsize=12)
        axes[4].legend(loc='upper right')
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def example_integrated_simulation():
    """Run example integrated simulation."""
    print("=" * 70)
    print("Integrated Multi-Level Arousal Simulation")
    print("=" * 70)
    print()

    # Create model
    print("Initializing integrated model...")
    model = IntegratedArousalModel(
        sex='male',
        excitation_sensitivity=1.2,
        inhibition_sensitivity=0.8,
        stress_level=0.1
    )
    print("✓ Model initialized")
    print()

    # Run simulation
    results = model.simulate(duration=2.0, dt=0.001)
    print()

    # Summary statistics
    print("Simulation Summary:")
    print("-" * 70)
    print(f"Peak arousal:         {results['arousal'].max():.2f}")
    print(f"Peak heart rate:      {results['heart_rate'].max():.0f} bpm")
    print(f"Peak genital flow:    {results['genital_blood_flow'].max():.2f}")
    print(f"Orgasms detected:     {results['orgasm_occurred'].sum()}")
    print(f"Peak testosterone:    {results['Testosterone'].max():.1f} nmol/L")
    print(f"Peak oxytocin:        {results['Oxytocin'].max():.2f} pg/mL")
    print(f"Peak prolactin:       {results['Prolactin'].max():.1f} ng/mL")
    print()

    # Plot
    print("Generating integrated plot...")
    fig = model.plot_integrated_results(results)
    fig.savefig('integrated_arousal_simulation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: integrated_arousal_simulation.png")
    print()

    print("=" * 70)
    print("Integrated simulation complete!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = example_integrated_simulation()

    # Show plot (comment out if running headless)
    # plt.show()
