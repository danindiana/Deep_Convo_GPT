"""
Arousal Dynamics Model

Simple mathematical model of sexual arousal dynamics based on the dual control model.
Models arousal as a dynamical system with excitatory and inhibitory components.

References:
- Janssen, E., & Bancroft, J. (2007). The dual control model.
- Toates, F. (2009). An integrative theoretical framework.

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Callable, Tuple, Optional


class ArousalModel:
    """
    Dual-process model of sexual arousal.

    Based on the dual control model (Janssen & Bancroft, 2007).
    Includes excitatory (E) and inhibitory (I) processes that
    compete to determine net arousal level.

    The model implements:
    - Excitatory system (SES - Sexual Excitation System)
    - Inhibitory system (SIS - Sexual Inhibition System)
    - Stimulus-driven activation
    - Positive feedback (self-amplification)
    - Negative feedback (homeostatic regulation)
    """

    def __init__(
        self,
        excitation_rate: float = 1.0,
        inhibition_rate: float = 0.5,
        coupling: float = 0.3,
        baseline: float = 0.1,
        feedback_strength: float = 0.2,
    ):
        """
        Initialize arousal model with parameters.

        Parameters
        ----------
        excitation_rate : float
            Rate of excitatory process activation (alpha)
            Higher values = faster arousal response to stimuli
        inhibition_rate : float
            Rate of inhibitory process activation (beta)
            Higher values = stronger arousal suppression
        coupling : float
            Coupling strength between E and I (gamma)
            How much inhibition suppresses excitation
        baseline : float
            Baseline arousal decay rate (delta)
            Natural return to baseline in absence of stimulation
        feedback_strength : float
            Positive feedback strength (eta)
            Self-amplification of arousal
        """
        self.alpha = excitation_rate
        self.beta = inhibition_rate
        self.gamma = coupling
        self.delta = baseline
        self.eta = feedback_strength

    def derivatives(
        self, state: np.ndarray, t: float, stimulus: Callable[[float], float]
    ) -> list:
        """
        Compute derivatives for ODE system.

        The model equations:
        dE/dt = alpha * S(t) - gamma * I * E - delta * E + eta * E^2/(1+E^2)
        dI/dt = beta * E - I

        Where:
        - E: Excitation level
        - I: Inhibition level
        - S(t): Stimulus strength at time t
        - eta * E^2/(1+E^2): Saturating positive feedback

        Parameters
        ----------
        state : array_like
            [E, I] current excitation and inhibition levels
        t : float
            Current time
        stimulus : callable
            Function returning stimulus strength at time t

        Returns
        -------
        derivatives : list
            [dE/dt, dI/dt]
        """
        E, I = state
        s = stimulus(t)

        # Excitation dynamics
        # - Driven by stimulus (alpha * s)
        # - Inhibited by I (gamma * I * E)
        # - Natural decay (delta * E)
        # - Positive feedback (eta * E^2/(1+E^2))
        dE_dt = (
            self.alpha * s
            - self.gamma * I * E
            - self.delta * E
            + self.eta * E ** 2 / (1 + E ** 2)
        )

        # Inhibition dynamics
        # - Built up by excitation (beta * E)
        # - Natural decay (I)
        dI_dt = self.beta * E - I

        return [dE_dt, dI_dt]

    def simulate(
        self,
        duration: float = 100,
        dt: float = 0.1,
        stimulus: Optional[Callable[[float], float]] = None,
        initial_state: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate arousal dynamics over time.

        Parameters
        ----------
        duration : float
            Simulation duration in arbitrary time units
        dt : float
            Time step for numerical integration
        stimulus : callable, optional
            Function of time returning stimulus strength
            If None, uses constant stimulus of 1.0
        initial_state : tuple, optional
            Initial (E, I) values
            If None, uses (baseline, 0)

        Returns
        -------
        t : ndarray
            Time points
        E : ndarray
            Excitation over time
        I : ndarray
            Inhibition over time
        arousal : ndarray
            Net arousal (E - I)
        """
        if stimulus is None:
            # Default: constant stimulus
            stimulus = lambda t: 1.0

        if initial_state is None:
            initial_state = (self.delta, 0.0)

        t = np.arange(0, duration, dt)

        # Solve ODE system
        solution = odeint(self.derivatives, initial_state, t, args=(stimulus,))

        E = solution[:, 0]
        I = solution[:, 1]
        arousal = E - I

        return t, E, I, arousal

    def plot(
        self,
        t: np.ndarray,
        E: np.ndarray,
        I: np.ndarray,
        arousal: np.ndarray,
        title: str = "Arousal Dynamics Simulation",
    ) -> plt.Figure:
        """
        Plot simulation results.

        Parameters
        ----------
        t : ndarray
            Time points
        E : ndarray
            Excitation values
        I : ndarray
            Inhibition values
        arousal : ndarray
            Net arousal values
        title : str
            Plot title

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot excitation
        axes[0].plot(t, E, "g-", linewidth=2, label="Excitation (E)")
        axes[0].set_ylabel("Excitation", fontsize=12)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=14, fontweight="bold")

        # Plot inhibition
        axes[1].plot(t, I, "r-", linewidth=2, label="Inhibition (I)")
        axes[1].set_ylabel("Inhibition", fontsize=12)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Plot net arousal
        axes[2].plot(t, arousal, "b-", linewidth=2, label="Net Arousal (E - I)")
        axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[2].set_ylabel("Arousal", fontsize=12)
        axes[2].set_xlabel("Time (arbitrary units)", fontsize=12)
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_phase_space(
        self, E: np.ndarray, I: np.ndarray, title: str = "Phase Space Trajectory"
    ) -> plt.Figure:
        """
        Plot phase space trajectory (E vs I).

        Parameters
        ----------
        E : ndarray
            Excitation values
        I : ndarray
            Inhibition values
        title : str
            Plot title

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot trajectory
        ax.plot(E, I, "b-", linewidth=2, alpha=0.7)

        # Mark start and end
        ax.scatter(E[0], I[0], s=200, c="green", marker="o", label="Start", zorder=5)
        ax.scatter(E[-1], I[-1], s=200, c="red", marker="s", label="End", zorder=5)

        # Add direction arrows
        skip = len(E) // 10
        for i in range(0, len(E) - skip, skip):
            ax.annotate(
                "",
                xy=(E[i + skip], I[i + skip]),
                xytext=(E[i], I[i]),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.5, lw=1.5),
            )

        ax.set_xlabel("Excitation (E)", fontsize=12)
        ax.set_ylabel("Inhibition (I)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def example_stimuli():
    """
    Define example stimulus functions.

    Returns
    -------
    stimuli : dict
        Dictionary of stimulus name -> function
    """

    def constant(t):
        """Constant stimulus"""
        return 1.0

    def pulse(t):
        """Pulse stimulus (on from t=10 to t=30)"""
        return 1.0 if 10 < t < 30 else 0.0

    def ramp_up(t):
        """Ramping up stimulus"""
        return min(t / 20, 1.0) if t < 40 else max(1.0 - (t - 40) / 20, 0.0)

    def intermittent(t):
        """Intermittent on/off stimulus"""
        return 1.0 if int(t / 5) % 2 == 0 else 0.0

    def gradual(t):
        """Gradually increasing stimulus"""
        return min(t / 50, 1.0)

    return {
        "Constant": constant,
        "Pulse": pulse,
        "Ramp": ramp_up,
        "Intermittent": intermittent,
        "Gradual": gradual,
    }


def run_comparison(duration=60, dt=0.1):
    """
    Run comparison of different stimulus patterns.

    Parameters
    ----------
    duration : float
        Simulation duration
    dt : float
        Time step
    """
    model = ArousalModel()
    stimuli = example_stimuli()

    fig, axes = plt.subplots(len(stimuli), 1, figsize=(12, 10), sharex=True)

    for idx, (name, stim_func) in enumerate(stimuli.items()):
        t, E, I, arousal = model.simulate(duration=duration, dt=dt, stimulus=stim_func)

        # Plot arousal response
        axes[idx].plot(t, arousal, linewidth=2.5, label="Arousal", color="blue")

        # Plot stimulus (scaled for visibility)
        stimulus_vals = [stim_func(ti) for ti in t]
        axes[idx].plot(
            t,
            stimulus_vals,
            "--",
            alpha=0.6,
            linewidth=1.5,
            label="Stimulus",
            color="orange",
        )

        axes[idx].set_ylabel("Response", fontsize=11)
        axes[idx].set_title(f"{name} Stimulus", fontsize=12, fontweight="bold")
        axes[idx].legend(loc="upper right", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color="k", linestyle=":", alpha=0.3)

    axes[-1].set_xlabel("Time (arbitrary units)", fontsize=12)
    plt.suptitle("Arousal Response to Different Stimulus Patterns", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    return fig


def run_parameter_sensitivity():
    """
    Explore how individual differences (parameter variations) affect arousal.
    """
    # Vary excitation rate
    excitation_rates = [0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Excitation rate sensitivity
    ax = axes[0]
    for rate in excitation_rates:
        m = ArousalModel(excitation_rate=rate)
        t, E, I, arousal = m.simulate(duration=50)
        ax.plot(t, arousal, linewidth=2, label=f"α = {rate}")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("Effect of Excitation Rate (α)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Inhibition rate sensitivity
    ax = axes[1]
    inhibition_rates = [0.2, 0.5, 0.8, 1.1]
    for rate in inhibition_rates:
        m = ArousalModel(inhibition_rate=rate)
        t, E, I, arousal = m.simulate(duration=50)
        ax.plot(t, arousal, linewidth=2, label=f"β = {rate}")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("Effect of Inhibition Rate (β)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Coupling strength sensitivity
    ax = axes[2]
    coupling_strengths = [0.1, 0.3, 0.5, 0.7]
    for coupling in coupling_strengths:
        m = ArousalModel(coupling=coupling)
        t, E, I, arousal = m.simulate(duration=50)
        ax.plot(t, arousal, linewidth=2, label=f"γ = {coupling}")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("Effect of Coupling Strength (γ)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Feedback strength sensitivity
    ax = axes[3]
    feedback_strengths = [0.0, 0.2, 0.4, 0.6]
    for feedback in feedback_strengths:
        m = ArousalModel(feedback_strength=feedback)
        t, E, I, arousal = m.simulate(duration=50)
        ax.plot(t, arousal, linewidth=2, label=f"η = {feedback}")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("Effect of Positive Feedback (η)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Sensitivity Analysis\nIndividual Differences in Arousal Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Arousal Dynamics Model - Demonstration")
    print("=" * 60)
    print()

    # Create model with default parameters
    print("Creating arousal model with default parameters...")
    model = ArousalModel(
        excitation_rate=1.5, inhibition_rate=0.8, coupling=0.5, baseline=0.1
    )

    # Define pulse stimulus
    def stimulus_pulse(t):
        """Stimulus active from t=10 to t=50"""
        if 10 < t < 50:
            return 1.0
        else:
            return 0.0

    # Run simulation
    print("Running simulation (duration=100 time units)...")
    t, E, I, arousal = model.simulate(duration=100, stimulus=stimulus_pulse)

    # Compute statistics
    print()
    print("Simulation Results:")
    print("-" * 40)
    print(f"Peak excitation (E):     {E.max():.3f}")
    print(f"Peak inhibition (I):     {I.max():.3f}")
    print(f"Peak net arousal:        {arousal.max():.3f}")
    print(f"Time to peak arousal:    {t[arousal.argmax()]:.2f}")
    print(f"Steady-state arousal:    {arousal[-100:].mean():.3f}")
    print()

    # Plot results
    print("Generating plots...")
    fig1 = model.plot(t, E, I, arousal, title="Arousal Response to Pulse Stimulus")
    fig1.savefig("arousal_simulation.png", dpi=150, bbox_inches="tight")
    print("  - Saved: arousal_simulation.png")

    # Phase space
    fig2 = model.plot_phase_space(E, I)
    fig2.savefig("arousal_phase_space.png", dpi=150, bbox_inches="tight")
    print("  - Saved: arousal_phase_space.png")

    # Stimulus comparison
    print()
    print("Running stimulus pattern comparison...")
    fig3 = run_comparison(duration=60)
    fig3.savefig("stimulus_comparison.png", dpi=150, bbox_inches="tight")
    print("  - Saved: stimulus_comparison.png")

    # Parameter sensitivity
    print()
    print("Running parameter sensitivity analysis...")
    fig4 = run_parameter_sensitivity()
    fig4.savefig("parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    print("  - Saved: parameter_sensitivity.png")

    print()
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Examine generated plots")
    print("  2. Modify parameters in ArousalModel() to explore individual differences")
    print("  3. Define custom stimulus functions")
    print("  4. Extend model with additional biological detail")
    print()

    # Show plots (comment out if running headless)
    plt.show()
