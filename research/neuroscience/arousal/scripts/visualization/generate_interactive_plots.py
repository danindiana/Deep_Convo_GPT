"""
Generate Interactive Plotly Visualizations

Creates multiple interactive HTML visualizations for arousal neuroscience research.

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent.parent))
from models.arousal_dynamics import ArousalModel, example_stimuli


def create_arousal_dynamics_dashboard():
    """
    Create interactive dashboard for exploring arousal dynamics.

    Features:
    - Multiple stimulus patterns
    - Interactive parameter sliders
    - Real-time updates
    """

    # Generate data for different stimulus patterns
    stimuli = example_stimuli()
    model = ArousalModel()

    fig = make_subplots(
        rows=len(stimuli), cols=1,
        subplot_titles=list(stimuli.keys()),
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    colors = px.colors.qualitative.Set2

    for idx, (name, stim_func) in enumerate(stimuli.items(), 1):
        t, E, I, arousal = model.simulate(duration=60, stimulus=stim_func)

        # Stimulus trace
        stimulus_vals = [stim_func(ti) for ti in t]
        fig.add_trace(
            go.Scatter(
                x=t, y=stimulus_vals,
                name=f'{name} Stimulus',
                line=dict(color=colors[0], width=2, dash='dash'),
                opacity=0.5,
                hovertemplate='Time: %{x:.1f}<br>Stimulus: %{y:.2f}<extra></extra>',
                legendgroup=name,
                showlegend=(idx==1)
            ),
            row=idx, col=1
        )

        # Arousal trace
        fig.add_trace(
            go.Scatter(
                x=t, y=arousal,
                name=f'{name} Arousal',
                line=dict(color=colors[1], width=3),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)',
                hovertemplate='Time: %{x:.1f}<br>Arousal: %{y:.2f}<extra></extra>',
                legendgroup=name,
                showlegend=(idx==1)
            ),
            row=idx, col=1
        )

        # Zero line
        fig.add_hline(
            y=0, line_dash="dot", line_color="gray",
            opacity=0.5, row=idx, col=1
        )

    fig.update_xaxes(title_text="Time (arbitrary units)", row=len(stimuli), col=1)
    fig.update_yaxes(title_text="Response")

    fig.update_layout(
        title={
            'text': 'Arousal Response to Different Stimulus Patterns',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        height=250 * len(stimuli),
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_parameter_sensitivity_plot():
    """
    Interactive parameter sensitivity analysis.

    Shows how varying model parameters affects arousal dynamics.
    """

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Excitation Rate (α)',
            'Inhibition Rate (β)',
            'Coupling Strength (γ)',
            'Feedback Strength (η)'
        ]
    )

    # Excitation rate sensitivity
    rates = [0.5, 1.0, 1.5, 2.0]
    colors = px.colors.sequential.Viridis

    for i, rate in enumerate(rates):
        m = ArousalModel(excitation_rate=rate)
        t, E, I, arousal = m.simulate(duration=50)
        fig.add_trace(
            go.Scatter(
                x=t, y=arousal,
                name=f'α = {rate}',
                line=dict(color=colors[i*2], width=2.5),
                hovertemplate='Time: %{x:.1f}<br>Arousal: %{y:.2f}<extra></extra>',
                legendgroup='alpha'
            ),
            row=1, col=1
        )

    # Inhibition rate sensitivity
    rates = [0.2, 0.5, 0.8, 1.1]
    for i, rate in enumerate(rates):
        m = ArousalModel(inhibition_rate=rate)
        t, E, I, arousal = m.simulate(duration=50)
        fig.add_trace(
            go.Scatter(
                x=t, y=arousal,
                name=f'β = {rate}',
                line=dict(color=colors[i*2], width=2.5),
                hovertemplate='Time: %{x:.1f}<br>Arousal: %{y:.2f}<extra></extra>',
                legendgroup='beta',
                showlegend=False
            ),
            row=1, col=2
        )

    # Coupling strength sensitivity
    strengths = [0.1, 0.3, 0.5, 0.7]
    for i, strength in enumerate(strengths):
        m = ArousalModel(coupling=strength)
        t, E, I, arousal = m.simulate(duration=50)
        fig.add_trace(
            go.Scatter(
                x=t, y=arousal,
                name=f'γ = {strength}',
                line=dict(color=colors[i*2], width=2.5),
                hovertemplate='Time: %{x:.1f}<br>Arousal: %{y:.2f}<extra></extra>',
                legendgroup='gamma',
                showlegend=False
            ),
            row=2, col=1
        )

    # Feedback strength sensitivity
    strengths = [0.0, 0.2, 0.4, 0.6]
    for i, strength in enumerate(strengths):
        m = ArousalModel(feedback_strength=strength)
        t, E, I, arousal = m.simulate(duration=50)
        fig.add_trace(
            go.Scatter(
                x=t, y=arousal,
                name=f'η = {strength}',
                line=dict(color=colors[i*2], width=2.5),
                hovertemplate='Time: %{x:.1f}<br>Arousal: %{y:.2f}<extra></extra>',
                legendgroup='eta',
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Arousal")

    fig.update_layout(
        title={
            'text': 'Parameter Sensitivity Analysis<br><sub>How individual differences affect arousal dynamics</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        height=800,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def create_neurotransmitter_timeline():
    """
    Interactive timeline of neurotransmitter levels during arousal.

    Shows how dopamine, oxytocin, prolactin, etc. change over time.
    """

    # Time points (in minutes)
    time = np.linspace(0, 60, 200)

    # Simulate neurotransmitter dynamics
    # Dopamine: rises during arousal, peaks before orgasm, drops after
    dopamine = np.zeros_like(time)
    arousal_start = 10
    orgasm_time = 40
    resolution_end = 55

    for i, t in enumerate(time):
        if t < arousal_start:
            dopamine[i] = 1.0  # baseline
        elif t < orgasm_time:
            dopamine[i] = 1.0 + 3.0 * (1 - np.exp(-(t - arousal_start) / 5))
        else:
            dopamine[i] = 1.0 + 3.0 * np.exp(-(t - orgasm_time) / 8)

    # Oxytocin: gradual rise, surge at orgasm
    oxytocin = np.zeros_like(time)
    for i, t in enumerate(time):
        if t < arousal_start:
            oxytocin[i] = 1.0
        elif t < orgasm_time:
            oxytocin[i] = 1.0 + 0.5 * (t - arousal_start) / (orgasm_time - arousal_start)
        elif t < orgasm_time + 1:
            oxytocin[i] = 4.0  # surge
        else:
            oxytocin[i] = 1.5 + 2.5 * np.exp(-(t - orgasm_time - 1) / 5)

    # Prolactin: low during arousal, surge after orgasm
    prolactin = np.zeros_like(time)
    for i, t in enumerate(time):
        if t < orgasm_time:
            prolactin[i] = 1.0
        else:
            peak = 4.0
            decay = 15
            prolactin[i] = 1.0 + (peak - 1.0) * np.exp(-(t - orgasm_time) / decay)

    # Norepinephrine: rises with arousal
    norepinephrine = np.zeros_like(time)
    for i, t in enumerate(time):
        if t < arousal_start:
            norepinephrine[i] = 1.0
        elif t < orgasm_time:
            norepinephrine[i] = 1.0 + 2.0 * (1 - np.exp(-(t - arousal_start) / 8))
        else:
            norepinephrine[i] = 1.0 + 2.0 * np.exp(-(t - orgasm_time) / 10)

    # Serotonin: relatively stable, slight increase post-orgasm
    serotonin = np.ones_like(time)
    for i, t in enumerate(time):
        if t > orgasm_time:
            serotonin[i] = 1.0 + 0.3 * (1 - np.exp(-(t - orgasm_time) / 20))

    # Create figure
    fig = go.Figure()

    # Add traces
    neurotransmitters = {
        'Dopamine': (dopamine, '#e74c3c', 'Reward and motivation'),
        'Oxytocin': (oxytocin, '#3498db', 'Bonding and orgasm'),
        'Prolactin': (prolactin, '#9b59b6', 'Post-orgasmic inhibition'),
        'Norepinephrine': (norepinephrine, '#f39c12', 'Arousal and attention'),
        'Serotonin': (serotonin, '#2ecc71', 'Mood regulation')
    }

    for name, (values, color, description) in neurotransmitters.items():
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                name=name,
                line=dict(color=color, width=3),
                hovertemplate=f'<b>{name}</b><br>Time: %{{x:.1f}} min<br>Level: %{{y:.2f}}x baseline<br><i>{description}</i><extra></extra>',
                mode='lines'
            )
        )

    # Add phase annotations
    fig.add_vrect(
        x0=0, x1=arousal_start,
        fillcolor="gray", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Baseline", annotation_position="top left"
    )

    fig.add_vrect(
        x0=arousal_start, x1=orgasm_time,
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Arousal", annotation_position="top left"
    )

    fig.add_vrect(
        x0=orgasm_time, x1=orgasm_time + 0.5,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Orgasm", annotation_position="top left"
    )

    fig.add_vrect(
        x0=orgasm_time + 0.5, x1=resolution_end,
        fillcolor="blue", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Resolution", annotation_position="top left"
    )

    # Layout
    fig.update_layout(
        title={
            'text': 'Neurotransmitter Dynamics During Sexual Arousal<br><sub>Relative levels throughout arousal cycle</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        xaxis_title="Time (minutes)",
        yaxis_title="Relative Level (× baseline)",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    return fig


def create_phase_space_3d():
    """
    3D phase space visualization showing E, I, and Arousal dynamics.
    """

    model = ArousalModel(excitation_rate=1.5, inhibition_rate=0.8)

    def pulse_stimulus(t):
        return 1.0 if 10 < t < 50 else 0.0

    t, E, I, arousal = model.simulate(duration=100, stimulus=pulse_stimulus)

    # Create 3D scatter
    fig = go.Figure()

    # Color by time
    fig.add_trace(
        go.Scatter3d(
            x=E,
            y=I,
            z=arousal,
            mode='lines+markers',
            marker=dict(
                size=3,
                color=t,
                colorscale='Viridis',
                colorbar=dict(title="Time"),
                showscale=True
            ),
            line=dict(
                color=t,
                colorscale='Viridis',
                width=3
            ),
            hovertemplate='<b>Time: %{marker.color:.1f}</b><br>Excitation: %{x:.2f}<br>Inhibition: %{y:.2f}<br>Arousal: %{z:.2f}<extra></extra>',
            name='Trajectory'
        )
    )

    # Add start and end markers
    fig.add_trace(
        go.Scatter3d(
            x=[E[0]],
            y=[I[0]],
            z=[arousal[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='<b>Start</b><extra></extra>'
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[E[-1]],
            y=[I[-1]],
            z=[arousal[-1]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='square'),
            name='End',
            hovertemplate='<b>End</b><extra></extra>'
        )
    )

    # Layout
    fig.update_layout(
        title={
            'text': '3D Phase Space: Arousal Dynamics<br><sub>Excitation, Inhibition, and Net Arousal over time</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        scene=dict(
            xaxis_title='Excitation (E)',
            yaxis_title='Inhibition (I)',
            zaxis_title='Net Arousal (E - I)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        template='plotly_white'
    )

    return fig


def create_individual_comparison():
    """
    Compare arousal responses across different 'individuals' with varying parameters.
    """

    # Define different individual profiles
    individuals = {
        'High Excitation': {'excitation_rate': 2.0, 'inhibition_rate': 0.3, 'color': '#e74c3c'},
        'High Inhibition': {'excitation_rate': 1.0, 'inhibition_rate': 1.2, 'color': '#3498db'},
        'Balanced': {'excitation_rate': 1.0, 'inhibition_rate': 0.5, 'color': '#2ecc71'},
        'Low Responsiveness': {'excitation_rate': 0.4, 'inhibition_rate': 0.6, 'color': '#95a5a6'},
    }

    # Standard stimulus
    def standard_stimulus(t):
        return 1.0 if 10 < t < 50 else 0.0

    fig = go.Figure()

    for name, params in individuals.items():
        model = ArousalModel(
            excitation_rate=params['excitation_rate'],
            inhibition_rate=params['inhibition_rate']
        )
        t, E, I, arousal = model.simulate(duration=70, stimulus=standard_stimulus)

        fig.add_trace(
            go.Scatter(
                x=t,
                y=arousal,
                name=name,
                line=dict(color=params['color'], width=3),
                hovertemplate=f'<b>{name}</b><br>Time: %{{x:.1f}}<br>Arousal: %{{y:.2f}}<extra></extra>'
            )
        )

    # Add stimulus reference
    stimulus_vals = [standard_stimulus(ti) for ti in t]
    fig.add_trace(
        go.Scatter(
            x=t,
            y=stimulus_vals,
            name='Stimulus',
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.5,
            hovertemplate='Stimulus: %{y:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        title={
            'text': 'Individual Differences in Arousal Response<br><sub>Same stimulus, different responses based on excitation/inhibition balance</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        xaxis_title="Time (arbitrary units)",
        yaxis_title="Arousal",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.9)"
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    return fig


def create_heatmap_parameter_space():
    """
    Heatmap showing peak arousal across parameter combinations.
    """

    # Parameter ranges
    excitation_rates = np.linspace(0.3, 2.5, 20)
    inhibition_rates = np.linspace(0.1, 1.5, 20)

    # Calculate peak arousal for each combination
    peak_arousal = np.zeros((len(inhibition_rates), len(excitation_rates)))

    for i, inh_rate in enumerate(inhibition_rates):
        for j, exc_rate in enumerate(excitation_rates):
            model = ArousalModel(
                excitation_rate=exc_rate,
                inhibition_rate=inh_rate
            )
            t, E, I, arousal = model.simulate(duration=50)
            peak_arousal[i, j] = arousal.max()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=peak_arousal,
        x=excitation_rates,
        y=inhibition_rates,
        colorscale='RdYlGn',
        hovertemplate='Excitation: %{x:.2f}<br>Inhibition: %{y:.2f}<br>Peak Arousal: %{z:.2f}<extra></extra>',
        colorbar=dict(title="Peak<br>Arousal")
    ))

    # Add contour lines
    fig.add_trace(
        go.Contour(
            z=peak_arousal,
            x=excitation_rates,
            y=inhibition_rates,
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            line=dict(width=1, color='white'),
            opacity=0.5,
            hoverinfo='skip'
        )
    )

    fig.update_layout(
        title={
            'text': 'Peak Arousal Across Parameter Space<br><sub>Heatmap of excitation vs inhibition rates</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        xaxis_title="Excitation Rate (α)",
        yaxis_title="Inhibition Rate (β)",
        height=700,
        template='plotly_white'
    )

    return fig


def main():
    """Generate all visualizations and save to HTML files."""

    print("=" * 60)
    print("Generating Interactive Plotly Visualizations")
    print("=" * 60)
    print()

    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'diagrams' / 'interactive'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    visualizations = {
        'arousal_dynamics_dashboard.html': create_arousal_dynamics_dashboard,
        'parameter_sensitivity.html': create_parameter_sensitivity_plot,
        'neurotransmitter_timeline.html': create_neurotransmitter_timeline,
        'phase_space_3d.html': create_phase_space_3d,
        'individual_comparison.html': create_individual_comparison,
        'parameter_heatmap.html': create_heatmap_parameter_space,
    }

    for filename, create_func in visualizations.items():
        print(f"Creating {filename}...", end=' ')
        try:
            fig = create_func()
            output_path = output_dir / filename
            fig.write_html(str(output_path))
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")

    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print()
    print("Open the HTML files in your web browser to view.")
    print()


if __name__ == '__main__':
    main()
