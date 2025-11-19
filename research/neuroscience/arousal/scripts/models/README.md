# Computational Models of Sexual Arousal

Comprehensive suite of computational models for understanding sexual arousal across multiple biological levels.

## 📊 Models Overview

### 1. **Arousal Dynamics Model** (`arousal_dynamics.py`)
Dual control model of excitation and inhibition.

**Key Features**:
- Ordinary differential equation (ODE) model
- Excitation and inhibition processes
- Positive feedback mechanisms
- Parameter sensitivity analysis

**Use Cases**:
- Understanding individual differences
- Testing dual control theory
- Simulating arousal trajectories
- Educational demonstrations

---

### 2. **Hormonal Dynamics Model** (`hormonal_dynamics.py`)
Endocrine system model including HPG axis.

**Key Features**:
- Hypothalamic-Pituitary-Gonadal (HPG) axis
- GnRH pulsatile release
- Testosterone, Estrogen regulation
- Oxytocin, Prolactin dynamics
- Cortisol (stress effects)
- Circadian rhythms
- Sex differences

**Use Cases**:
- Hormone time-course predictions
- Stress effect analysis
- Sex difference research
- Refractory period modeling

---

### 3. **Neural Network Classifier** (`neural_network_classifier.py`)
Machine learning model for arousal state classification.

**Key Features**:
- Multi-layer perceptron (MLP)
- 11 physiological/hormonal features
- 5 arousal states classification
- Prediction confidence scores
- Feature importance analysis

**Use Cases**:
- Arousal state prediction
- Pattern recognition research
- Clinical decision support
- Feature importance analysis

---

### 4. **Integrated Multi-Level Model** (`integrated_arousal_model.py`)
Comprehensive model combining behavioral, hormonal, and physiological levels.

**Key Features**:
- Integrates all three models above
- Multi-timescale dynamics
- Bi-directional coupling between levels
- Orgasm detection
- Complete arousal cycle simulation

**Use Cases**:
- Comprehensive arousal simulation
- Cross-level interaction research
- Clinical scenario modeling
- Educational demonstrations

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to repository
cd /path/to/Deep_Convo_GPT/research/neuroscience/arousal

# Install dependencies
pip install -r scripts/requirements.txt
```

### Running Models

#### Arousal Dynamics
```python
from models.arousal_dynamics import ArousalModel

# Create model
model = ArousalModel(excitation_rate=1.5, inhibition_rate=0.8)

# Simulate
t, E, I, arousal = model.simulate(duration=100)

# Plot
fig = model.plot(t, E, I, arousal)
fig.savefig('arousal_dynamics.png')
```

#### Hormonal Dynamics
```python
from models.hormonal_dynamics import HormonalDynamicsModel

# Create model (male, moderate stress)
model = HormonalDynamicsModel(sex='male', stress_level=0.3)

# Define arousal episode
def arousal_signal(t):
    return 1.0 if 1.0 < t < 2.0 else 0.0

# Simulate
results = model.simulate(
    duration=6.0,
    arousal_signal=arousal_signal,
    orgasm_times=[1.5]
)

# Plot
fig = model.plot_arousal_hormones(results)
fig.savefig('hormonal_dynamics.png')
```

#### Neural Network Classifier
```python
from models.neural_network_classifier import ArousalStateClassifier

# Create classifier
classifier = ArousalStateClassifier()

# Generate training data
X, y = classifier.generate_synthetic_data(n_samples=1000)

# Train
metrics = classifier.train(X, y)

# Predict new samples
states, probabilities = classifier.predict(X_new)
```

#### Integrated Model
```python
from models.integrated_arousal_model import IntegratedArousalModel

# Create integrated model
model = IntegratedArousalModel(
    sex='male',
    excitation_sensitivity=1.2,
    stress_level=0.1
)

# Run comprehensive simulation
results = model.simulate(duration=2.0)

# Plot all levels
fig = model.plot_integrated_results(results)
fig.savefig('integrated_simulation.png')
```

---

## 🧪 Testing

Run comprehensive test suite:

```bash
cd scripts/models
python test_all_models.py
```

Expected output:
```
🧪🧪🧪... COMPREHENSIVE MODEL TEST SUITE  🧪🧪🧪

TEST 1: Arousal Dynamics Model
✓ Model instantiation successful
✓ Simulation successful
✓ Output validation passed
✅ Arousal Dynamics Model: ALL TESTS PASSED

... (additional tests)

TEST SUMMARY
Arousal Dynamics........................... ✅ PASS
Hormonal Dynamics.......................... ✅ PASS
Neural Network............................. ✅ PASS
Integrated Model........................... ✅ PASS

🎉 ALL TESTS PASSED!
```

---

## 📖 Detailed Documentation

### Arousal Dynamics Model

**Mathematical Formulation**:
```
dE/dt = α·S(t) - γ·I·E - δ·E + η·E²/(1+E²)
dI/dt = β·E - I

Arousal = E - I
```

**Parameters**:
- `α` (alpha): Excitation rate [0.3 - 3.0]
- `β` (beta): Inhibition rate [0.1 - 2.0]
- `γ` (gamma): Coupling strength [0.1 - 1.0]
- `η` (eta): Feedback strength [0.0 - 0.8]
- `δ` (delta): Baseline decay [~0.1]

**Example Scenarios**:
```python
# High excitation, low inhibition (easily aroused)
model1 = ArousalModel(excitation_rate=2.0, inhibition_rate=0.3)

# Low excitation, high inhibition (arousal difficulties)
model2 = ArousalModel(excitation_rate=0.5, inhibition_rate=1.2)

# Balanced (typical response)
model3 = ArousalModel(excitation_rate=1.0, inhibition_rate=0.5)
```

---

### Hormonal Dynamics Model

**Hormone Pathways**:
```
GnRH (pulsatile) → LH/FSH → Testosterone/Estrogen
Arousal → Oxytocin ↑
Orgasm → Prolactin ↑ (inhibits arousal)
Stress → Cortisol ↑ → HPG axis ↓
```

**Sex Differences**:
| Hormone | Male Baseline | Female Baseline |
|---------|--------------|-----------------|
| Testosterone | 15 nmol/L | 0.75 nmol/L |
| Estrogen | 0.1 nmol/L | 10 nmol/L |
| Refractory | Longer | Shorter/Absent |

**Time Constants**:
- GnRH: ~10 minutes
- LH/FSH: ~60 minutes
- Testosterone: ~2 hours
- Oxytocin: ~3 minutes (fast)
- Prolactin: ~60 minutes

**Clinical Applications**:
```python
# Simulate stress effect on arousal
model_stressed = HormonalDynamicsModel(stress_level=0.8)

# Simulate hypogonadism
model_low_t = HormonalDynamicsModel(testosterone_baseline=5.0)

# Simulate hyperprolactinemia
model_high_prl = HormonalDynamicsModel(prolactin_sensitivity=2.0)
```

---

### Neural Network Classifier

**Architecture**:
```
Input Layer (11 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Hidden Layer 3 (16 neurons, ReLU)
    ↓
Output Layer (5 states, Softmax)
```

**Input Features**:
1. Heart Rate (bpm)
2. Systolic Blood Pressure (mmHg)
3. Diastolic Blood Pressure (mmHg)
4. Skin Conductance (μS)
5. Genital Blood Flow (index)
6. Pupil Dilation (mm)
7. Respiration Rate (breaths/min)
8. Testosterone (nmol/L)
9. Oxytocin (pg/mL)
10. Prolactin (ng/mL)
11. Cortisol (μg/dL)

**Output States**:
- 0: Baseline
- 1: Desire
- 2: Arousal
- 3: Orgasm
- 4: Resolution

**Performance**:
- Typical accuracy: 85-95%
- Typical F1 score: 0.85-0.92
- Training time: ~30 seconds (500 samples)

**Customization**:
```python
# Change architecture
classifier = ArousalStateClassifier(
    hidden_layers=(128, 64, 32),  # Deeper network
    learning_rate=0.0005,
    max_iterations=1000
)

# Generate more training data
X, y = classifier.generate_synthetic_data(
    n_samples=5000,
    noise_level=0.2  # More realistic noise
)
```

---

### Integrated Multi-Level Model

**System Hierarchy**:
```
Stimulus
    ↓
Behavioral Level (E/I dynamics)
    ↓
Hormonal Level (HPG axis, neurohormones)
    ↓
Physiological Level (cardiovascular, genital)
    ↓ ↑ (bidirectional feedback)
Observable Responses
```

**Coupling Mechanisms**:
1. **Top-Down**: Arousal → hormone release → physiological changes
2. **Bottom-Up**: Physiological state → hormone feedback → arousal modulation
3. **Lateral**: Hormones ↔ physiological responses

**Timescales**:
- Behavioral: Seconds to minutes
- Hormonal: Minutes to hours
- Physiological: Seconds to minutes

**Example Applications**:
```python
# Simulate complete arousal episode
model = IntegratedArousalModel(sex='female')
results = model.simulate(duration=3.0)  # 3 hours

# Analyze peak responses
print(f"Peak arousal: {results['arousal'].max():.2f}")
print(f"Peak HR: {results['heart_rate'].max():.0f} bpm")
print(f"Orgasms: {results['orgasm_occurred'].sum()}")

# Compare male vs female
model_m = IntegratedArousalModel(sex='male')
model_f = IntegratedArousalModel(sex='female')

results_m = model_m.simulate(duration=2.0)
results_f = model_f.simulate(duration=2.0)

# Analyze refractory period differences
# (females typically shorter/absent)
```

---

## 🎯 Use Case Examples

### Research Applications

#### 1. Testing Dual Control Theory
```python
# Vary E/I sensitivity
ses_values = [0.5, 1.0, 1.5, 2.0]  # SES variation
sis_values = [0.5, 1.0, 1.5, 2.0]  # SIS variation

results_grid = []
for ses in ses_values:
    for sis in sis_values:
        model = ArousalModel(
            excitation_rate=1.5 * ses,
            inhibition_rate=0.8 * sis
        )
        t, E, I, arousal = model.simulate(duration=100)
        results_grid.append({
            'SES': ses,
            'SIS': sis,
            'peak_arousal': arousal.max(),
            'time_to_peak': t[arousal.argmax()]
        })

# Analyze how SES/SIS affect arousal patterns
```

#### 2. Hormone Replacement Therapy Simulation
```python
# Baseline (hypogonadal)
model_baseline = HormonalDynamicsModel(
    sex='male',
    testosterone_baseline=5.0  # Low T
)

# After testosterone therapy
model_treated = HormonalDynamicsModel(
    sex='male',
    testosterone_baseline=15.0  # Normal T
)

# Compare arousal responses
```

#### 3. Stress Effect Quantification
```python
stress_levels = [0.0, 0.3, 0.6, 0.9]

for stress in stress_levels:
    model = IntegratedArousalModel(stress_level=stress)
    results = model.simulate(duration=2.0)

    print(f"Stress {stress:.1f}: Peak arousal = {results['arousal'].max():.2f}")
    # Quantify stress-induced arousal suppression
```

### Clinical Applications

#### Arousal Disorder Diagnosis
```python
# Patient presents with arousal difficulties
# Collect physiological data during arousal attempt

patient_data = np.array([[
    75,   # HR
    125,  # Sys BP
    85,   # Dia BP
    6,    # Skin conductance
    0.2,  # Genital flow (LOW)
    3.5,  # Pupil
    16,   # Respiration
    8,    # Testosterone (LOW)
    1.1,  # Oxytocin
    5,    # Prolactin
    12    # Cortisol
]])

# Classify arousal state
classifier = ArousalStateClassifier()  # Pre-trained
state, prob = classifier.predict(patient_data)

# Interpret: Low genital flow + low testosterone suggests...
# → Potential hypogonadism or vascular issue
```

### Educational Demonstrations

```python
# Interactive demo for students
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_arousal_demo():
    """Let students adjust parameters and see effects."""

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial simulation
    model = ArousalModel()
    t, E, I, arousal = model.simulate(duration=100)
    line, = ax.plot(t, arousal)

    # Parameter sliders
    ax_alpha = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_alpha = Slider(ax_alpha, 'Excitation', 0.3, 3.0, valinit=1.5)

    def update(val):
        model = ArousalModel(excitation_rate=slider_alpha.val)
        t, E, I, arousal = model.simulate(duration=100)
        line.set_ydata(arousal)
        fig.canvas.draw_idle()

    slider_alpha.on_changed(update)
    plt.show()
```

---

## 📊 Model Outputs

### Arousal Dynamics
- `time`: Time array
- `E`: Excitation levels
- `I`: Inhibition levels
- `arousal`: Net arousal (E - I)

### Hormonal Dynamics
- `time`: Time array
- `GnRH`, `LH`, `FSH`: HPG axis hormones
- `Testosterone`, `Estrogen`: Sex hormones
- `Prolactin`, `Oxytocin`: Neurohormones
- `Cortisol`: Stress hormone
- `arousal`: Input arousal signal

### Neural Network
- `states`: Predicted arousal states (0-4)
- `probabilities`: Class probabilities
- `feature_importance`: Relative importance scores

### Integrated Model
- All arousal dynamics outputs
- All hormonal dynamics outputs
- `heart_rate`: Heart rate (bpm)
- `blood_pressure_sys/dia`: Blood pressure
- `genital_blood_flow`: Genital response
- `orgasm_occurred`: Boolean orgasm detection

---

## 🔬 Scientific Validation

### Model Assumptions
1. **Linearity**: Many processes modeled linearly (approximation)
2. **Parameter Constancy**: Parameters assumed constant (may vary individually)
3. **Deterministic**: No stochasticity (could add noise for realism)
4. **Simplified Anatomy**: Aggregate brain regions (not neuron-level)

### Validation Against Literature

**Arousal Dynamics**:
- ✓ Matches dual control model predictions (Janssen & Bancroft, 2007)
- ✓ Captures individual differences
- ✓ Qualitatively matches arousal trajectories

**Hormonal Dynamics**:
- ✓ Prolactin surge post-orgasm (Krüger et al., 2002)
- ✓ Oxytocin surge at orgasm (Carmichael et al., 1987)
- ✓ Circadian testosterone variation (Dabbs, 1990)
- ✓ Stress-induced HPG suppression (Rivier & Rivest, 1991)

**Neural Network**:
- ✓ High accuracy on synthetic data (85-95%)
- ⚠ Needs validation on real data
- ⚠ Feature selection based on literature (may be incomplete)

**Integrated Model**:
- ✓ Multi-level integration conceptually sound
- ✓ Physiological ranges reasonable
- ⚠ Coupling strengths estimated (need empirical validation)

---

## 🛠️ Extending the Models

### Adding New Features

#### Custom Stimulus Patterns
```python
def custom_stimulus(t):
    """Your custom stimulus function."""
    # Example: sinusoidal modulation
    return 0.5 + 0.5 * np.sin(2 * np.pi * t / 10)

model = ArousalModel()
t, E, I, arousal = model.simulate(stimulus=custom_stimulus)
```

#### Additional Hormones
```python
# Extend HormonalDynamicsModel.derivatives()
# Add new state variable and dynamics
# Example: Add dopamine

def enhanced_derivatives(self, state, t, ...):
    # Existing variables
    GnRH, LH, FSH, T, E, Prl, Oxy, Cort = state[:8]

    # New variable
    Dopamine = state[8]

    # Existing dynamics...

    # New dynamics
    dDopamine = arousal * 2.0 - Dopamine / 0.5  # Fast dynamics

    return np.array([...existing..., dDopamine])
```

#### Stochastic Variation
```python
# Add noise to make more realistic
def simulate_with_noise(model, noise_level=0.1):
    t, E, I, arousal = model.simulate(duration=100)

    # Add Gaussian noise
    E_noisy = E + np.random.normal(0, noise_level, len(E))
    I_noisy = I + np.random.normal(0, noise_level, len(I))
    arousal_noisy = E_noisy - I_noisy

    return t, E_noisy, I_noisy, arousal_noisy
```

---

## 📚 References

### Foundational Papers
- Janssen, E., & Bancroft, J. (2007). The dual control model.
- Toates, F. (2009). An integrative theoretical framework.
- Georgiadis & Kringelbach (2012). The human sexual response cycle.

### Hormonal Research
- Krüger et al. (2002). Neuroendocrine response to orgasm.
- Bancroft, J. (2005). The endocrinology of sexual arousal.
- Carmichael et al. (1987). Oxytocin and vasopressin release.

### Computational Modeling
- Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions.
- Hopfield, J. J. (1984). Neurons with graded response.

---

## 🤝 Contributing

To add new models or improve existing ones:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include example usage
4. Add tests to `test_all_models.py`
5. Update this README

---

## 📧 Support

**Issues**: [GitHub Issues](https://github.com/danindiana/Deep_Convo_GPT/issues)
**Discussions**: [GitHub Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)

---

**Created**: 2025-11-19
**Version**: 1.0
**Authors**: Deep Convo GPT Research Team
