# Interactive Visualizations

Web-based interactive visualizations for exploring sexual arousal neuroscience.

## 🎯 Quick Start

**Open in Browser**: Simply open `index.html` in any modern web browser to access the visualization gallery.

Or open individual visualizations directly:
- `brain_network.html` - Interactive brain network graph
- `arousal_dynamics.html` - Dual control model dashboard
- `neurotransmitter_timeline.html` - Neurochemical timeline

**No installation required!** All visualizations run directly in your browser using CDN-hosted libraries.

---

## 📊 Visualizations

### 1. Brain Network (`brain_network.html`)

**Technology**: D3.js force-directed graph
**Features**:
- Interactive network of 12 brain regions
- Color-coded by system (cortical, limbic, subcortical, brainstem)
- Connection types: excitatory, inhibitory, modulatory
- Drag nodes, zoom, filter connections
- Click for detailed region information

**Controls**:
- **Connection Filter**: Show only specific connection types
- **Node Size**: Scale by importance or connections
- **Reset Layout**: Restart physics simulation
- **Toggle Physics**: Freeze/unfreeze the network

**Regions Included**:
- Prefrontal Cortex, Anterior Cingulate, Insula
- Amygdala, Hippocampus, Hypothalamus, Nucleus Accumbens
- VTA, Thalamus, Pituitary
- Brainstem, Spinal Cord

---

### 2. Arousal Dynamics (`arousal_dynamics.html`)

**Technology**: Plotly.js interactive plots
**Features**:
- Real-time dual control model simulation
- Adjustable parameters (α, β, γ, η)
- Dual y-axis showing E/I components and net arousal
- Stimulus visualization

**Parameters**:
- **Excitation Rate (α)**: 0.3 - 3.0 (default: 1.5)
  - How quickly arousal responds to stimuli
- **Inhibition Rate (β)**: 0.1 - 2.0 (default: 0.8)
  - How quickly inhibition builds up
- **Coupling (γ)**: 0.1 - 1.0 (default: 0.5)
  - Strength of inhibitory suppression
- **Feedback (η)**: 0.0 - 0.8 (default: 0.2)
  - Positive feedback (self-amplification)

**Clinical Insights**:
- High α, Low β: Easily aroused
- Low α, High β: Arousal difficulties
- High γ: Strong inhibitory control

---

### 3. Neurotransmitter Timeline (`neurotransmitter_timeline.html`)

**Technology**: Plotly.js time series
**Features**:
- 5 neurotransmitter trajectories
- Phase annotations (Baseline, Arousal, Orgasm, Resolution)
- Unified hover mode for cross-comparison
- Informational cards for each neurotransmitter

**Neurotransmitters Tracked**:
- **Dopamine** (red): Reward and motivation
- **Oxytocin** (blue): Bonding and orgasm
- **Prolactin** (purple): Post-orgasmic inhibition
- **Norepinephrine** (orange): Arousal and attention
- **Serotonin** (green): Mood regulation

**Timeline Phases**:
- 0-10 min: Baseline
- 10-40 min: Arousal
- 40-40.5 min: Orgasm
- 40.5-55 min: Resolution

---

## 🐍 Python Generator Script

**Location**: `../../scripts/visualization/generate_interactive_plots.py`

Generates additional Plotly visualizations (requires Python with packages installed):

**Generated Visualizations**:
1. `arousal_dynamics_dashboard.html` - Multi-stimulus comparison
2. `parameter_sensitivity.html` - 4-panel sensitivity analysis
3. `neurotransmitter_timeline.html` - (alternate version)
4. `phase_space_3d.html` - 3D E-I-Arousal trajectory
5. `individual_comparison.html` - Different individual profiles
6. `parameter_heatmap.html` - 2D parameter space exploration

**Usage**:
```bash
# Install dependencies (if needed)
pip install -r ../../scripts/requirements.txt

# Generate visualizations
cd /path/to/repository/research/neuroscience/arousal
python scripts/visualization/generate_interactive_plots.py
```

**Output**: HTML files saved to `diagrams/interactive/`

---

## 🛠️ Technologies Used

### D3.js (Data-Driven Documents)
- Version: 7.0
- Use: Brain network force-directed graph
- CDN: `https://d3js.org/d3.v7.min.js`

### Plotly.js
- Version: 2.27.0
- Use: Interactive plots, dashboards, timelines
- CDN: `https://cdn.plot.ly/plotly-2.27.0.min.js`

### HTML5 / CSS3
- Responsive design
- Modern gradients and animations
- Grid layouts

### JavaScript (ES6+)
- Mathematical simulations
- Event handling
- Data manipulation

---

## 🎨 Design Features

### Visual Design
- Modern gradient backgrounds
- Card-based layouts
- Smooth transitions and animations
- Responsive grid systems
- Hover effects

### Color Schemes
**Brain Regions**:
- Cortical: Blue (#3498db)
- Limbic: Red (#e74c3c)
- Subcortical: Green (#2ecc71)
- Brainstem: Orange (#f39c12)

**Connection Types**:
- Excitatory: Green (#27ae60)
- Inhibitory: Red (#e74c3c)
- Modulatory: Blue (#3498db)

**Neurotransmitters**:
- Dopamine: #e74c3c
- Oxytocin: #3498db
- Prolactin: #9b59b6
- Norepinephrine: #f39c12
- Serotonin: #2ecc71

### Accessibility
- High contrast text
- Clear labels and legends
- Tooltips for additional information
- Keyboard navigation (where applicable)

---

## 📱 Browser Compatibility

**Fully Supported**:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

**Requirements**:
- JavaScript enabled
- Modern CSS support (Grid, Flexbox)
- SVG rendering
- Canvas (for Plotly)

**Note**: Internet Explorer not supported

---

## 🔧 Customization

### Modifying Visualizations

All visualizations are self-contained HTML files with embedded CSS and JavaScript. To customize:

1. **Open the HTML file** in a text editor
2. **Locate the `<script>` section** for data and logic
3. **Modify parameters, colors, or data**
4. **Save and refresh** in browser

### Example: Changing Colors
```javascript
// In brain_network.html
const colorMap = {
    'cortical': '#3498db',  // Change to your preferred color
    'limbic': '#e74c3c',
    // ...
};
```

### Example: Adjusting Model Parameters
```javascript
// In arousal_dynamics.html
const model = ArousalModel({
    excitation_rate: 2.0,  // Adjust default
    inhibition_rate: 0.5,
    // ...
});
```

---

## 📚 Educational Use

These visualizations are designed for:
- **Neuroscience education**: Teaching arousal mechanisms
- **Research presentations**: Interactive demos for talks
- **Self-study**: Exploring individual differences
- **Clinical training**: Understanding dysfunctions

**Suggested Workflow**:
1. Start with `neurotransmitter_timeline.html` to understand phases
2. Explore `brain_network.html` to see anatomical connections
3. Experiment with `arousal_dynamics.html` to grasp the dual control model
4. Generate additional plots with Python script for deeper analysis

---

## 🤝 Contributing

To add new visualizations:

1. Create new HTML file in this directory
2. Use CDN libraries (D3.js, Plotly, etc.)
3. Follow existing design patterns
4. Add entry to `index.html` gallery
5. Document in this README

**Visualization Checklist**:
- [ ] Self-contained (no external dependencies except CDN)
- [ ] Responsive design
- [ ] Informational tooltips
- [ ] Legends and labels
- [ ] Consistent color scheme
- [ ] Browser tested

---

## 📖 Related Resources

**Main Documentation**:
- [Project README](../../README.md)
- [Glossary](../../docs/glossary.md)
- [Literature Review](../../docs/literature_review.md)

**Code**:
- [Python Models](../../scripts/models/)
- [Jupyter Notebook](../../notebooks/arousal_exploration.ipynb)

**Static Diagrams**:
- [Mermaid Diagrams](../)

---

## 🐛 Troubleshooting

### Visualization Not Loading
- **Check browser console** for errors (F12)
- **Ensure JavaScript is enabled**
- **Try different browser**
- **Check CDN availability** (network connection)

### Performance Issues
- **Close other tabs** (memory-intensive visualizations)
- **Disable physics** in brain network (Toggle Physics button)
- **Use smaller parameter ranges** in simulations

### Mobile Display
- **Use landscape orientation** for best experience
- **Pinch to zoom** on network visualizations
- Some controls may be condensed on small screens

---

## 📝 License

**Visualizations**: MIT License (code), CC BY 4.0 (content)
**Libraries**: See respective licenses (D3.js: BSD, Plotly: MIT)

---

## 📧 Contact & Feedback

**Issues**: Report via [GitHub Issues](https://github.com/danindiana/Deep_Convo_GPT/issues)
**Discussions**: [GitHub Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)

---

**Created**: 2025-11-19
**Version**: 1.0
**Maintainers**: Deep Convo GPT Research Team
