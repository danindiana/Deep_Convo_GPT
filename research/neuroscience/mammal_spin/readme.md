
<img width="1176" height="570" alt="Screenshot_2026-01-20_19-21-47" src="https://github.com/user-attachments/assets/d3433a64-a2fe-4b06-bc84-45ac41bcaedc" />

# Dimensional Reduction Visualization

A Python visualization tool that demonstrates the concept of "rotation" in high-dimensional neural space using Principal Component Analysis (PCA).

## Overview

This project provides an intuitive visualization of how complex changes in high-dimensional neural activation patterns can be understood as "rotations" in a lower-dimensional space. It bridges the gap between the mathematical abstraction of neural state changes and their visual representation.

## Key Features

- **High-dimensional rotation simulation**: Simulates neural activity patterns rotating in a 100-dimensional space
- **PCA projection**: Projects high-dimensional data into 3D space for visualization
- **Multi-view visualization**: Shows three complementary views simultaneously
- **Real-time animation**: Animated trajectory showing the rotation over time
- **Similarity metrics**: Tracks cosine similarity between neural states

## Visual Components

### 1. 3D PCA Projection (Left Panel)
- Shows the trajectory of neural states projected into 3D space
- Wireframe cube and unit sphere provide spatial context
- Green dot marks the starting point
- Red dot shows current position
- Blue line traces the path

### 2. Cosine Similarity Matrix (Center Panel)
- Heatmap showing pairwise cosine similarity between neural states
- Red line tracks current time point
- Reveals periodic patterns in the rotation

### 3. Neural Activity View (Right Panel)
- Heatmap showing activation levels across all 100 neurons over time
- Yellow vertical line indicates current time
- Shows the raw high-dimensional data being rotated

## Installation

### Prerequisites
- Python 3.7+
- Required packages (install via pip):

```bash
pip install numpy matplotlib scikit-learn
```

### Running the Visualization

```bash
python dimensional_reduction_viz.py
```

## How It Works

### High-Dimensional Rotation
The code simulates neural states as points on a 100-dimensional hypersphere. A rotation is applied in a 20-dimensional subspace, creating a smooth trajectory through the high-dimensional space.

### Dimensionality Reduction
Principal Component Analysis (PCA) is used to find the three most important directions (principal components) in the data. The high-dimensional trajectory is then projected onto these components for 3D visualization.

### Visualization Strategy
- The 3D view shows only ~X% of the total variance (indicated in the title)
- The similarity matrix reveals periodicity in the rotation
- The activity heatmap shows how individual neurons contribute to the rotation

## Technical Details

### Key Parameters
- `n_neurons = 100`: Number of dimensions in the simulated neural space
- `n_steps = 200`: Number of time points in the rotation
- `subspace_dim = 20`: Dimensionality of the rotation subspace
- Explained variance in 3D projection: ~X% (calculated automatically)

### Mathematical Concepts
1. **High-dimensional vectors**: Neural states represented as points on a hypersphere
2. **Cosine similarity**: Measures directional similarity between states
3. **PCA**: Linear dimensionality reduction preserving maximum variance
4. **Rotation matrices**: Orthogonal transformations preserving distances

## Educational Value

This visualization helps understand:
- Why "rotation" is a metaphor for changes in high-dimensional patterns
- How much information is lost in dimensionality reduction
- The relationship between neural similarity and geometric distance
- The periodic nature of some neural dynamics

## Extending the Project

### Possible Modifications
1. Change rotation speed: Modify the angle increment in the simulation
2. Add noise: Introduce random perturbations to simulate biological variability
3. Compare different reduction methods: Try t-SNE or UMAP instead of PCA
4. Add interactive controls: Sliders for dimension count, rotation speed, etc.

### Example Extension: Adding t-SNE
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, perplexity=30)
tsne_proj = tsne.fit_transform(trajectory)
```

## Troubleshooting

### Common Issues
1. **Points outside bounding cube**: The code includes automatic scaling to prevent this
2. **Slow animation**: Reduce `n_steps` or increase animation interval
3. **Memory issues**: Reduce `n_neurons` if running on limited hardware

### Performance Tips
- Pre-compute expensive operations (like similarity matrix)
- Use vectorized numpy operations
- Adjust frame skipping for smoother animation

## License

This project is provided for educational and research purposes. Feel free to modify and distribute with attribution.

## References

- Jolliffe, I. T. (2002). Principal Component Analysis
- Cunningham, J. P., & Yu, B. M. (2014). Dimensionality reduction for neural data
- Churchland, M. M., et al. (2012). Neural population dynamics during reaching

## Acknowledgments

This visualization was inspired by research on neural population dynamics and dimensionality reduction techniques in neuroscience.

Key improvements made:

Scaled PCA projection: Added scaling factor (0.9) to ensure points stay inside [-1, 1] cube

Added wireframe cube: Shows the bounding box clearly

Added unit sphere: Shows the "hypersphere" of possible directions

Better axis limits: Explicitly set to [-1, 1] for all axes

Color-coded starting point: Green dot shows where the trajectory began

Added similarity indicator: Red line moves through similarity matrix

Dynamic titles: Show current similarity value

Alternative simpler fix (if you just want to contain the rotation):

python
# Right after PCA, add this scaling:
pca_proj = pca.fit_transform(trajectory)

# Find the maximum absolute value and scale down
max_val = np.max(np.abs(pca_proj))
if max_val > 1:
    pca_proj = pca_proj / max_val * 0.9  # Scale to fit in [-0.9, 0.9]
The visualization now shows:

A properly bounded 3D rotation inside a visible cube

The unit sphere representing all possible directions (since vectors are normalized)

Clear boundaries so you can see when the rotation is "contained" vs "escaping"

Better context with the starting point marked in green

This makes it clearer that we're looking at a projection of a high-dimensional rotation that's been scaled to fit our visualization space!
