import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

# High-dimensional rotation simulation
def simulate_high_dim_rotation(n_neurons=100, n_steps=100):
    """Simulate rotation in high-dimensional space."""
    # Random initial direction
    np.random.seed(42)
    vec = np.random.randn(n_neurons)
    vec /= np.linalg.norm(vec)
    
    # Create rotation matrix in random subspace
    subspace_dim = 20  # Rotate in 20D subspace
    subspace = np.random.randn(n_neurons, subspace_dim)
    subspace, _ = np.linalg.qr(subspace)  # Orthonormal basis
    
    # Small rotations in subspace
    trajectory = []
    for i in range(n_steps):
        angle = 2 * np.pi * i / n_steps
        # Create rotation in subspace
        rot_subspace = np.eye(subspace_dim)
        # Rotate first two dimensions of subspace
        rot_subspace[0, 0] = np.cos(angle)
        rot_subspace[0, 1] = -np.sin(angle)
        rot_subspace[1, 0] = np.sin(angle)
        rot_subspace[1, 1] = np.cos(angle)
        
        # Apply to full space
        vec = subspace @ rot_subspace @ subspace.T @ vec
        vec /= np.linalg.norm(vec)
        trajectory.append(vec)
    
    return np.array(trajectory)

# Generate trajectory
trajectory = simulate_high_dim_rotation(100, 200)

# PCA to visualize
pca = PCA(n_components=3)
pca_proj = pca.fit_transform(trajectory)

# Scale PCA projection to fit nicely in [-1, 1] cube
scale_factor = 0.9 / np.max(np.abs(pca_proj))  # 0.9 to leave some margin
pca_proj_scaled = pca_proj * scale_factor

# Calculate explained variance
explained_var = pca.explained_variance_ratio_[:3].sum() * 100

fig = plt.figure(figsize=(14, 6))

# Original high-D view
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title(f'3D PCA Projection\n({explained_var:.1f}% of variance)', fontweight='bold')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

# Set axis limits to nice cube
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])

# Add grid and cube edges for reference
# Draw a wireframe cube
cube_edges = [
    [[-1, -1, -1], [1, -1, -1]],
    [[1, -1, -1], [1, 1, -1]],
    [[1, 1, -1], [-1, 1, -1]],
    [[-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, 1], [1, -1, 1]],
    [[1, -1, 1], [1, 1, 1]],
    [[1, 1, 1], [-1, 1, 1]],
    [[-1, 1, 1], [-1, -1, 1]],
    [[-1, -1, -1], [-1, -1, 1]],
    [[1, -1, -1], [1, -1, 1]],
    [[1, 1, -1], [1, 1, 1]],
    [[-1, 1, -1], [-1, 1, 1]]
]

for edge in cube_edges:
    x = [edge[0][0], edge[1][0]]
    y = [edge[0][1], edge[1][1]]
    z = [edge[0][2], edge[1][2]]
    ax1.plot(x, y, z, 'gray', alpha=0.3, linewidth=0.5)

# High-D similarity matrix
ax2 = fig.add_subplot(132)
ax2.set_title('High-D Cosine Similarity Matrix', fontweight='bold')
ax2.set_xlabel('Time point')
ax2.set_ylabel('Time point')

# Dimensional view
ax3 = fig.add_subplot(133)
ax3.set_title(f'Activity in All {trajectory.shape[1]} Dimensions', fontweight='bold')
ax3.set_xlabel('Time')
ax3.set_ylabel('Neuron index')

# Plot initial states
# PCA trajectory - plot as scatter points for smoother update
traj_line, = ax1.plot([], [], [], 'b-', alpha=0.5, linewidth=2)
current_point, = ax1.plot([], [], [], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1)

# Plot starting point
ax1.scatter(pca_proj_scaled[0, 0], pca_proj_scaled[0, 1], pca_proj_scaled[0, 2], 
            color='green', s=50, label='Start', alpha=0.7)

# Add unit sphere wireframe for context
phi = np.linspace(0, 2*np.pi, 30)
theta = np.linspace(0, np.pi, 30)

x_sphere = np.outer(np.cos(phi), np.sin(theta))
y_sphere = np.outer(np.sin(phi), np.sin(theta))
z_sphere = np.outer(np.ones_like(phi), np.cos(theta))

ax1.plot_surface(x_sphere, y_sphere, z_sphere, 
                 alpha=0.05, color='gray', linewidth=0, antialiased=True)

ax1.legend(loc='upper right')

# Similarity matrix (pre-compute)
similarity_matrix = trajectory @ trajectory.T
sim_img = ax2.imshow(similarity_matrix[:50, :50], cmap='viridis', 
                     aspect='auto', vmin=-1, vmax=1,
                     extent=[0, 50, 50, 0])
plt.colorbar(sim_img, ax=ax2, label='Cosine similarity')

# High-D activity
activity_img = ax3.imshow(trajectory.T, cmap='RdBu_r', aspect='auto',
                          vmin=-0.5, vmax=0.5)
plt.colorbar(activity_img, ax=ax3, label='Activation')

# Time indicator
time_line = ax3.axvline(x=0, color='yellow', linewidth=2, alpha=0.8)

# Add similarity indicator line on similarity matrix
sim_time_line = ax2.axhline(y=0, color='red', linewidth=2, alpha=0.8)

def update(frame):
    frame_idx = frame % len(pca_proj_scaled)
    
    # Update PCA plot - ensure we stay within bounds
    traj_line.set_data(pca_proj_scaled[:frame_idx+1, 0], pca_proj_scaled[:frame_idx+1, 1])
    traj_line.set_3d_properties(pca_proj_scaled[:frame_idx+1, 2])
    
    current_point.set_data([pca_proj_scaled[frame_idx, 0]], [pca_proj_scaled[frame_idx, 1]])
    current_point.set_3d_properties([pca_proj_scaled[frame_idx, 2]])
    
    # Update time indicators
    time_line.set_xdata([frame_idx, frame_idx])
    sim_time_line.set_ydata([frame_idx, frame_idx])
    
    # Update title with variance info
    current_similarity = np.dot(trajectory[0], trajectory[frame_idx])
    ax1.set_title(f'3D PCA ({explained_var:.1f}% variance)\nSimilarity: {current_similarity:.3f}', 
                  fontweight='bold')
    
    return traj_line, current_point, time_line, sim_time_line

ani = FuncAnimation(fig, update, frames=range(0, 200, 2), 
                    interval=50, blit=False)

plt.tight_layout()
plt.show()
