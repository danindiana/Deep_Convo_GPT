import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import datetime

# Parameters
num_neurons = 50
num_nanobots = 100
space_size = 10
steps = 100
step_size = 0.5

# Initialize Neurons
np.random.seed(42)
neurons = np.random.uniform(-space_size, space_size, (num_neurons, 3))

# Initialize Nanobots
nanobots = np.random.uniform(-space_size, space_size, (num_nanobots, 3))

# Function to update nanobot positions
def update_nanobots(nanobots, neurons, step_size, mode, frame):
    for i in range(len(nanobots)):
        if mode == 1:  # Random movement
            direction = neurons[np.random.randint(0, len(neurons))] - nanobots[i]
        elif mode == 2:  # Target nearest neuron
            distances = np.linalg.norm(neurons - nanobots[i], axis=1)
            nearest_neuron = neurons[np.argmin(distances)]
            direction = nearest_neuron - nanobots[i]
        elif mode == 3:  # Cluster-based movement
            cluster_center = np.mean(neurons, axis=0)
            direction = cluster_center - nanobots[i]
        elif mode == 4:  # Broadcast packet visualization
            if frame % 10 == 0:  # Simulate broadcasting every 10 frames
                direction = neurons[np.random.randint(0, len(neurons))] - nanobots[i]
            else:
                direction = np.zeros(3)  # No movement for non-broadcast frames
        else:
            direction = np.random.uniform(-1, 1, 3)  # Default random movement

        if np.linalg.norm(direction) > 0:
            nanobots[i] += step_size * direction / np.linalg.norm(direction)
    return nanobots

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-space_size, space_size])
ax.set_ylim([-space_size, space_size])
ax.set_zlim([-space_size, space_size])
ax.set_title("Nanobot Swarm and Neuron Interaction")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Use gradient rainbow colors for neurons and nanobots
neuron_colors = plt.cm.rainbow(np.linspace(0, 1, num_neurons))
nanobot_colors = plt.cm.rainbow(np.linspace(0, 1, num_nanobots))

neuron_scatter = ax.scatter(neurons[:, 0], neurons[:, 1], neurons[:, 2], c=neuron_colors, label='Neurons')
nanobot_scatter = ax.scatter(nanobots[:, 0], nanobots[:, 1], nanobots[:, 2], c=nanobot_colors, label='Nanobots')

# Prompt user for interaction mode
print("Choose a nanobot interaction mode:")
print("1. Random movement towards neurons")
print("2. Target nearest neuron")
print("3. Move towards cluster center of neurons")
print("4. Broadcast packet visualization (wave simulation)")
interaction_mode = int(input("Enter your choice (1/2/3/4): ").strip())

# Update function for animation
def update(frame):
    global nanobots
    nanobots = update_nanobots(nanobots, neurons, step_size, interaction_mode, frame)
    nanobot_scatter._offsets3d = (nanobots[:, 0], nanobots[:, 1], nanobots[:, 2])
    if interaction_mode == 4 and frame % 10 == 0:
        ax.scatter(nanobots[:, 0], nanobots[:, 1], nanobots[:, 2], c='cyan', alpha=0.5, label='Broadcast Wave')

# Ask user if they want to save the animation
save_animation = input("Do you want to save the animation to a file? (yes/no): ").strip().lower()

if save_animation == 'yes':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nanobot_swarm_{timestamp}.mp4"
    print(f"Saving animation to {filename}...")
    ani = FuncAnimation(fig, update, frames=steps, interval=100)
    ani.save(filename, writer='ffmpeg')
    print("Animation saved successfully.")
else:
    # Display the animation interactively
    print("Displaying animation interactively.")
    ani = FuncAnimation(fig, update, frames=steps, interval=100)
    plt.show()
