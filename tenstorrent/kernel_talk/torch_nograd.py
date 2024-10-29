import pybuda
import torch
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure the directory for saving visualizations exists
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# Sample PyTorch module with convolutional and normalization layers
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)  # Adjusted to match the actual size
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.layernorm = torch.nn.LayerNorm(128)
    
    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            visualize_activation_maps(x)  # Save high-resolution activation map visualization
            
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = x.view(-1, 64 * 4 * 4)  # Adjusted to match the actual size
            
            x = self.fc1(x)
            x = self.relu(x)
            x = self.layernorm(x)
            x = self.fc2(x)
            x = self.softmax(x)
        except RuntimeError as e:
            logging.error(f"Error in forward pass: {e}")
            raise
        return x

def show_input_images(input_tensor):
    """Save input images to a file with increased size and resolution for better visibility."""
    try:
        fig, axes = plt.subplots(1, input_tensor.size(0), figsize=(12, 6), dpi=150)
        for i in range(input_tensor.size(0)):
            image = input_tensor[i, 0].cpu().numpy()  # Assuming single-channel images
            axes[i].imshow(image, cmap='gray')
            axes[i].axis('off')
        plt.savefig(os.path.join(output_dir, "input_images.png"))
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error in show_input_images: {e}")

def visualize_activation_maps(activation_tensor):
    """Save high-resolution activation maps (feature maps) from a given layer to a file with filter titles."""
    try:
        fig, axes = plt.subplots(4, 8, figsize=(16, 16), dpi=200)
        for i, ax in enumerate(axes.flat):
            if i < activation_tensor.size(1):
                ax.imshow(activation_tensor[0, i].detach().cpu().numpy(), cmap='viridis')
                ax.set_title(f'Filter {i}', fontsize=8)  # Add title to each subplot
            ax.axis('off')
        plt.savefig(os.path.join(output_dir, "activation_maps_high_res.png"))
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error in visualize_activation_maps: {e}")

def plot_output_probabilities(output_tensor):
    """Save output softmax probabilities for each input to individual files."""
    try:
        for i in range(output_tensor.size(0)):  # Loop through each batch item
            fig, ax = plt.subplots()
            ax.bar(range(10), output_tensor[i].detach().cpu().numpy())  # Detach before converting to NumPy
            ax.set_title(f'Sample {i} - Class Probabilities')
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
            plt.savefig(os.path.join(output_dir, f"output_probabilities_sample_{i}.png"))
            plt.close(fig)
    except Exception as e:
        logging.error(f"Error in plot_output_probabilities: {e}")

def test_module_direct_pytorch():
    # Generate example input tensor, and handle potential errors if shape mismatch occurs
    try:
        input_tensor = torch.rand(4, 1, 16, 16)  # Example input tensor
        show_input_images(input_tensor)  # Save input images visualization
        
        model = PyTorchTestModule()
        output = model(input_tensor)
        
        plot_output_probabilities(output)  # Save softmax output probabilities visualization

        # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
        with torch.no_grad():
            output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input_tensor)
        print(output)
        print("PyBuda installation was a success!")

    except RuntimeError as e:
        logging.error(f"An error occurred while running the test module: {e}")

if __name__ == "__main__":
    test_module_direct_pytorch()
