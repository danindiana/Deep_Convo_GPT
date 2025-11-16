import pybuda
import torch

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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 64 * 4 * 4)  # Adjusted to match the actual size
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def test_module_direct_pytorch():
    input_tensor = torch.rand(4, 1, 16, 16)  # Adjusted for Conv2d input (batch_size, channels, height, width)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input_tensor)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
