import pybuda
import torch

# Sample PyTorch module with a convolutional layer and reduced tensor sizes
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.weights1 = torch.nn.Parameter(torch.rand(28, 28), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(28, 28), requires_grad=True)
    
    def forward(self, act1, act2):
        conv_output = self.conv(act1)
        relu_output = self.relu(conv_output)
        m1 = torch.matmul(relu_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 1, 28, 28)  # Adjusted for Conv2d input
    input2 = torch.rand(4, 28, 28)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
