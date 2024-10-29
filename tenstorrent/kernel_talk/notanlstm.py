import pybuda
import torch

# Sample PyTorch module with a simplified recurrent layer (LSTM) and no tanh
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(32, 128)
        self.relu = torch.nn.ReLU()
        self.weights1 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
    
    def forward(self, act1, act2):
        lstm_output, _ = self.lstm(act1)
        lstm_output = self.fc(lstm_output[:, -1, :])  # Use the last time step output
        relu_output = self.relu(lstm_output)
        m1 = torch.matmul(relu_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 10, 128)  # Adjusted for LSTM input (batch_size, sequence_length, input_size)
    input2 = torch.rand(4, 128, 128)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
