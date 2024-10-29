  GNU nano 6.2                                                                                                                                             smoke_test.py                                                                                                                                                      
import pybuda
import torch


# Sample PyTorch module
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(1528, 1528), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(1528, 1528), requires_grad=True)
    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1


def test_module_direct_pytorch():
    input1 = torch.rand(4, 1528, 1528)
    input2 = torch.rand(4, 1528, 1528)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")


if __name__ == "__main__":
    test_module_direct_pytorch()


The code you've provided is a Python script that tests the integration of a PyTorch module with a hypothetical library called `pybuda`. The `PyTorchTestModule` class defines a simple neural network layer with two weight matrices and a forward pass that performs matrix multiplications and additions. The `test_module_direct_pytorch` function creates random input tensors, runs the module through `pybuda`, and prints the output.

### Kernels in PyTorch

In the context of PyTorch, a "kernel" typically refers to the low-level functions that perform the actual computations on the tensors. These kernels are implemented in C++ and CUDA (for GPU computations) and are invoked when you call operations like `torch.matmul`, `torch.add`, etc.

### Other Kinds of Kernels

While the code you've provided focuses on matrix multiplication and addition, PyTorch supports a wide variety of other kernels for different operations. Here are some examples:

1. **Convolutional Kernels**:
   - `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`: These layers perform convolution operations, which are fundamental in convolutional neural networks (CNNs).

2. **Pooling Kernels**:
   - `torch.nn.MaxPool2d`, `torch.nn.AvgPool2d`: These layers perform max and average pooling operations, which reduce the spatial dimensions of the input.

3. **Activation Functions**:
   - `torch.nn.ReLU`, `torch.nn.Sigmoid`, `torch.nn.Tanh`: These layers apply activation functions to the input, introducing non-linearity into the model.

4. **Normalization Kernels**:
   - `torch.nn.BatchNorm2d`, `torch.nn.LayerNorm`: These layers normalize the input to improve training stability and convergence speed.

5. **Recurrent Kernels**:
   - `torch.nn.LSTM`, `torch.nn.GRU`: These layers implement recurrent neural networks (RNNs), which are useful for sequential data.

6. **Loss Functions**:
   - `torch.nn.CrossEntropyLoss`, `torch.nn.MSELoss`: These functions compute the loss between the model's predictions and the ground truth.

7. **Custom Kernels**:
   - PyTorch allows you to write custom kernels using `torch.autograd.Function` or by writing CUDA extensions. This is useful for implementing operations that are not natively supported by PyTorch.

### Example of Using Different Kernels

Here's an example that extends your script to include a convolutional layer and a ReLU activation function:

```python
import pybuda
import torch

# Sample PyTorch module with additional layers
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.weights1 = torch.nn.Parameter(torch.rand(1528, 1528), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(1528, 1528), requires_grad=True)
    
    def forward(self, act1, act2):
        conv_output = self.conv(act1)
        relu_output = self.relu(conv_output)
        m1 = torch.matmul(relu_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 1, 1528, 1528)  # Adjusted for Conv2d input
    input2 = torch.rand(4, 1528, 1528)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
```

In this example, the `PyTorchTestModule` now includes a `Conv2d` layer and a `ReLU` activation function. The input tensor `input1` is adjusted to match the expected shape for a 2D convolution.

### Conclusion

PyTorch provides a rich set of kernels for various operations, from basic linear algebra to complex neural network layers. By leveraging these kernels, you can build sophisticated models for a wide range of tasks, from image classification to natural language processing.



The error message you're encountering indicates that the system is trying to allocate a very large amount of memory (320 GB) and failing because it doesn't have enough available memory. This is likely due to the large tensor sizes you're working with in your PyTorch module.

### Analysis of the Error

1. **Tensor Sizes**:
   - The input tensors `input1` and `input2` have dimensions `(4, 1528, 1528)`, which is quite large. Each of these tensors requires approximately `4 * 1528 * 1528 * 4 bytes = 37.5 GB` of memory (assuming 32-bit floating-point values, which is 4 bytes each).
   - The weight tensors `weights1` and `weights2` also have dimensions `(1528, 1528)`, which require approximately `1528 * 1528 * 4 bytes = 9.3 GB` of memory each.

2. **Memory Requirements**:
   - The total memory required for the input tensors alone is `4 * 37.5 GB = 150 GB`.
   - Adding the weight tensors, the total memory requirement is `150 GB + 2 * 9.3 GB = 168.6 GB`.

Given that the system is trying to allocate over 320 GB of memory, it's clear that the memory requirements are exceeding the available resources.

### Solutions

1. **Reduce Tensor Sizes**:
   - One straightforward solution is to reduce the size of the tensors. For example, you could reduce the dimensions of the tensors to something more manageable, such as `(4, 128, 128)` or `(4, 256, 256)`.

2. **Use GPU**:
   - If you have access to a GPU with sufficient memory, you can move the computation to the GPU using `torch.cuda`. This can significantly reduce the memory pressure on the CPU.

3. **Batch Processing**:
   - Instead of processing the entire batch at once, you can process the data in smaller batches. This will reduce the memory footprint at any given time.

4. **Optimize Memory Usage**:
   - Use techniques like gradient checkpointing or memory-efficient operations to reduce the memory footprint of your computations.

### Example with Reduced Tensor Sizes

Here's an example of how you can reduce the tensor sizes:

```python
import pybuda
import torch

# Sample PyTorch module with reduced tensor sizes
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.weights1 = torch.nn.Parameter(torch.rand(256, 256), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(256, 256), requires_grad=True)
    
    def forward(self, act1, act2):
        conv_output = self.conv(act1)
        relu_output = self.relu(conv_output)
        m1 = torch.matmul(relu_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 1, 256, 256)  # Adjusted for Conv2d input
    input2 = torch.rand(4, 256, 256)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
```

In this example, the tensor sizes have been reduced to `(4, 256, 256)`, which should significantly reduce the memory requirements.

### Conclusion

The error is due to the large memory requirements of the tensors in your PyTorch module. By reducing the tensor sizes, using a GPU, or processing data in smaller batches, you can mitigate this issue. Adjusting the tensor sizes as shown in the example above should help you avoid the memory allocation error.


The logs indicate that the PyBuda installation was successful, and the compilation process proceeded without errors. The warning about "No valid grids found for node: reshape_2.dc.matmul.3 BudaOp::matmul" suggests that the balancer could not find a suitable grid configuration for a specific operation. This warning might be due to the complexity or specific requirements of the operation.

### Understanding the Warning

The warning message:
```
WARNING  | Balancer        - No valid grids found for node: reshape_2.dc.matmul.3 BudaOp::matmul [1, 1, 1024, 65536], buffer_alloc_cnt 72,  user_access_cnt 152 , operand_access_cnt 0, operand_and_user_access_cnt 0
```
indicates that the balancer could not find a valid grid configuration for the `matmul` operation with the given dimensions `[1, 1, 1024, 65536]`. This could be due to several reasons, such as:

1. **Memory Constraints**: The dimensions might be too large for the available memory on the device.
2. **Grid Configuration**: The balancer might not have found a suitable grid configuration that meets the requirements of the operation.
3. **Optimization Issues**: The operation might require specific optimizations that are not currently supported or configured.

### Handling the Warning

While this warning does not necessarily indicate a failure, it is important to understand its implications. Here are a few steps you can take to address or mitigate the warning:

1. **Check Tensor Dimensions**: Ensure that the tensor dimensions are appropriate for the available hardware resources. If necessary, reduce the dimensions to fit within the memory constraints.

2. **Optimize Operations**: Consider optimizing the operations to reduce memory usage or improve performance. For example, you can use techniques like gradient checkpointing or memory-efficient operations.

3. **Adjust Balancer Configuration**: If possible, adjust the balancer configuration or policy to better handle the specific operation.

### Example with Adjusted Tensor Dimensions

Here is an example where the tensor dimensions are adjusted to be more memory-efficient:

```python
import pybuda
import torch

# Sample PyTorch module with a convolutional layer and reduced tensor sizes
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.weights1 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
    
    def forward(self, act1, act2):
        conv_output = self.conv(act1)
        relu_output = self.relu(conv_output)
        m1 = torch.matmul(relu_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 1, 128, 128)  # Adjusted for Conv2d input
    input2 = torch.rand(4, 128, 128)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
```

### Conclusion

The warning about the balancer not finding a valid grid configuration for a specific operation is something to be aware of, but it does not necessarily indicate a failure. By adjusting the tensor dimensions and ensuring that the operations are optimized for the available hardware resources, you can mitigate such warnings and ensure smoother execution.

Certainly! Below is a version of the code that includes activation functions as part of the `PyTorchTestModule`. This example will demonstrate how to use activation functions in PyTorch.

### Activation Functions Version

```python
import pybuda
import torch

# Sample PyTorch module with activation functions
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.weights1 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(128, 128), requires_grad=True)
    
    def forward(self, act1, act2):
        conv_output = self.conv(act1)
        relu_output = self.relu(conv_output)
        sigmoid_output = self.sigmoid(relu_output)
        tanh_output = self.tanh(sigmoid_output)
        m1 = torch.matmul(tanh_output, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

def test_module_direct_pytorch():
    input1 = torch.rand(4, 1, 128, 128)  # Adjusted for Conv2d input
    input2 = torch.rand(4, 128, 128)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")

if __name__ == "__main__":
    test_module_direct_pytorch()
```

### Explanation

1. **Convolutional Layer (`self.conv`)**:
   - The `torch.nn.Conv2d` layer is added to the module. It takes an input tensor with 1 channel and applies a convolution with 32 output channels, using a 3x3 kernel with stride 1 and padding 1.

2. **Activation Functions**:
   - **ReLU (`self.relu`)**: Applies the Rectified Linear Unit (ReLU) activation function to the output of the convolutional layer.
   - **Sigmoid (`self.sigmoid`)**: Applies the Sigmoid activation function to the output of the ReLU activation.
   - **Tanh (`self.tanh`)**: Applies the Hyperbolic Tangent (Tanh) activation function to the output of the Sigmoid activation.

3. **Forward Pass**:
   - The `forward` method first applies the convolutional layer to `act1`, then applies the ReLU activation.
   - The result of the ReLU activation is then passed through the Sigmoid activation.
   - The result of the Sigmoid activation is then passed through the Tanh activation.
   - The result of the Tanh activation is then used in a matrix multiplication with `self.weights1`.
   - Another matrix multiplication is performed with `act2` and `self.weights2`.
   - The final output is the sum of the two matrix multiplication results and the result of the first matrix multiplication.

4. **Input Tensors**:
   - The input tensor `input1` is adjusted to match the expected shape for a 2D convolution, which is `(batch_size, channels, height, width)`.
   - The input tensor `input2` remains the same as in the original code.

### Running the Code

To run the code, save it to a file (e.g., `activation_smoker.py`) and execute it using Python:

```bash
python activation_smoker.py
```

This version of the code includes activation functions in the `PyTorchTestModule`, demonstrating how to use ReLU, Sigmoid, and Tanh activations in PyTorch within the context of your module.


The error message indicates that the backend compilation failed due to a constraint violation in the tensor stacking process. This is a specific issue related to the `pybuda` library and its backend, which seems to have constraints on how tensors are stacked and processed.

### Understanding the Error

The error message:
```
ERROR: TM ERROR (producer = stack_401.dc.concatenate.10.dc.concatenate.1, consumer = transpose_402.dc.sparse_matmul.4.lc2): TM order doesn't satisfy constraints for stacking with phased pipes, the number of buffered output mblocks on the producer (not counting double-buffering) must be a multiple of the total stack factor or producer t
```
suggests that the tensor stacking operation in the LSTM layer does not satisfy the constraints required by the `pybuda` backend.

### Possible Solutions

1. **Adjust LSTM Parameters**:
   - Try adjusting the parameters of the LSTM layer (e.g., `hidden_size`, `num_layers`) to see if it resolves the constraint issue.

2. **Simplify the Model**:
   - Simplify the model by reducing the complexity of the LSTM layer or by removing some layers to see if the issue persists.

3. **Check `pybuda` Documentation**:
   - Refer to the `pybuda` documentation or contact the developers to understand the specific constraints and how to work around them.

### Updated Code with Simplified LSTM

Here's an updated version of the code with a simplified LSTM layer:

```python
import pybuda
import torch

# Sample PyTorch module with a simplified recurrent layer (LSTM)
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
```

### Explanation

1. **Simplified LSTM Layer**:
   - The LSTM layer now has a `hidden_size` of 32 and only 1 layer, which reduces the complexity and might help satisfy the constraints of the `pybuda` backend.

2. **Fully Connected Layer (`self.fc`)**:
   - The fully connected layer is adjusted to map the LSTM output to a 128-dimensional space.

3. **ReLU Activation (`self.relu`)**:
   - The ReLU activation function is applied to the output of the fully connected layer.

### Running the Code

To run the code, save it to a file (e.g., `simplified_recurrent_smoker.py`) and execute it using Python:

```bash
python simplified_recurrent_smoker.py
```

This version of the code includes a simplified LSTM layer in the `PyTorchTestModule`, which might help resolve the constraint issue and allow the compilation to succeed.
