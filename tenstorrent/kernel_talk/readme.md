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
