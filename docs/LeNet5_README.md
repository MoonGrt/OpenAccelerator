# LeNet-5 CNN Network Implementation

## Overview

This project implements a complete LeNet-5 convolutional neural network using SpinalHDL. The network is designed for digit recognition on 28×28 grayscale images and outputs 10 class probabilities.

## Network Architecture

```
Input: 28×28 grayscale image
    ↓
Conv1: 6 filters of 5×5 → 6×24×24
    ↓
ReLU1: Activation
    ↓
Pool1: 2×2 max pooling → 6×12×12
    ↓
Conv2: 12 filters of 5×5 → 12×8×8
    ↓
ReLU2: Activation
    ↓
Pool2: 2×2 max pooling → 12×4×4
    ↓
Flatten: 12×4×4 → 192
    ↓
FC: 192 → 10 (output classes)
```

## Files Structure

```
src/main/scala/cnn/
├── Convolution.scala      # 3x3 convolution with padding/stride
├── MaxPooling.scala       # 2x2/3x3 max pooling
├── ReLU.scala            # ReLU activation functions
├── FullConnection.scala   # Fully connected layer
├── LeNet5.scala          # Basic LeNet-5 implementation
└── LeNet5Complete.scala  # Complete multi-channel LeNet-5

src/test/scala/cnn/
├── CNNModulesTest.scala  # Individual module tests
└── LeNet5Test.scala      # Complete LeNet-5 test
```

## Key Features

### 1. Multi-Channel Support
- **MultiChannelConv**: Handles multiple input/output channels
- **MultiChannelPool**: Processes multiple channels in parallel
- **MultiChannelReLU**: Applies activation to all channels

### 2. Stream-Based Interface
- All modules use SpinalHDL Stream interface
- Easy integration and data flow control
- Backpressure support

### 3. Configurable Parameters
- Data width: 8-bit (configurable)
- Weight width: 8-bit (configurable)
- Input size: 28×28 (configurable)
- Number of classes: 10 (configurable)

### 4. Weight Management
- External weight interfaces for all layers
- Support for pre-trained weights
- Bias terms for all layers

## Usage

### 1. Basic Usage

```scala
// Create configuration
val config = LeNet5CompleteConfig(
  dataWidth = 8,
  weightWidth = 8,
  biasWidth = 8,
  inputSize = 28,
  numClasses = 10,
  useBias = true,
  signed = false,
  quantization = false
)

// Instantiate network
val lenet5 = new LeNet5Complete(UInt(8 bits), config)
```

### 2. Weight Loading

```scala
// Load pre-trained weights
for (i <- 0 until 6) {
  for (j <- 0 until 25) {
    lenet5.io.conv1_weights(i)(j) := conv1_weights(i)(j)
  }
  lenet5.io.conv1_bias(i) := conv1_bias(i)
}

// Similar for conv2 and fc layers
```

### 3. Data Processing

```scala
// Input: Stream 28×28 pixel values
lenet5.io.input.valid := pixel_valid
lenet5.io.input.payload := pixel_value

// Output: 10 class probabilities
val class_probs = lenet5.io.output.payload
val output_valid = lenet5.io.output.valid
```

## Parameter Count

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Conv1 | 150     | 6      | 156   |
| Conv2 | 1,800   | 12     | 1,812 |
| FC    | 1,920   | 10     | 1,930 |
| **Total** | **3,870** | **28** | **3,898** |

## Implementation Details

### Convolution Layers
- **Conv1**: 6 filters, 5×5 kernel, no padding
- **Conv2**: 12 filters, 5×5 kernel, no padding
- Support for padding and stride parameters
- Matrix-based convolution computation

### Pooling Layers
- **Pool1**: 2×2 max pooling, stride 2
- **Pool2**: 2×2 max pooling, stride 2
- Configurable kernel size and stride

### Activation Functions
- **ReLU**: f(x) = max(0, x)
- Applied after each convolution layer
- Support for other activation types (Leaky ReLU, ELU, etc.)

### Fully Connected Layer
- **Input**: 192 (flattened from 12×4×4)
- **Output**: 10 (class probabilities)
- Matrix multiplication with bias addition
- Support for quantization

## Testing

### Individual Module Tests
```bash
sbt "runMain cnn.CNNModulesTest"
```

### Complete LeNet-5 Test
```bash
sbt "runMain cnn.LeNet5Test"
```

### Generate Verilog
```bash
sbt "runMain cnn.LeNet5CompleteGen"
```

## Performance Considerations

### 1. Parallel Processing
- Multiple channels processed in parallel
- Stream-based pipeline for high throughput
- Configurable data width for precision/speed trade-off

### 2. Memory Usage
- Line buffers for convolution operations
- Weight storage in external memory
- Minimal internal state

### 3. Timing
- Pipeline stages for better clock frequency
- Stream handshaking for data synchronization
- Configurable quantization for speed optimization

## Extensions

### 1. Additional Activation Functions
- Leaky ReLU: f(x) = max(αx, x)
- Parametric ReLU: Learnable α parameter
- ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0

### 2. Quantization Support
- Configurable weight and activation quantization
- Support for different bit widths
- Dynamic quantization during inference

### 3. Batch Processing
- Support for batch input processing
- Parallel processing of multiple images
- Optimized memory access patterns

## Example Applications

### 1. MNIST Digit Recognition
- Input: 28×28 grayscale digit images
- Output: 10 class probabilities (0-9)
- Pre-trained weights available

### 2. Custom Classification
- Modify input size for different image dimensions
- Adjust number of classes for different tasks
- Retrain with custom datasets

### 3. Hardware Acceleration
- FPGA implementation for real-time processing
- ASIC design for embedded applications
- High-throughput inference engine

## Troubleshooting

### Common Issues

1. **Stream Backpressure**: Ensure downstream modules can accept data
2. **Weight Loading**: Verify weight dimensions match layer specifications
3. **Timing Violations**: Check clock frequency and pipeline stages
4. **Memory Overflow**: Monitor line buffer sizes for large images

### Debug Features

- Stream valid/ready signal monitoring
- Weight and bias value verification
- Layer output validation
- Performance profiling

## Future Improvements

1. **Advanced Architectures**: Support for ResNet, VGG, etc.
2. **Training Integration**: On-chip training capabilities
3. **Dynamic Networks**: Runtime reconfigurable architectures
4. **Optimization**: Pruning, quantization, compression
5. **Multi-FPGA**: Distributed processing across multiple devices

## References

- LeCun, Y., et al. "Gradient-based learning applied to document recognition." (1998)
- SpinalHDL Documentation: https://spinalhdl.github.io/SpinalDoc-RTD/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
