package cnn

import spinal.core._
import spinal.lib._

object LeNet5Test {
  def main(args: Array[String]): Unit = {
    println("Testing complete LeNet-5 CNN network...")
    
    // Test configuration
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
    
    // Generate complete LeNet-5
    println("1. Generating complete LeNet-5...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new LeNet5Complete(UInt(8 bits), config)
    ).printPruned()
    
    // Generate test bench
    println("2. Generating LeNet-5 test bench...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new LeNet5TestBench()
    ).printPruned()
    
    // Generate simplified LeNet-5
    println("3. Generating simplified LeNet-5...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new SimpleLeNet5(UInt(8 bits), config)
    ).printPruned()
    
    println("All LeNet-5 variants generated successfully!")
    
    // Print detailed architecture information
    println("\n=== LeNet-5 Network Architecture ===")
    println("Input Layer:")
    println("  - Size: 28×28 grayscale image")
    println("  - Data width: 8 bits")
    println("  - Total pixels: 784")
    
    println("\nConvolution Layer 1:")
    println("  - Filters: 6 filters of 5×5")
    println("  - Padding: 0")
    println("  - Stride: 1")
    println("  - Output: 6×24×24")
    println("  - Parameters: 6 × 25 = 150 weights + 6 biases")
    
    println("\nReLU Activation 1:")
    println("  - Type: ReLU")
    println("  - Input: 6×24×24")
    println("  - Output: 6×24×24")
    
    println("\nMax Pooling Layer 1:")
    println("  - Kernel: 2×2")
    println("  - Stride: 2")
    println("  - Output: 6×12×12")
    
    println("\nConvolution Layer 2:")
    println("  - Filters: 12 filters of 5×5")
    println("  - Input channels: 6")
    println("  - Padding: 0")
    println("  - Stride: 1")
    println("  - Output: 12×8×8")
    println("  - Parameters: 12 × 150 = 1,800 weights + 12 biases")
    
    println("\nReLU Activation 2:")
    println("  - Type: ReLU")
    println("  - Input: 12×8×8")
    println("  - Output: 12×8×8")
    
    println("\nMax Pooling Layer 2:")
    println("  - Kernel: 2×2")
    println("  - Stride: 2")
    println("  - Output: 12×4×4")
    
    println("\nFlatten Layer:")
    println("  - Input: 12×4×4")
    println("  - Output: 192")
    
    println("\nFully Connected Layer:")
    println("  - Input: 192")
    println("  - Output: 10 (classes)")
    println("  - Parameters: 192 × 10 = 1,920 weights + 10 biases")
    
    println("\n=== Total Parameters ===")
    println("Conv1: 150 + 6 = 156")
    println("Conv2: 1,800 + 12 = 1,812")
    println("FC: 1,920 + 10 = 1,930")
    println("Total: 3,898 parameters")
    
    println("\n=== Implementation Features ===")
    println("✓ Multi-channel convolution support")
    println("✓ Multi-channel pooling support")
    println("✓ Multi-channel ReLU activation")
    println("✓ Proper weight and bias management")
    println("✓ Stream-based data flow")
    println("✓ Configurable parameters")
    println("✓ Padding and stride support")
    println("✓ Quantization support")
    println("✓ Test bench included")
    
    println("\n=== Usage Instructions ===")
    println("1. Load pre-trained weights into conv1_weights, conv1_bias")
    println("2. Load pre-trained weights into conv2_weights, conv2_bias")
    println("3. Load pre-trained weights into fc_weights, fc_bias")
    println("4. Stream 28×28 pixel values through input")
    println("5. Read 10 class probabilities from output")
  }
}
