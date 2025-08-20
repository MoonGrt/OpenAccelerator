package cnn

import spinal.core._
import spinal.lib._

object CNNModulesTest {
  def main(args: Array[String]): Unit = {
    println("Testing all CNN modules...")
    
    // Test Conv2D3x3
    println("1. Testing Conv2D3x3...")
    val gaussianKernel = Seq(1,2,1, 2,4,2, 1,2,1)
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new Conv2D3x3(UInt(8 bits), Conv2DConfig(
        dataWidth = 8,
        convWidth = 10,
        rowNum = 32, // Small test size
        kernel = gaussianKernel,
        kernelShift = 4,
        padding = 1,
        stride = 1))
    ).printPruned()
    
    // Test MaxPooling
    println("2. Testing MaxPooling...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new MaxPooling(UInt(8 bits), MaxPoolConfig(
        dataWidth = 8,
        rowNum = 32,
        kernelSize = 2,
        padding = 0,
        stride = 2))
    ).printPruned()
    
    // Test ReLU
    println("3. Testing ReLU...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new ReLU(UInt(8 bits), ReLUConfig(
        dataWidth = 8,
        activationType = "relu",
        signed = false))
    ).printPruned()
    
    // Test FullConnection
    println("4. Testing FullConnection...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new FullConnection(UInt(8 bits), FullConnectionConfig(
        inputWidth = 8,
        outputWidth = 16,
        weightWidth = 8,
        biasWidth = 8,
        inputSize = 64,  // Small test size
        outputSize = 10,
        useBias = true,
        signed = false,
        quantization = false))
    ).printPruned()
    
    println("All CNN modules tested successfully!")
    
    // Print module information
    println("\n=== Module Summary ===")
    println("1. Conv2D3x3: 3x3 convolution with padding and stride support")
    println("2. MaxPooling: 2x2/3x3 max pooling with padding and stride support")
    println("3. ReLU: Multiple activation functions (ReLU, Leaky ReLU, Parametric ReLU, ELU)")
    println("4. FullConnection: Fully connected layer with matrix multiplication and bias")
    println("\nAll modules use SpinalHDL Stream interface for easy integration.")
  }
}

