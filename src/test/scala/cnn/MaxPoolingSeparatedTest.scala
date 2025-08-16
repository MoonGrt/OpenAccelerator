package cnn

import spinal.core._
import spinal.lib._

object MaxPoolingSeparatedTest {
  def main(args: Array[String]): Unit = {
    println("Testing separated MaxPooling modules...")
    
    // Test 2x2 MaxPooling
    println("1. Testing 2x2 MaxPooling...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new MaxPooling2x2(UInt(8 bits), MaxPool2x2Config(
        dataWidth = 8,
        lineLength = 32, // Small test size
        padding = 0,
        stride = 2))
    ).printPruned()
    
    // Test 3x3 MaxPooling
    println("2. Testing 3x3 MaxPooling...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new MaxPooling3x3(UInt(8 bits), MaxPool3x3Config(
        dataWidth = 8,
        lineLength = 32, // Small test size
        padding = 1,
        stride = 2))
    ).printPruned()
    
    // Test 2x2 MaxPooling with stride 1
    println("3. Testing 2x2 MaxPooling with stride 1...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new MaxPooling2x2(UInt(8 bits), MaxPool2x2Config(
        dataWidth = 8,
        lineLength = 32,
        padding = 0,
        stride = 1))
    ).printPruned()
    
    // Test 3x3 MaxPooling with stride 1
    println("4. Testing 3x3 MaxPooling with stride 1...")
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new MaxPooling3x3(UInt(8 bits), MaxPool3x3Config(
        dataWidth = 8,
        lineLength = 32,
        padding = 1,
        stride = 1))
    ).printPruned()
    
    println("All separated MaxPooling modules generated successfully!")
    
    // Print detailed information
    println("\n=== Separated MaxPooling Architecture ===")
    println("2x2 MaxPooling:")
    println("  - Kernel: 2×2")
    println("  - Matrix Interface: Matrix2x2Interface")
    println("  - Matrix Builder: Matrix2x2Builder")
    println("  - Max Pooling: MaxPooling2x2")
    println("  - Config: MaxPool2x2Config")
    println("  - Padding: 0 (no padding)")
    println("  - Stride: 1 or 2")
    
    println("\n3x3 MaxPooling:")
    println("  - Kernel: 3×3")
    println("  - Matrix Interface: Matrix3x3Interface")
    println("  - Matrix Builder: Matrix3x3Builder")
    println("  - Max Pooling: MaxPooling3x3")
    println("  - Config: MaxPool3x3Config")
    println("  - Padding: 0 or 1")
    println("  - Stride: 1 or 2")
    
    println("\n=== Key Differences ===")
    println("1. Separate Matrix Interfaces:")
    println("   - Matrix2x2Interface: m11, m12, m21, m22")
    println("   - Matrix3x3Interface: m11, m12, m13, m21, m22, m23, m31, m32, m33")
    
    println("\n2. Separate Matrix Builders:")
    println("   - Matrix2x2Builder: 2 line buffers, 4 shift registers")
    println("   - Matrix3x3Builder: 2 line buffers + 1 register, 9 shift registers")
    
    println("\n3. Separate Max Pooling Logic:")
    println("   - maxPool2x2: Compare 4 values")
    println("   - maxPool3x3: Compare 9 values")
    
    println("\n4. Separate Configurations:")
    println("   - MaxPool2x2Config: No kernelSize parameter")
    println("   - MaxPool3x3Config: No kernelSize parameter")
    
    println("\n=== Benefits ===")
    println("✓ Clear separation of concerns")
    println("✓ Optimized implementations for each kernel size")
    println("✓ Type safety (no runtime kernel size checks)")
    println("✓ Better resource utilization")
    println("✓ Easier to understand and maintain")
    println("✓ Reduced complexity in each class")
    
    println("\n=== Usage Examples ===")
    println("// 2x2 MaxPooling")
    println("val pool2x2 = new MaxPooling2x2(UInt(8 bits), MaxPool2x2Config(")
    println("  dataWidth = 8,")
    println("  lineLength = 24,")
    println("  padding = 0,")
    println("  stride = 2")
    println("))")
    
    println("\n// 3x3 MaxPooling")
    println("val pool3x3 = new MaxPooling3x3(UInt(8 bits), MaxPool3x3Config(")
    println("  dataWidth = 8,")
    println("  lineLength = 24,")
    println("  padding = 1,")
    println("  stride = 2")
    println("))")
  }
}
