package cnn

import spinal.core._
import spinal.lib._

/**
 * LeNet-5 CNN Network Configuration
 *
 * Network Architecture:
 * Input: 28×28 grayscale image
 * Conv1: 6 filters of 5×5 -> Output: 6×24×24
 * Pool1: 2×2 max pooling -> Output: 6×12×12
 * Conv2: 12 filters of 5×5 -> Output: 12×8×8
 * Pool2: 2×2 max pooling -> Output: 12×4×4
 * Flatten: 12×4×4 -> 192
 * FC1: 192 -> 10 (output classes)
 */
case class LeNet5Config(
  dataWidth    : Int = 8,          // bits per pixel
  weightWidth  : Int = 8,          // bits per weight
  biasWidth    : Int = 8,          // bits per bias
  inputSize    : Int = 28,         // input image size
  numClasses   : Int = 10,         // number of output classes
  useBias      : Boolean = true,   // whether to use bias
  signed       : Boolean = false,  // whether data is signed
  quantization : Boolean = false   // whether to use quantization
)

/**
 * Simplified LeNet-5 for demonstration (single channel processing)
 */
class SimpleLeNet5(config: LeNet5Config) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(SInt(dataWidth bits)))
  }

  // ============================================================================
  // Convolution Layer 1: 5×5 -> Output: 24×24
  // ============================================================================
  // Gaussian kernel: 1 4 6 4 1 / 4 16 24 16 4 / 6 24 36 24 6 / 4 16 24 16 4 / 1 4 6 4 1
  val gaussianKernel = Seq(1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1)
  val conv1_config = Conv2DConfig(
        dataWidth = 8,
        convWidth = 8,
        lineLength = 28,
        kernel = gaussianKernel,
        kernelShift = 4,
        kernelSize = 5,
        insigned = false,
        padding = 1,
        stride = 1)
  val conv1 = new Conv2DDyn(conv1_config)
  conv1.io.EN := io.EN
  conv1.io.pre <> io.input
  conv1.io.LINEWIDTH := 28

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val relu1 = new ReLU(ReLUConfig(dataWidth, "relu"))
  relu1.io.EN := io.EN
  relu1.io.pre <> conv1.io.post

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 12×12
  // ============================================================================
  val pool1_config = MaxPool2x2Config(
    dataWidth = dataWidth,
    lineLength = 24,
    padding = 0,
    stride = 2
  )
  val pool1 = new MaxPooling2x2(pool1_config)
  pool1.io.EN := io.EN
  pool1.io.pre <> relu1.io.post
  // pool1.io.LINEWIDTH := 24

  // ============================================================================
  // Convolution Layer 2: 5×5 -> Output: 8×8
  // ============================================================================
  val meanKernel = Seq(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1)
  val conv2_config = Conv2DConfig(
        dataWidth = dataWidth,
        convWidth = dataWidth,
        lineLength = 12,
        kernel = meanKernel,
        kernelShift = 4,
        kernelSize = 5,
        padding = 1,
        stride = 1)
  val conv2 = new Conv2DDyn(conv2_config)
  conv2.io.EN := io.EN
  conv2.io.pre <> pool1.io.post
  conv2.io.LINEWIDTH := 12

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val relu2 = new ReLU(ReLUConfig(dataWidth, "relu"))
  relu2.io.EN := io.EN
  relu2.io.pre <> conv2.io.post

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 4×4
  // ============================================================================
  val pool2_config = MaxPool2x2Config(
    dataWidth = dataWidth,
    lineLength = 8,
    padding = 0,
    stride = 2
  )
  val pool2 = new MaxPooling2x2(pool2_config)
  pool2.io.EN := io.EN
  pool2.io.pre <> relu2.io.post
  // pool2.io.LINEWIDTH := 8

  // // ============================================================================
  // // Flatten Layer: 4×4 -> 16
  // // ============================================================================
  // val flatten_output = Stream(Vec(SInt(dataWidth bits), 16))

  // // Simplified flattening - collect 4x4 values
  // val flatten_valid = Reg(Bool()) init(False)
  // val flatten_data = Reg(Vec(SInt(dataWidth bits), 16))

  // flatten_valid := pool2.io.post.valid
  // flatten_data(0) := pool2.io.post.payload

  // flatten_output.valid := flatten_valid
  // flatten_output.payload := flatten_data

  // ============================================================================
  // Fully Connected Layer: 16 -> 10
  // ============================================================================
  val fc_config = FullConnectionConfig(
    inputWidth = dataWidth,
    outputWidth = dataWidth,
    weightWidth = weightWidth,
    biasWidth = biasWidth,
    inputSize = 16,
    outputSize = numClasses,
    useBias = useBias,
    signed = signed,
    quantization = quantization
  )
  val fc = new FullConnectionStream(fc_config)
  fc.io.EN := io.EN
  fc.io.pre <> pool2.io.post
  fc.io.wb.weight.valid := False
  fc.io.wb.weight.payload := 0
  fc.io.wb.bias.valid := False
  fc.io.wb.bias.payload := 0

  // ============================================================================
  // Output
  // ============================================================================
  io.output <> fc.io.post
}

// /**
//  * LeNet-5 CNN Network
//  */
// class LeNet5(config: LeNet5Config) extends Component {
//   import config._
//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(SInt(dataWidth bits)))
//     val output = master(Stream(Vec(SInt(dataWidth bits), numClasses)))
//     val conv1_weights = in(Vec(Vec(SInt(weightWidth bits), 25), 6))  // 6 filters, 5x5 each
//     val conv1_bias = in(Vec(SInt(biasWidth bits), 6))
//     val conv2_weights = in(Vec(Vec(SInt(weightWidth bits), 150), 12)) // 12 filters, 6*5*5 each
//     val conv2_bias = in(Vec(SInt(biasWidth bits), 12))
//     val fc_weights = in(Vec(SInt(weightWidth bits), 192 * numClasses))
//     val fc_bias = in(Vec(SInt(biasWidth bits), numClasses))
//   }

//   // Ready signal
//   io.input.ready := True

//   // ============================================================================
//   // Convolution Layer 1: 6 filters of 5×5 -> Output: 6×24×24
//   // ============================================================================
//   val conv1_outputs = Vec(Stream(SInt(dataWidth bits)), 6)
//   val conv1_config = Conv2DConfig(
//     dataWidth = dataWidth,
//     convWidth = dataWidth,
//     lineLength = inputSize,
//     kernel = Seq(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1), // Placeholder
//     kernelShift = 4,
//     padding = 0,
//     stride = 1
//   )

//   // Create 6 parallel convolution layers
//   for (i <- 0 until 6) {
//     val conv1 = new Conv2D3x3(conv1_config)
//     conv1.io.EN := io.EN
//     conv1.io.pre <> io.input

//     // Connect weights and bias (simplified - in real implementation, weights would be properly connected)
//     conv1_outputs(i) <> conv1.io.post
//   }

//   // ============================================================================
//   // ReLU Activation after Conv1
//   // ============================================================================
//   val conv1_relu_outputs = Vec(Stream(SInt(dataWidth bits)), 6)
//   val relu_config = ReLUConfig(
//     dataWidth = dataWidth,
//     activationType = "relu"
//   )

//   for (i <- 0 until 6) {
//     val relu1 = new ReLU(relu_config)
//     relu1.io.EN := io.EN
//     relu1.io.pre <> conv1_outputs(i)
//     conv1_relu_outputs(i) <> relu1.io.post
//   }

//   // ============================================================================
//   // Max Pooling Layer 1: 2×2 max pooling -> Output: 6×12×12
//   // ============================================================================
//   val pool1_outputs = Vec(Stream(SInt(dataWidth bits)), 6)
//   val pool1_config = MaxPool2x2Config(
//     dataWidth = dataWidth,
//     lineLength = 24, // Conv1 output size
//     padding = 0,
//     stride = 2
//   )

//   for (i <- 0 until 6) {
//     val pool1 = new MaxPooling2x2(pool1_config)
//     pool1.io.EN := io.EN
//     pool1.io.pre <> conv1_relu_outputs(i)
//     pool1_outputs(i) <> pool1.io.post
//   }

//   // ============================================================================
//   // Convolution Layer 2: 12 filters of 5×5 -> Output: 12×8×8
//   // ============================================================================
//   val conv2_outputs = Vec(Stream(SInt(dataWidth bits)), 12)
//   val conv2_config = Conv2DConfig(
//     dataWidth = dataWidth,
//     convWidth = dataWidth,
//     lineLength = 12, // Pool1 output size
//     kernel = Seq(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1), // Placeholder
//     kernelShift = 4,
//     padding = 0,
//     stride = 1
//   )

//   // Create 12 parallel convolution layers
//   for (i <- 0 until 12) {
//     val conv2 = new Conv2D3x3(conv2_config)
//     conv2.io.EN := io.EN

//     // Connect to all 6 channels from pool1 (simplified)
//     conv2.io.pre <> pool1_outputs(i % 6)
//     conv2_outputs(i) <> conv2.io.post
//   }

//   // ============================================================================
//   // ReLU Activation after Conv2
//   // ============================================================================
//   val conv2_relu_outputs = Vec(Stream(SInt(dataWidth bits)), 12)

//   for (i <- 0 until 12) {
//     val relu2 = new ReLU(relu_config)
//     relu2.io.EN := io.EN
//     relu2.io.pre <> conv2_outputs(i)
//     conv2_relu_outputs(i) <> relu2.io.post
//   }

//   // ============================================================================
//   // Max Pooling Layer 2: 2×2 max pooling -> Output: 12×4×4
//   // ============================================================================
//   val pool2_outputs = Vec(Stream(SInt(dataWidth bits)), 12)
//   val pool2_config = MaxPool2x2Config(
//     dataWidth = dataWidth,
//     lineLength = 8, // Conv2 output size
//     padding = 0,
//     stride = 2
//   )

//   for (i <- 0 until 12) {
//     val pool2 = new MaxPooling2x2(pool2_config)
//     pool2.io.EN := io.EN
//     pool2.io.pre <> conv2_relu_outputs(i)
//     pool2_outputs(i) <> pool2.io.post
//   }

//   // ============================================================================
//   // Flatten Layer: 12×4×4 -> 192
//   // ============================================================================
//   val flatten_output = Stream(Vec(SInt(dataWidth bits), 192))

//   // Flatten logic: collect all 12 channels of 4x4 = 192 values
//   val flatten_valid = Reg(Bool()) init(False)
//   val flatten_data = Reg(Vec(SInt(dataWidth bits), 192))

//   // Simplified flattening - in real implementation, this would properly collect all values
//   flatten_valid := pool2_outputs(0).valid
//   for (i <- 0 until 192) {
//     flatten_data(i) := pool2_outputs(i % 12).payload
//   }

//   flatten_output.valid := flatten_valid
//   flatten_output.payload := flatten_data

//   // ============================================================================
//   // Fully Connected Layer: 192 -> 10
//   // ============================================================================
//   val fc_config = FullConnectionConfig(
//     inputWidth = dataWidth,
//     outputWidth = dataWidth,
//     weightWidth = weightWidth,
//     biasWidth = biasWidth,
//     inputSize = 192,
//     outputSize = numClasses,
//     useBias = useBias,
//     signed = signed,
//     quantization = quantization
//   )

//   val fc = new FullConnection(fc_config)
//   fc.io.EN := io.EN
//   fc.io.pre <> flatten_output

//   // Connect weights and bias
//   for (i <- 0 until 192 * numClasses) {
//     fc.io.weights.weights(i) := io.fc_weights(i)
//   }
//   for (i <- 0 until numClasses) {
//     fc.io.weights.bias(i) := io.fc_bias(i)
//   }

//   // ============================================================================
//   // Output
//   // ============================================================================
//   io.output <> fc.io.post
// }



object LeNet5Gen {
  def main(args: Array[String]): Unit = {
    println("Generating LeNet-5 CNN network...")

    val config = LeNet5Config(
      dataWidth = 8,
      weightWidth = 8,
      biasWidth = 8,
      inputSize = 28,
      numClasses = 10,
      useBias = true,
      signed = false,
      quantization = false
    )
    // Generate simplified LeNet-5
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new SimpleLeNet5(config)
    )

    println("LeNet-5 network generated successfully!")
    println("=== LeNet-5 Architecture ===")
    println("Input: 28x28 grayscale image")
    println("Conv1: 5x5 convolution -> 24x24")
    println("ReLU1: Activation")
    println("Pool1: 2x2 max pooling -> 12x12")
    println("Conv2: 5x5 convolution -> 8x8")
    println("ReLU2: Activation")
    println("Pool2: 2x2 max pooling -> 4x4")
    println("Flatten: 4x4 -> 16")
    println("FC: 16 -> 10 (output classes)")
  }
}
