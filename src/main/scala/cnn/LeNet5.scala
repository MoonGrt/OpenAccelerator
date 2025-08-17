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
    val kernel1 = slave(Stream(SInt(dataWidth bits)))
    val kernel2 = slave(Stream(SInt(dataWidth bits)))
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = slave(Stream(SInt(biasWidth bits)))
  }

  // ============================================================================
  // Convolution Layer 1: 5×5 -> Output: 24×24
  // ============================================================================
  val conv1_config = Conv2DConfig(
    dataWidth = 8,
    convWidth = 8,
    lineLength = 28,
    kernelSize = 5,
    kernelShift = 4,
    lineLengthDyn = false,
    kernelDyn = true,
    padding = 2,
    stride = 1)
  val conv1 = new Conv2DStream(conv1_config)
  conv1.io.EN := io.EN
  conv1.io.pre <> io.input
  conv1.io.kernel <> io.kernel1

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val relu1 = new ReLU(ReLUConfig(dataWidth, "relu"))
  relu1.io.EN := io.EN
  relu1.io.pre <> conv1.io.post

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 12×12
  // ============================================================================
  val pool1_config = MaxPoolConfig(
    dataWidth = dataWidth,
    lineLength = 24,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    lineLengthDyn = false
  )
  val pool1 = new MaxPool(pool1_config)
  pool1.io.EN := io.EN
  pool1.io.pre <> relu1.io.post

  // ============================================================================
  // Convolution Layer 2: 5×5 -> Output: 8×8
  // ============================================================================
  val conv2_config = Conv2DConfig(
    dataWidth = dataWidth,
    convWidth = dataWidth,
    lineLength = 12,
    kernelSize = 5,
    kernelShift = 4,
    lineLengthDyn = false,
    kernelDyn = true,
    padding = 2,
    stride = 1)
  val conv2 = new Conv2DStream(conv2_config)
  conv2.io.EN := io.EN
  conv2.io.pre <> pool1.io.post
  conv2.io.kernel <> io.kernel2

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val relu2 = new ReLU(ReLUConfig(dataWidth, "relu"))
  relu2.io.EN := io.EN
  relu2.io.pre <> conv2.io.post

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 4×4
  // ============================================================================
  val pool2_config = MaxPoolConfig(
    dataWidth = dataWidth,
    lineLength = 8,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    lineLengthDyn = false
  )
  val pool2 = new MaxPool(pool2_config)
  pool2.io.EN := io.EN
  pool2.io.pre <> relu2.io.post

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
  fc.io.wb.weight <> io.weight
  fc.io.wb.bias <> io.bias

  // ============================================================================
  // Output
  // ============================================================================
  io.output <> fc.io.post
}

/**
 * LeNet-5 CNN Network
 */
class LeNet5(config: LeNet5Config) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(SInt(dataWidth bits)))
    val kernel1 = slave(Stream(SInt(dataWidth bits)))
    val kernel2 = slave(Stream(SInt(dataWidth bits)))
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = slave(Stream(SInt(biasWidth bits)))
  }

  // ============================================================================
  // Convolution Layer 1: 6 kernels of 5×5 -> Output: 6×24×24
  // ============================================================================
  val convLayer1_config = Conv2DLayerConfig(
    kernelNum = 6,
    convConfig = Conv2DConfig(
      dataWidth = 8,
      convWidth = 8,
      lineLength = 28,
      kernelSize = 5,
      kernelShift = 4,
      lineLengthDyn = false,
      kernelDyn = true,
      padding = 1,
      stride = 1))
  val convLayer1 = new Conv2DLayerStreamZip(convLayer1_config)
  convLayer1.io.EN := io.EN
  convLayer1.io.pre <> io.input
  convLayer1.io.kernel <> io.kernel1

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val relu1_outputs = Vec(Stream(SInt(dataWidth bits)), 6)
  val relu1_config = ReLUConfig(
    dataWidth = dataWidth,
    activationType = "relu"
  )
  for (i <- 0 until 6) {
    val relu1 = new ReLU(relu1_config)
    relu1.io.EN := io.EN
    relu1.io.pre <> convLayer1.io.post(i)
    relu1_outputs(i) <> relu1.io.post
  }

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 6×12×12
  // ============================================================================
  val pool1_outputs = Vec(Stream(SInt(dataWidth bits)), 6)
  val pool1_config = MaxPoolConfig(
    dataWidth = dataWidth,
    lineLength = 24,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    lineLengthDyn = false
  )
  for (i <- 0 until 6) {
    val pool1 = new MaxPool(pool1_config)
    pool1.io.EN := io.EN
    pool1.io.pre <> relu1_outputs(i)
    pool1_outputs(i) <> pool1.io.post
  }

  // ============================================================================
  // Convolution Layer 2: 12 kernels of 5×5 -> Output: 12×8×8
  // ============================================================================
  val convLayer2_config = Conv2DLayerConfig(
    kernelNum = 12,
    convConfig = Conv2DConfig(
      dataWidth = 8,
      convWidth = 8,
      lineLength = 28,
      kernelSize = 5,
      kernelShift = 4,
      lineLengthDyn = false,
      kernelDyn = true,
      padding = 1,
      stride = 1))
  val convLayer2 = new Conv2DLayerStreamZipMultiIn(convLayer2_config)
  convLayer2.io.EN := io.EN
  convLayer2.io.kernel <> io.kernel2
  for (i <- 0 until 6) {
    convLayer2.io.pre(i) <> pool1_outputs(i)
  }
  for (i <- 0 until 6) {
    convLayer2.io.pre(i+6).payload := pool1_outputs(i).payload
    convLayer2.io.pre(i+6).valid := pool1_outputs(i).valid
  }

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val relu2_outputs = Vec(Stream(SInt(dataWidth bits)), 12)
  val relu2_config = ReLUConfig(
    dataWidth = dataWidth,
    activationType = "relu"
  )
  for (i <- 0 until 12) {
    val relu2 = new ReLU(relu2_config)
    relu2.io.EN := io.EN
    relu2.io.pre <> convLayer2.io.post(i)
    relu2_outputs(i) <> relu2.io.post
  }

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 12×4×4
  // ============================================================================
  val pool2_outputs = Vec(Stream(SInt(dataWidth bits)), 12)
  val pool2_config = MaxPoolConfig(
    dataWidth = dataWidth,
    lineLength = 8,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    lineLengthDyn = false
  )
  for (i <- 0 until 12) {
    val pool2 = new MaxPool(pool2_config)
    pool2.io.EN := io.EN
    pool2.io.pre <> relu2_outputs(i)
    pool2_outputs(i) <> pool2.io.post
  }

  // ============================================================================
  // Fully Connected Layer: 12×4×4 -> 10
  // ============================================================================
  val fc_config = FullConnectionConfig(
    inputWidth = dataWidth,
    outputWidth = dataWidth,
    weightWidth = weightWidth,
    biasWidth = biasWidth,
    inputSize = 4*4*12,
    outputSize = numClasses,
    useBias = useBias,
    signed = signed,
    quantization = quantization
  )
  val fc = new FullConnectionStreamMultiIn(12, fc_config)
  fc.io.EN := io.EN
  fc.io.wb.weight <> io.weight
  fc.io.wb.bias <> io.bias
  for (i <- 0 until 12) {
    fc.io.pre(i) <> pool2_outputs(i)
  }

  // ============================================================================
  // Output
  // ============================================================================
  io.output <> fc.io.post
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
object LeNet5Gen {
  def main(args: Array[String]): Unit = {
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
    // // Generate simplified LeNet-5
    // SpinalConfig(targetDirectory = "rtl").generateVerilog(
    //   new SimpleLeNet5(config)
    // )
    // Generate complete LeNet-5
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new LeNet5(config)
    )
    // println("LeNet-5 network generated successfully!")
    // println("=== LeNet-5 Architecture ===")
    // println("Input: 28x28 grayscale image")
    // println("Conv1: 5x5 convolution -> 24x24")
    // println("ReLU1: Activation")
    // println("Pool1: 2x2 max pooling -> 12x12")
    // println("Conv2: 5x5 convolution -> 8x8")
    // println("ReLU2: Activation")
    // println("Pool2: 2x2 max pooling -> 4x4")
    // println("Flatten: 4x4 -> 16")
    // println("FC: 16 -> 10 (output classes)")
  }
}
