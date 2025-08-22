package cnn

import misc._
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
  quantization : Boolean = false   // whether to use quantization
)

/**
 * Simplified LeNet-5 for demonstration (single channel processing)
 */
class SimpleLeNet5(config: LeNet5Config) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val kernel1 = slave(Stream(SInt(dataWidth bits)))
    val kernel2 = slave(Stream(SInt(dataWidth bits)))
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = useBias generate in SInt(biasWidth bits)
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(SInt(dataWidth bits)))
  }

  // ============================================================================
  // Convolution Layer 1: 5×5 -> Output: 24×24
  // ============================================================================
  val conv1_config = ConvConfig(
    dataWidth = 8,
    convWidth = 8,
    rowNum = 28,
    colNum = 28,
    kernelWidth = 8,
    kernelSize = 5,
    kernelShift = 4,
    rowNumDyn = false,
    padding = 0,
    stride = 1)
  val conv1 = new Conv2D(conv1_config)
  conv1.io.EN := io.EN
  conv1.io.pre <> io.input
  conv1.io.kernel <> io.kernel1

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val relu1 = new ReLU(ReLUConfig(dataWidth, dataWidth, 0, "relu"))
  relu1.io.pre <> conv1.io.post

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 12×12
  // ============================================================================
  val pool1_config = MaxPoolConfig(
    dataWidth = dataWidth,
    rowNum = 24,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    rowNumDyn = false
  )
  val pool1 = new MaxPool(pool1_config)
  pool1.io.EN := io.EN
  pool1.io.pre <> relu1.io.post

  // ============================================================================
  // Convolution Layer 2: 5×5 -> Output: 8×8
  // ============================================================================
  val conv2_config = ConvConfig(
    dataWidth = dataWidth,
    convWidth = dataWidth,
    rowNum = 12,
    colNum = 12,
    kernelWidth = 8,
    kernelSize = 5,
    kernelShift = 4,
    rowNumDyn = false,
    padding = 0,
    stride = 1)
  val conv2 = new Conv2D(conv2_config)
  conv2.io.EN := io.EN
  conv2.io.pre <> pool1.io.post
  conv2.io.kernel <> io.kernel2

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val relu2 = new ReLU(ReLUConfig(dataWidth, dataWidth, 0, "relu"))
  relu2.io.pre <> conv2.io.post

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 4×4
  // ============================================================================
  val pool2_config = MaxPoolConfig(
    dataWidth = dataWidth,
    rowNum = 8,
    kernelSize = 2,
    padding = 0,
    stride = 2,
    rowNumDyn = false
  )
  val pool2 = new MaxPool(pool2_config)
  pool2.io.EN := io.EN
  pool2.io.pre <> relu2.io.post

  // ============================================================================
  // Fully Connected Layer: 4×4 -> 10
  // ============================================================================
  val fc_config = FullConnectConfig(
    inputWidth = dataWidth,
    outputWidth = dataWidth,
    weightWidth = weightWidth,
    biasWidth = biasWidth,
    inputSize = 4*4,
    useBias = useBias,
    quantization = quantization
  )
  val fcLayer = new FullConnect(fc_config)
  fcLayer.io.EN := io.EN
  fcLayer.io.pre <> pool2.io.post
  fcLayer.io.weight <> io.weight
  if (useBias) { fcLayer.io.bias <> io.bias }

  // ============================================================================
  // Output
  // ============================================================================
  io.output <> fcLayer.io.post
}

/**
 * LeNet-5 CNN Network
 */
class LeNet5(config: LeNet5Config) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) slave(Stream(SInt(biasWidth bits))) else null
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(UInt(log2Up(numClasses) bits)))
  }

  // ============================================================================
  // Convolution Layer Steam Map
  // ============================================================================
  val kernelMapCfg = StreamMapConfig(
    dataWidth = 8,
    streamSize = Seq(
      6 * 5 * 5,
      6 * 5 * 5 * 12
    )
  )
  val kernelMap = new StreamMap(kernelMapCfg)
  kernelMap.io.streamIn <> io.kernel

  // ============================================================================
  // Convolution Layer 1: 6 kernels of 5×5 -> Output: 6×24×24
  // ============================================================================
  val convLayer1_config = Conv2DLayerConfig(
    convNum = 6,
    convConfig = ConvConfig(
      dataWidth = 8,
      convWidth = 32,
      rowNum = 28,
      colNum = 28,
      kernelSize = 5,
      insigned = false, // imagetype is unsigned
      rowNumDyn = false,
      padding = 0,
      stride = 1))
  val convLayer1 = new Conv2DLayer(convLayer1_config)
  convLayer1.io.EN := io.EN
  convLayer1.io.pre <> io.input
  convLayer1.io.kernel <> kernelMap.io.streamOut(0)

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val reluLayer1_config = ReLULayerConfig(
    reluNum = 6,
    reluConfig = ReLUConfig(
      indataWidth = 32,
      outdataWidth = 8,
      shift = 10,
      activationType = "relu"))
  val reluLayer1 = new ReLULayer(reluLayer1_config)
  reluLayer1.io.pre <> convLayer1.io.post

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 6×12×12
  // ============================================================================
  val poolLayer1_config = MaxPoolLayerConfig(
    maxpoolNum = 6,
    maxpoolConfig = MaxPoolConfig(
      dataWidth = 8,
      rowNum = 24,
      colNum = 24,
      kernelSize = 2,
      padding = 0,
      stride = 2,
      rowNumDyn = false))
  val poolLayer1 = new MaxPoolLayer(poolLayer1_config)
  poolLayer1.io.EN := io.EN
  poolLayer1.io.pre <> reluLayer1.io.post

  // ============================================================================
  // Convolution Layer 2: 12×6 kernels of 5×5 -> Output: 12×8×8
  // ============================================================================
  val convLayer2_config = Conv2DLayerConfig(
    convNum = 12,
    convConfig = ConvConfig(
      channelNum = 6,
      dataWidth = 8,
      convWidth = 32,
      rowNum = 12,
      colNum = 12,
      kernelSize = 5,
      rowNumDyn = false,
      padding = 0,
      stride = 1))
  val convLayer2 = new Conv3DLayer(convLayer2_config)
  convLayer2.io.EN := io.EN
  convLayer2.io.kernel <> kernelMap.io.streamOut(1)
  convLayer2.io.pre <> poolLayer1.io.post

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val reluLayer2_config = ReLULayerConfig(
    reluNum = 12,
    reluConfig = ReLUConfig(
      indataWidth = 32,
      outdataWidth = 8,
      shift = 10,
      activationType = "relu"))
  val reluLayer2 = new ReLULayer(reluLayer2_config)
  reluLayer2.io.pre <> convLayer2.io.post

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 12×4×4
  // ============================================================================
  val poolLayer2_config = MaxPoolLayerConfig(
    maxpoolNum = 12,
    maxpoolConfig = MaxPoolConfig(
      dataWidth = 8,
      rowNum = 8,
      colNum = 8,
      kernelSize = 2,
      padding = 0,
      stride = 2,
      rowNumDyn = false))
  val poolLayer2 = new MaxPoolLayer(poolLayer2_config)
  poolLayer2.io.EN := io.EN
  poolLayer2.io.pre <> reluLayer2.io.post

  // ============================================================================
  // Fully Connected Layer: 12×4×4 -> 10
  // ============================================================================
  val fc_config = FullConnectLayerConfig(
    fullconnectNum = numClasses,
    fullconnectConfig = FullConnectConfig(
      kernelNum = 12,
      inputWidth = 8,
      outputWidth = 8,
      weightWidth = weightWidth,
      biasWidth = biasWidth,
      inputSize = 4*4*12,
      useBias = useBias,
      quantization = quantization))
  val fcLayer = new FullConnect2DLayer(fc_config)
  fcLayer.io.EN := io.EN
  fcLayer.io.weight <> io.weight
  if (useBias) { fcLayer.io.bias <> io.bias }
  fcLayer.io.pre <> poolLayer2.io.post
  fcLayer.io.post.ready := io.output.ready

  // ============================================================================
  // Output
  // ============================================================================
  // Max Result
  val maxVal = Reg(SInt(8 bits)) init(fcLayer.io.post.payload(0))
  val maxIdx = Reg(UInt(log2Up(numClasses) bits)) init(0)
  maxVal := fcLayer.io.post.payload(0)
  maxIdx := 0
  for(i <- 1 until numClasses) {
    when(fcLayer.io.post.payload(i) > maxVal) {
      maxVal := fcLayer.io.post.payload(i)
      maxIdx := U(i, log2Up(numClasses) bits)
    }
  }
  // Output Stream
  io.output.valid := RegNext(fcLayer.io.post.valid)
  io.output.payload := maxIdx
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object SimpleLeNet5Gen {
//   def main(args: Array[String]): Unit = {
//     val config = LeNet5Config(
//       dataWidth = 8,
//       weightWidth = 8,
//       biasWidth = 8,
//       inputSize = 28,
//       numClasses = 10,
//       useBias = true,
//       quantization = false
//     )
//     // Generate simplified LeNet-5
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new SimpleLeNet5(config)
//     )
//   }
// }

// object LeNet5Gen {
//   def main(args: Array[String]): Unit = {
//     val config = LeNet5Config(
//       dataWidth = 8,
//       weightWidth = 8,
//       biasWidth = 8,
//       inputSize = 28,
//       numClasses = 10,
//       useBias = true,
//       quantization = false
//     )
//     // Generate Complete LeNet-5
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new LeNet5(config)
//     )
//   }
// }

/* ---------------------------------------------------------------------------- */
/* --------------------------------- TestBench -------------------------------- */
/* ---------------------------------------------------------------------------- */
/**
 * Test bench for LeNet-5
 */
class LeNet5TestBench extends Component {
  val cnn = new LeNet5(LeNet5Config(
    dataWidth = 8,
    weightWidth = 8,
    biasWidth = 8,
    numClasses = 10,
    useBias = false,
    quantization = false
  ))

  // ----------------------------
  // Initialize input data Mem
  // ----------------------------
  val inputMem = Mem(SInt(8 bits), wordCount = 28*28)
  val convWMem = Mem(SInt(8 bits), wordCount = 6*5*5 + 6*5*5*12)
  val fcWMem = Mem(SInt(8 bits), wordCount = 4*4*12*10)
  // Basic calculation test
  // MemTools.initMem(inputMem, "test/LeNet5/weight/0.txt", "bin")
  // MemTools.initMem(convWMem, "test/LeNet5/weight/cw.txt", "sdec")
  // MemTools.initMem(fcWMem, "test/LeNet5/weight/fcw.txt", "sdec")
  // LeNet-5 calculation test
  MemTools.initMem(inputMem, "test/LeNet5/test/bar.txt", "bin")
  MemTools.initMem(convWMem, "test/LeNet5/test/sobel.txt", "sdec")
  MemTools.initMem(fcWMem, "test/LeNet5/test/fcw.txt", "sdec")

  // ----------------------------
  // Drive logic
  // ----------------------------
  // Enable
  cnn.io.EN := ~cnn.io.kernel.ready && ~cnn.io.weight.ready
  cnn.io.output.ready := True

  // Kernel driver
  val kernelWCnt = Reg(UInt(log2Up(6*5*5+6*5*5*12) bits)) init(0)
  cnn.io.kernel.valid := kernelWCnt < convWMem.wordCount
  cnn.io.kernel.payload := convWMem.readAsync(kernelWCnt)
  when(cnn.io.kernel.fire) { kernelWCnt := kernelWCnt + 1 }
  // FC Weight driver
  val fcWCnt = Reg(UInt(log2Up(4*4*12*10) bits)) init(0)
  cnn.io.weight.valid := fcWCnt < fcWMem.wordCount
  cnn.io.weight.payload := fcWMem.readAsync(fcWCnt)
  when(cnn.io.weight.fire) { fcWCnt := fcWCnt + 1 }
  // Input driver
  val inputCnt = Reg(UInt(log2Up(28*28) bits)) init(0)
  cnn.io.input.valid := (inputCnt < inputMem.wordCount) && cnn.io.EN
  cnn.io.input.payload := inputMem.readAsync(inputCnt)
  when(cnn.io.input.fire) { inputCnt := inputCnt + 1 }

  // Print output results
  when(cnn.io.output.valid){ report(s"LeNet5 Output = ${cnn.io.output.payload}") }
}

object LeNet5TBGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(targetDirectory = "rtl").generateVerilog(new LeNet5TestBench)
  }
}
