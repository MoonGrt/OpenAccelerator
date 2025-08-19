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
    val bias = if (useBias) slave(Stream(SInt(biasWidth bits))) else null
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(SInt(dataWidth bits)))
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
    padding = 2,
    stride = 1)
  val conv1 = new Conv2D(conv1_config)
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
    padding = 2,
    stride = 1)
  val conv2 = new Conv2D(conv2_config)
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
  // Fully Connected Layer: 4×4 -> 10
  // ============================================================================
  val fc_config = FullConnectionConfig(
    inputWidth = dataWidth,
    outputWidth = dataWidth,
    weightWidth = weightWidth,
    biasWidth = biasWidth,
    inputSize = 4*4,
    outputSize = numClasses,
    useBias = useBias,
    quantization = quantization
  )
  val fc = new FullConnection(fc_config)
  fc.io.EN := io.EN
  fc.io.pre <> pool2.io.post
  fc.io.wb.weight <> io.weight
  if (useBias) {
    fc.io.wb.bias <> io.bias
  }

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
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) slave(Stream(SInt(biasWidth bits))) else null
    val input = slave(Stream(SInt(dataWidth bits)))
    val output = master(Stream(SInt(dataWidth bits)))
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
  kernelMap.io.kernelIn <> io.kernel

  // ============================================================================
  // Convolution Layer 1: 6 kernels of 5×5 -> Output: 6×24×24
  // ============================================================================
  val convLayer1_config = Conv2DLayerConfig(
    convNum = 6,
    convConfig = Conv2DConfig(
      dataWidth = 8,
      convWidth = 8,
      lineLength = 28,
      kernelSize = 5,
      kernelShift = 4,
      lineLengthDyn = false,
      padding = 1,
      stride = 1))
  val convLayer1 = new Conv2DLayer(convLayer1_config)
  convLayer1.io.EN := io.EN
  convLayer1.io.pre <> io.input
  convLayer1.io.kernel <> kernelMap.io.kernelOut(0)

  // ============================================================================
  // ReLU Activation after Conv1
  // ============================================================================
  val reluLayer1_config = ReLULayerConfig(
    reluNum = 6,
    reluConfig = ReLUConfig(
      dataWidth = 8,
      activationType = "relu"))
  val reluLayer1 = new ReLULayer(reluLayer1_config)
  reluLayer1.io.EN := io.EN
  reluLayer1.io.pre <> convLayer1.io.post

  // ============================================================================
  // Max Pooling Layer 1: 2×2 max pooling -> Output: 6×12×12
  // ============================================================================
  val poolLayer1_config = MaxPoolLayerConfig(
    maxpoolNum = 6,
    maxpoolConfig = MaxPoolConfig(
      dataWidth = dataWidth,
      lineLength = 24,
      kernelSize = 2,
      padding = 0,
      stride = 2,
      lineLengthDyn = false))
  val poolLayer1 = new MaxPoolLayer(poolLayer1_config)
  poolLayer1.io.EN := io.EN
  poolLayer1.io.pre <> reluLayer1.io.post

  // ============================================================================
  // Convolution Layer 2: 12×6 kernels of 5×5 -> Output: 12×8×8
  // ============================================================================
  val convLayer2_config = Conv2DLayerConfig(
    convNum = 12,
    convConfig = Conv2DConfig(
      channelNum = 6,
      dataWidth = 8,
      convWidth = 8,
      lineLength = 28,
      kernelSize = 5,
      kernelShift = 4,
      lineLengthDyn = false,
      padding = 1,
      stride = 1))
  val convLayer2 = new Conv2DLayerMultiChannel(convLayer2_config)
  convLayer2.io.EN := io.EN
  convLayer2.io.kernel <> kernelMap.io.kernelOut(1)
  convLayer2.io.pre <> poolLayer1.io.post

  // ============================================================================
  // ReLU Activation after Conv2
  // ============================================================================
  val reluLayer2_config = ReLULayerConfig(
    reluNum = 12,
    reluConfig = ReLUConfig(
      dataWidth = 8,
      activationType = "relu"))
  val reluLayer2 = new ReLULayer(reluLayer2_config)
  reluLayer2.io.EN := io.EN
  reluLayer2.io.pre <> convLayer2.io.post

  // ============================================================================
  // Max Pooling Layer 2: 2×2 max pooling -> Output: 12×4×4
  // ============================================================================
  val poolLayer2_config = MaxPoolLayerConfig(
    maxpoolNum = 12,
    maxpoolConfig = MaxPoolConfig(
      dataWidth = dataWidth,
      lineLength = 8,
      kernelSize = 2,
      padding = 0,
      stride = 2,
      lineLengthDyn = false))
  val poolLayer2 = new MaxPoolLayer(poolLayer2_config)
  poolLayer2.io.EN := io.EN
  poolLayer2.io.pre <> reluLayer2.io.post

  // ============================================================================
  // Fully Connected Layer: 12×4×4 -> 10
  // ============================================================================
  val fc_config = FullConnectionLayerConfig(
    fullconnectionNum = 12,
    fullconnectionConfig = FullConnectionConfig(
      inputWidth = dataWidth,
      outputWidth = dataWidth,
      weightWidth = weightWidth,
      biasWidth = biasWidth,
      inputSize = 4*4*12,
      outputSize = numClasses,
      useBias = useBias,
      quantization = quantization))
  val fc = new FullConnectionLayer(fc_config)
  fc.io.EN := io.EN
  fc.io.wb.weight <> io.weight
  if (useBias) {
    fc.io.wb.bias <> io.bias
  }
  fc.io.pre <> poolLayer2.io.post

  // ============================================================================
  // Output
  // ============================================================================
  io.output <> fc.io.post
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
  // 初始化输入数据 Mem
  // ----------------------------
  val inputMem = Mem(SInt(8 bits), wordCount = 28*28)
  val convWMem = Mem(SInt(8 bits), wordCount = 6*5*5 + 6*5*5*12)
  val fcWMem = Mem(SInt(8 bits), wordCount = 4*4*12*10)
  MemTools.initMem(inputMem, "test/LeNet5/weight/0.txt", "bin")
  MemTools.initMem(convWMem, "test/LeNet5/weight/cw.txt", "sdec")
  MemTools.initMem(fcWMem, "test/LeNet5/weight/fcw.txt", "sdec")

  // ----------------------------
  // 驱动逻辑
  // ----------------------------
  // Enable
  cnn.io.EN := ~cnn.io.kernel.valid && ~cnn.io.weight.valid
  cnn.io.output.ready := True

  // Kernel 驱动
  val convWCnt = Reg(UInt(log2Up(6*5*5+6*5*5*12) bits)) init(0)
  cnn.io.kernel.valid := convWCnt < convWMem.wordCount
  cnn.io.kernel.payload := convWMem.readSync(convWCnt)
  when(cnn.io.kernel.fire) {
    convWCnt := convWCnt + 1
  }
  // FC Weight 驱动
  val fcWCnt   = Reg(UInt(log2Up(4*4*12*10) bits)) init(0)
  cnn.io.weight.valid := fcWCnt < fcWMem.wordCount
  cnn.io.weight.payload := fcWMem.readSync(fcWCnt)
  when(cnn.io.weight.fire) {
    fcWCnt := fcWCnt + 1
  }
  // Input 驱动
  val inputCnt = Reg(UInt(log2Up(28*28) bits)) init(0)
  cnn.io.input.valid := (inputCnt < inputMem.wordCount) && cnn.io.EN
  cnn.io.input.payload := inputMem.readSync(inputCnt)
  when(cnn.io.input.fire) {
    inputCnt := inputCnt + 1
  }

  // 输出结果打印 (仿真时用 $display)
  when(cnn.io.output.valid){
    report(s"LeNet5 Output = ${cnn.io.output.payload}")
  }
}

object LeNet5TBGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new LeNet5TestBench
    )
  }
}
