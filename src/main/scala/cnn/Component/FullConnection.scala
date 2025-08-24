package cnn

import misc._
import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* ----------------------------- Full Connection ----------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * Full Connection
 *
 * This module implements a fully connected layer with configurable input/output dimensions.
 * It can be used for:
 * - Matrix multiplication: Y = W * X + b
 * - Linear transformation
 * - Classification layers
 */
case class FullConnectConfig(
  kernelNum    : Int = 1,         // number of input kernels
  inputWidth   : Int = 8,         // bits per input element
  outputWidth  : Int = 8,         // bits per output element
  weightWidth  : Int = 8,         // bits per weight element
  biasWidth    : Int = 8,         // bits per bias element
  inputSize    : Int = 8,         // number of input neurons
  useBias      : Boolean = true,  // whether to use bias
  quantization : Boolean = false  // whether to use quantization
)

/**
 * Full Connection module (streaming weights/bias)
 */
class FullConnect(config: FullConnectConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) in SInt(biasWidth bits) else null
    val pre = slave(Stream(SInt(inputWidth bits)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // Ready signal
  io.pre.ready := io.EN

  // Storage for weights
  val weightMem = Mem(SInt(weightWidth bits), inputSize)
  val weightCnt = Reg(UInt(log2Up(inputSize + 1) bits)) init(0)
  // Receive weights
  io.weight.ready := (weightCnt < inputSize)
  when(io.weight.fire) {
    weightMem.write(weightCnt.resized, io.weight.payload)
    weightCnt := weightCnt + 1
  }

  // Temporary accumulator register for each output neuron
  val accVec = Reg(SInt(outputWidth bits)) init(0)
  // Record how many data groups each stream has received so far.
  val inputCnt = Counter(inputSize)

  // When EN is active and all inputs are valid, accumulate to the register.
  io.post.valid := False
  io.post.payload := 0
  when(io.EN && io.pre.valid) {
    accVec := (accVec + weightMem.readSync(inputCnt.resized) * io.pre.payload).resized
    inputCnt.increment()
  }
  when(RegNext(inputCnt.willOverflow)) {
    val withBias = if (useBias) accVec + io.bias else accVec.resize(outputWidth)
    val quantized =
      if (quantization) (withBias >> weightWidth).resize(outputWidth)
      else withBias.resize(outputWidth)
    io.post.valid := True
    io.post.payload := quantized
    accVec := 0
  }
}

/**
 * Full Connection module (streaming weights/bias) with multiple kernel inputs
 */
class FullConnect2D(config: FullConnectConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) in SInt(biasWidth bits) else null
    val pre = slave(Stream(Vec(SInt(inputWidth bits), kernelNum)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // --- StreamMap ---
  val weightStreamMap = new StreamMap(StreamMapConfig(weightWidth, Seq.fill(kernelNum)(inputSize / kernelNum)))
  weightStreamMap.io.streamIn <> io.weight
  // --- Full Connection ---
  val fcArray = (0 until kernelNum).map { ch =>
    val fc = new FullConnect(config.copy(inputSize = inputSize / kernelNum, useBias = false))
    fc.io.EN <> io.EN
    fc.io.pre.valid := io.pre.valid
    fc.io.pre.payload := io.pre.payload(ch)
    fc.io.post.ready := io.post.ready
    fc.io.weight <> weightStreamMap.io.streamOut(ch)
    fc
  }
  io.pre.ready := fcArray.map(_.io.pre.ready).reduce(_ && _)
  // ---Output ---
  val adder = Stream(SInt(outputWidth bits))
  adder.valid := fcArray.map(_.io.post.valid).reduce(_ || _)
  if (useBias) adder.payload := fcArray.map(_.io.post.payload).reduce(_ + _) + io.bias
  else adder.payload := fcArray.map(_.io.post.payload).reduce(_ + _)
  io.post <> adder
}


/* --------------------------------------------------------------------------- */
/* -------------------------- Full Connection Layer -------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * Full Connection module (streaming weights/bias) with multiple kernel inputs
 */
case class FullConnectLayerConfig(
  fullconnectNum: Int,
  fullconnectConfig: FullConnectConfig
)

class FullConnectLayer(layerCfg: FullConnectLayerConfig) extends Component {
  import layerCfg._
  import fullconnectConfig._
  val io = new Bundle {
    val EN = in Bool()
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) slave(Stream(SInt(biasWidth bits))) else null
    val pre = slave(Stream(SInt(inputWidth bits)))
    val post = master(Stream(Vec(SInt(outputWidth bits), fullconnectNum)))
  }

  // --- StreamMap ---
  val weightStreamMap = new StreamMap(StreamMapConfig(weightWidth, Seq.fill(fullconnectNum)(inputSize)))
  weightStreamMap.io.streamIn <> io.weight
  // --- Bias ---
  val biasRegs = Vec(Reg(SInt(biasWidth bits)) init(0), fullconnectNum)
  val biasCnt = Reg(UInt(log2Up(fullconnectNum) bits)) init(0)
  io.bias.ready := biasCnt < fullconnectNum
  when(io.bias.fire) {
    biasRegs(biasCnt) := io.bias.payload
    biasCnt := biasCnt + 1
  }
  // --- Convolution ---
  val fcArray = (0 until fullconnectNum).map { ch =>
    val fc = new FullConnect(fullconnectConfig)
    fc.io.EN <> io.EN
    fc.io.pre.valid := io.pre.valid
    fc.io.pre.payload := io.pre.payload
    fc.io.post.ready := io.post.ready
    fc.io.weight <> weightStreamMap.io.streamOut(ch)
    if (useBias) fc.io.bias <> biasRegs(ch)
    fc
  }
  // ---Output ---
  io.pre.ready := fcArray.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := fcArray.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until fullconnectNum) {
    io.post.payload(i) := fcArray(i).io.post.payload
  }
}

class FullConnect2DLayer(layerCfg: FullConnectLayerConfig) extends Component {
  import layerCfg._
  import fullconnectConfig._
  val io = new Bundle {
    val EN = in Bool()
    val weight = slave(Stream(SInt(weightWidth bits)))
    val bias = if (useBias) slave(Stream(SInt(biasWidth bits))) else null
    val pre = slave(Stream(Vec(SInt(inputWidth bits), kernelNum)))
    val post = master(Stream(Vec(SInt(outputWidth bits), fullconnectNum)))
  }

  // --- StreamMap ---
  val weightStreamMap = new StreamMap(StreamMapConfig(weightWidth, Seq.fill(fullconnectNum)(inputSize)))
  weightStreamMap.io.streamIn <> io.weight
  // --- Bias ---
  val biasRegs = if (useBias) Vec(Reg(SInt(biasWidth bits)) init(0), fullconnectNum) else null
  val biasCnt = if (useBias) Reg(UInt(log2Up(fullconnectNum) bits)) init(0) else null
  if (useBias) {
    io.bias.ready := biasCnt < fullconnectNum
    when(io.bias.fire) {
      biasRegs(biasCnt) := io.bias.payload
      biasCnt := biasCnt + 1
    }
  }
  // --- Convolution ---
  val fcArray = (0 until fullconnectNum).map { ch =>
    val fc = new FullConnect2D(fullconnectConfig)
    fc.io.EN <> io.EN
    fc.io.pre.valid := io.pre.valid
    fc.io.pre.payload := io.pre.payload
    fc.io.post.ready := io.post.ready
    fc.io.weight <> weightStreamMap.io.streamOut(ch)
    if (useBias) fc.io.bias <> biasRegs(ch)
    fc
  }
  // ---Output ---
  io.pre.ready := fcArray.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := fcArray.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until fullconnectNum) {
    io.post.payload(i) := fcArray(i).io.post.payload
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object FullConnectGen {
//   def main(args: Array[String]): Unit = {
//     // Basic full connection layer
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new FullConnect(FullConnectConfig(
//         inputWidth = 8,
//         outputWidth = 16,
//         weightWidth = 8,
//         biasWidth = 8,
//         inputSize = 192,
//         useBias = true,
//         quantization = false))
//     ).printPruned()

//     // // Basic full connection layer streaming weights/bias with multi-input
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new FullConnect2D(FullConnectConfig(
//     //     kernelNum = 6,
//     //     inputWidth = 8,
//     //     outputWidth = 16,
//     //     weightWidth = 8,
//     //     biasWidth = 8,
//     //     inputSize = 192,
//     //     useBias = true,
//     //     quantization = false))
//     // ).printPruned()
//   }
// }

// object FullConnectLayerGen {
//   def main(args: Array[String]): Unit = {
//     // Basic full connection layer streaming weights/bias
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new FullConnectLayer(FullConnectLayerConfig(
//         fullconnectNum = 6,
//         fullconnectConfig = FullConnectConfig(
//           inputWidth = 8,
//           outputWidth = 16,
//           weightWidth = 8,
//           biasWidth = 8,
//           inputSize = 192,
//           useBias = true,
//           quantization = false)))
//     ).printPruned()

//     // // Basic full connection layer streaming weights/bias with multi-input
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new FullConnect2DLayer(FullConnectLayerConfig(
//     //     fullconnectNum = 6,
//     //     fullconnectConfig = FullConnectConfig(
//     //       kernelNum = 6,
//     //       inputWidth = 8,
//     //       outputWidth = 16,
//     //       weightWidth = 8,
//     //       biasWidth = 8,
//     //       inputSize = 192,
//     //       useBias = true,
//     //       quantization = false)))
//     // ).printPruned()
//   }
// }
