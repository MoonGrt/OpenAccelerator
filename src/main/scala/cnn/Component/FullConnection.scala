package cnn

import spinal.core._
import spinal.lib._

/**
 * Full Connection Layer
 *
 * This module implements a fully connected layer with configurable input/output dimensions.
 * It can be used for:
 * - Matrix multiplication: Y = W * X + b
 * - Linear transformation
 * - Classification layers
 */
case class FullConnectionConfig(
  inputWidth   : Int,             // bits per input element
  outputWidth  : Int,             // bits per output element
  weightWidth  : Int = 8,         // bits per weight element
  biasWidth    : Int = 8,         // bits per bias element
  inputSize    : Int,             // number of input neurons
  outputSize   : Int,             // number of output neurons
  useBias      : Boolean = true,  // whether to use bias
  signed       : Boolean = true,  // whether data is signed
  quantization : Boolean = false  // whether to use quantization
)

/**
 * Weight memory interface
 */
case class WeightBiasInterface(
  weightWidth: Int,
  biasWidth: Int,
  useBias: Boolean = true
) extends Bundle with IMasterSlave {
  val weight = Stream(SInt(weightWidth bits))
  val bias = useBias generate Stream(SInt(biasWidth bits))

  override def asMaster(): Unit = {
    out(weight.valid, weight.payload)
    in(weight.ready)
    if (useBias) {
      out(bias.valid, bias.payload)
      in(bias.ready)
    }
  }
  override def asSlave(): Unit = {
    in(weight.valid, weight.payload)
    out(weight.ready)
    if (useBias) {
      in(bias.valid, bias.payload)
      out(bias.ready)
    }
  }
  override def clone = WeightBiasInterface(weightWidth, biasWidth, useBias)
}

/**
 * Full Connection module (streaming weights/bias) with multiple kernel inputs
 */
class FullConnection(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val wb   = slave(WeightBiasInterface(weightWidth, biasWidth, useBias))
    val pre  = slave(Stream(SInt(inputWidth bits)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // Ready signal
  io.pre.ready := True

  // Storage for weights and bias
  val weightMem = Mem(SInt(weightWidth bits), inputSize * outputSize)
  val weightCnt = Reg(UInt(log2Up(inputSize * outputSize) bits)) init(0)
  val biasMem = if (useBias) Mem(SInt(biasWidth bits), outputSize) else null
  val biasCnt = if (useBias) Reg(UInt(log2Up(outputSize) bits)) init(0) else null

  // Receive weights and biases
  io.wb.weight.ready := (weightCnt < inputSize * outputSize)
  when(io.wb.weight.fire) {
    weightMem.write(weightCnt, io.wb.weight.payload)
    weightCnt := weightCnt + 1
  }
  if (useBias) {
    io.wb.bias.ready := (biasCnt < outputSize)
    when(io.wb.bias.fire) {
      biasMem.write(biasCnt, io.wb.bias.payload)
      biasCnt := biasCnt + 1
    }
  }

  // Stream computing
  val computing = RegInit(False)
  val outputCnt = Counter(outputSize)
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
    when(inputCnt.willOverflow) {
      // Accumulate a complete set of inputs and output the results.
      val dotSum = accVec
      val withBias =
        if (useBias) dotSum + biasMem.readSync(outputCnt.value).resize(outputWidth)
        else dotSum.resize(outputWidth)
      val quantized =
        if (quantization) (withBias >> weightWidth).asUInt.resize(outputWidth)
        else withBias.asUInt.resize(outputWidth)
      // Clear the accumulator register to prepare for the next round.
      accVec := 0
      inputCnt.clear()
      outputCnt.increment()
      when(outputCnt.willOverflow) {
        computing := False
      } .otherwise {
        computing := True
      }
      io.post.valid := True
      io.post.payload := quantized.asSInt
    }
  }
}


/* --------------------------------------------------------------------------- */
/* -------------------------- Full Connection Layer -------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * Full Connection module (streaming weights/bias) with multiple kernel inputs
 */
case class FullConnectionLayerConfig(
  fullconnectionNum: Int,
  fullconnectionConfig: FullConnectionConfig
)

class FullConnectionLayer(layerCfg: FullConnectionLayerConfig) extends Component {
  import layerCfg._
  import fullconnectionConfig._
  val io = new Bundle {
    val EN   = in Bool()
    val wb   = slave(WeightBiasInterface(weightWidth, biasWidth, useBias))
    val pre  = slave(Stream(Vec(SInt(inputWidth bits), fullconnectionNum)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // Ready signal
  io.pre.ready := True

  // Storage for weights and bias
  val weightMem = Mem(SInt(weightWidth bits), inputSize * outputSize)
  val weightCnt = Reg(UInt(log2Up(inputSize * outputSize) bits)) init(0)
  val biasMem = if (useBias) Mem(SInt(biasWidth bits), outputSize) else null
  val biasCnt = if (useBias) Reg(UInt(log2Up(outputSize) bits)) init(0) else null

  // Receive weights and biases
  io.wb.weight.ready := weightCnt < inputSize * outputSize
  when(io.wb.weight.fire) {
    weightMem.write(weightCnt, io.wb.weight.payload)
    weightCnt := weightCnt + 1
  }
  if (useBias) {
    io.wb.bias.ready := biasCnt < outputSize
    when(io.wb.bias.fire) {
      biasMem.write(biasCnt, io.wb.bias.payload)
      biasCnt := biasCnt + 1
    }
  }

  // Stream computing
  val computing = RegInit(False)
  val outputCnt = Counter(outputSize)
  // Temporary accumulator register for each output neuron
  val accVec = Vec(Reg(SInt(outputWidth bits)) init(0), fullconnectionNum)
  // Record how many data groups each stream has received so far.
  val inputCnt = Counter(inputSize / fullconnectionNum)
  // When EN is active and all inputs are valid, accumulate to the register.
  io.post.valid := False
  io.post.payload := 0
  when(io.EN && io.pre.valid) {
    for (k <- 0 until fullconnectionNum) {
      val baseIdx = inputCnt.value * fullconnectionNum
      accVec(k) := (accVec(k) + (0 until fullconnectionNum).map(j =>
        weightMem.readSync((outputCnt.value * inputSize + baseIdx + j).resized) * io.pre.payload(j)
      ).reduce(_ + _)).resized
    }
    inputCnt.increment()
    when(inputCnt.willOverflow) {
      // Accumulate a complete set of inputs and output the results.
      val dotSum = accVec.reduce(_ + _)
      val withBias =
        if (useBias) dotSum + biasMem.readSync(outputCnt.value).resize(outputWidth)
        else dotSum.resize(outputWidth)
      val quantized =
        if (quantization) (withBias >> weightWidth).asUInt.resize(outputWidth)
        else withBias.asUInt.resize(outputWidth)
      // Clear the accumulator register to prepare for the next round.
      for (k <- 0 until fullconnectionNum) accVec(k) := 0
      inputCnt.clear()
      outputCnt.increment()
      when(outputCnt.willOverflow) { computing := False } .otherwise { computing := True }
      io.post.valid := True
      io.post.payload := quantized.asSInt
    }
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object FullConnectionGen {
//   def main(args: Array[String]): Unit = {
//     // // Basic full connection layer
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new FullConnection(FullConnectionConfig(
//     //     inputWidth = 8,
//     //     outputWidth = 16,
//     //     weightWidth = 8,
//     //     biasWidth = 8,
//     //     inputSize = 192,  // 4x4x12 flattened
//     //     outputSize = 10,  // 10 classes
//     //     useBias = true,
//     //     quantization = false))
//     // ).printPruned()

//     // Basic full connection layer streaming weights/bias with multi-input
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new FullConnection(FullConnectionConfig(
//         inputWidth = 8,
//         outputWidth = 16,
//         weightWidth = 8,
//         biasWidth = 8,
//         inputSize = 192,  // 4x4x12 flattened
//         outputSize = 10,  // 10 classes
//         useBias = true,
//         quantization = false))
//     ).printPruned()
//   }
// }

// object FullConnectionLayerGen {
//   def main(args: Array[String]): Unit = {
//     // Basic full connection layer streaming weights/bias with multi-input
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new FullConnectionLayer(FullConnectionLayerConfig(
//         fullconnectionNum = 6,
//         fullconnectionConfig = FullConnectionConfig(
//           inputWidth = 8,
//           outputWidth = 16,
//           weightWidth = 8,
//           biasWidth = 8,
//           inputSize = 192,  // 4x4x12 flattened
//           outputSize = 10,  // 10 classes
//           useBias = true,
//           quantization = false)))
//     ).printPruned()
//   }
// }
