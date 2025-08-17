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
  weightWidth  : Int,             // bits per weight element
  biasWidth    : Int,             // bits per bias element
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
  inputSize: Int,
  outputSize: Int,
  useBias: Boolean = true
) extends Bundle with IMasterSlave {
  val weight = Vec(SInt(weightWidth bits), inputSize * outputSize)
  val bias = Vec(SInt(biasWidth bits), outputSize)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = WeightBiasInterface(weightWidth, biasWidth, inputSize, outputSize, useBias)
}

case class WeightBiasStreamInterface(
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
  override def clone = WeightBiasStreamInterface(weightWidth, biasWidth, useBias)
}

/**
 * Full Connection module
 */
class FullConnection(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(Vec(SInt(inputWidth bits), inputSize)))
    val post = master(Stream(Vec(SInt(outputWidth bits), outputSize)))
    val wb = slave(WeightBiasInterface(weightWidth, biasWidth, inputSize, outputSize, useBias))
  }

  // Ready signal
  io.pre.ready := True

  // Matrix multiplication: Y = W * X + b
  def matrixMultiply(input: Vec[SInt], weight: Vec[SInt]): Vec[SInt] = {
    val result = Vec(SInt(outputWidth bits), outputSize)

    for (i <- 0 until outputSize) {
      val sum = Reg(SInt(outputWidth bits)) init(0)
      val acc = SInt(outputWidth bits)

      // Compute dot product for output neuron i
      val dotProduct = (0 until inputSize).map(j => {
        val w = weight(i * inputSize + j)
        w * input(j)
      }).reduce(_ + _)
      acc := dotProduct.resize(outputWidth)

      // Add bias if enabled
      val withBias = if (useBias) {
        acc + io.wb.bias(i).resize(outputWidth)
      } else {
        acc
      }
      // Quantization if enabled
      val quantized = if (quantization) {
        // Simple quantization: right shift by weightWidth
        (withBias >> weightWidth).asUInt.resize(outputWidth)
      } else {
        withBias.asUInt.resize(outputWidth)
      }

      result(i) := quantized.asSInt
    }
    result
  }

  val outputValues = matrixMultiply(io.pre.payload, io.wb.weight)

  // Stream output logic
  io.post.valid := Mux(io.EN, io.pre.valid, io.pre.valid)
  io.post.payload := Mux(io.EN, outputValues, Vec(io.pre.payload.take(outputSize)))
}

/**
 * Full Connection module (streaming weights/bias)
 */
class FullConnectionStream(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val wb   = slave(WeightBiasStreamInterface(weightWidth, biasWidth, useBias))
    val pre  = slave(Stream(SInt(inputWidth bits)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // Storage for weights and bias
  val weightMem = Mem(SInt(weightWidth bits), inputSize * outputSize)
  val biasMem   = if (useBias) Mem(SInt(biasWidth bits), outputSize) else null
  val weightCnt = Counter(inputSize * outputSize)
  val biasCnt   = if (useBias) Counter(outputSize) else null

  // Streaming reception weighting
  io.wb.weight.ready := True
  when(io.wb.weight.valid && io.wb.weight.ready) {
    weightMem.write(weightCnt.value, io.wb.weight.payload)
    weightCnt.increment()
  }
  if (useBias) {
    io.wb.bias.ready := True
    when(io.wb.bias.valid && io.wb.bias.ready) {
      biasMem.write(biasCnt.value, io.wb.bias.payload)
      biasCnt.increment()
    }
  }

  // Input buffer
  val inputVec = Vec(Reg(SInt(inputWidth bits)) init(0), inputSize)
  val inputCnt = Counter(inputSize)
  val frameReady = Reg(Bool()) init(False)
  io.pre.ready := !frameReady
  when(io.pre.valid && io.pre.ready) {
    inputVec(inputCnt.value) := io.pre.payload
    inputCnt.increment()
    when(inputCnt.willOverflow) {
      frameReady := True
    }
  }

  // Output buffer
  val outputCnt = Counter(outputSize)
  val outputReg = Reg(SInt(outputWidth bits)) init(0)
  val computing = Reg(Bool()) init(False)
  io.post.valid := computing
  io.post.payload := outputReg

  when(frameReady && !computing) {
    computing := True
    outputCnt.clear()
  }

  when(computing && io.post.ready) {
    // Compute each output neuron
    val dotProduct = (0 until inputSize).map(j =>
      weightMem.readSync((outputCnt.value * inputSize + j).resized) * inputVec(j)
    ).reduce(_ + _)
    val acc = dotProduct.resize(outputWidth)
    val withBias = if (useBias) acc + biasMem(outputCnt.value).resize(outputWidth) else acc
    val quantized = if (quantization) (withBias >> weightWidth).asUInt.resize(outputWidth) else withBias.asUInt.resize(outputWidth)
    outputReg := quantized.asSInt

    outputCnt.increment()
    when(outputCnt.willOverflow) {
      computing := False
      frameReady := False
      inputCnt.clear()
    }
  }
}

/**
 * Pipelined Full Connection for better performance
 */
class PipelinedFullConnection(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val wb = slave(WeightBiasInterface(weightWidth, biasWidth, inputSize, outputSize, useBias))
    val pre = slave(Stream(Vec(SInt(inputWidth bits), inputSize)))
    val post = master(Stream(Vec(SInt(outputWidth bits), outputSize)))
  }

  // Ready signal
  io.pre.ready := True

  // Pipeline stages
  val stage1_valid = Reg(Bool()) init(False)
  val stage1_input = Reg(Vec(SInt(inputWidth bits), inputSize))
  val stage2_valid = Reg(Bool()) init(False)
  val stage2_result = Reg(Vec(SInt(outputWidth bits), outputSize))

  // Stage 1: Register input
  when(io.pre.valid && io.pre.ready) {
    stage1_valid := True
    stage1_input := io.pre.payload
  } otherwise {
    stage1_valid := False
  }

  // Stage 2: Matrix multiplication
  when(stage1_valid) {
    stage2_valid := True

    // Compute each output neuron
    for (i <- 0 until outputSize) {
      val dotProduct = (0 until inputSize).map(j => {
        val weight = io.wb.weight(i * inputSize + j)
        weight * stage1_input(j)
      }).reduce(_ + _)
      val acc = dotProduct.resize(outputWidth)

      // Add bias if enabled
      val withBias = if (useBias) {
        acc + io.wb.bias(i).resize(outputWidth)
      } else {
        acc
      }
      // Quantization if enabled
      val quantized = if (quantization) {
        (withBias >> weightWidth).asUInt.resize(outputWidth)
      } else {
        withBias.asUInt.resize(outputWidth)
      }

      stage2_result(i) := quantized.asSInt
    }
  } otherwise {
    stage2_valid := False
  }

  // Stream output logic
  io.post.valid := Mux(io.EN, stage2_valid, io.pre.valid)
  io.post.payload := Mux(io.EN, stage2_result, Vec(io.pre.payload.take(outputSize)))
}

/**
 * Pipelined Full Connection (streaming weights/bias)
 */
class PipelinedFullConnectionStream(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val wb   = slave(WeightBiasStreamInterface(weightWidth, biasWidth, useBias))
    val pre  = slave(Stream(SInt(inputWidth bits)))
    val post = master(Stream(SInt(outputWidth bits)))
  }

  // Storage for weights and bias
  val weightMem = Mem(SInt(weightWidth bits), inputSize * outputSize)
  val biasMem   = if (useBias) Mem(SInt(biasWidth bits), outputSize) else null
  val weightCnt = Counter(inputSize * outputSize)
  val biasCnt   = if (useBias) Counter(outputSize) else null

  // Streaming reception weighting
  io.wb.weight.ready := True
  when(io.wb.weight.valid && io.wb.weight.ready) {
    weightMem.write(weightCnt.value, io.wb.weight.payload)
    weightCnt.increment()
  }
  if (useBias) {
    io.wb.bias.ready := True
    when(io.wb.bias.valid && io.wb.bias.ready) {
      biasMem.write(biasCnt.value, io.wb.bias.payload)
      biasCnt.increment()
    }
  }

  // Input buffer
  val inputVec = Vec(Reg(SInt(inputWidth bits)) init(0), inputSize)
  val inputCnt = Counter(inputSize)
  val frameReady = Reg(Bool()) init(False)
  io.pre.ready := !frameReady
  when(io.pre.valid && io.pre.ready) {
    inputVec(inputCnt.value) := io.pre.payload
    inputCnt.increment()
    when(inputCnt.willOverflow) {
      frameReady := True
    }
  }

  // Stage1 -> Ready to enter
  val stage1_valid = Reg(Bool()) init(False)
  val stage1_idx   = Reg(UInt(log2Up(outputSize) bits)) init(0)
  when(frameReady && !stage1_valid) {
    stage1_valid := True
    stage1_idx := 0
  }

  // Stage2 -> Compute each output neuron
  val stage2_valid = Reg(Bool()) init(False)
  val stage2_val   = Reg(SInt(outputWidth bits)) init(0)
  val stage2_idx   = Reg(UInt(log2Up(outputSize) bits)) init(0)

  when(stage1_valid) {
    val dotProduct = (0 until inputSize).map(j =>
      weightMem.readSync(stage1_idx * inputSize + j) * inputVec(j)
    ).reduce(_ + _)
    val acc = dotProduct.resize(outputWidth)
    val withBias = if (useBias) acc + biasMem(stage1_idx).resize(outputWidth) else acc
    val quantized = if (quantization) (withBias >> weightWidth).asUInt.resize(outputWidth) else withBias.asUInt.resize(outputWidth)
    stage2_val := quantized.asSInt
    stage2_idx := stage1_idx
    stage2_valid := True

    when(stage1_idx === outputSize - 1) {
      stage1_valid := False
      frameReady := False
      inputCnt.clear()
    } otherwise {
      stage1_idx := stage1_idx + 1
    }
  }

  // Stream output logic
  io.post.valid := stage2_valid
  io.post.payload := stage2_val
  when(io.post.fire) {
    stage2_valid := False
  }
}


// object FullConnectionGen {
//   def main(args: Array[String]): Unit = {
//     println("Generating FullConnection modules...")

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
//     //     signed = false,
//     //     quantization = false))
//     // ).printPruned()
//     // // Pipelined full connection
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new PipelinedFullConnection(FullConnectionConfig(
//     //     inputWidth = 8,
//     //     outputWidth = 16,
//     //     weightWidth = 8,
//     //     biasWidth = 8,
//     //     inputSize = 192,  // 4x4x12 flattened
//     //     outputSize = 128,
//     //     useBias = true,
//     //     signed = false,
//     //     quantization = true))
//     // ).printPruned()

//     // Basic full connection layer streaming weights/bias
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new FullConnectionStream(FullConnectionConfig(
//         inputWidth = 8,
//         outputWidth = 16,
//         weightWidth = 8,
//         biasWidth = 8,
//         inputSize = 192,  // 4x4x12 flattened
//         outputSize = 10,  // 10 classes
//         useBias = true,
//         signed = false,
//         quantization = false))
//     ).printPruned()
//     // // Pipelined full connection streaming weights/bias
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new PipelinedFullConnectionStream(FullConnectionConfig(
//     //     inputWidth = 8,
//     //     outputWidth = 16,
//     //     weightWidth = 8,
//     //     biasWidth = 8,
//     //     inputSize = 192,  // 4x4x12 flattened
//     //     outputSize = 128,
//     //     useBias = true,
//     //     signed = false,
//     //     quantization = true))
//     // ).printPruned()

//     println("FullConnection modules generated successfully!")
//   }
// }
