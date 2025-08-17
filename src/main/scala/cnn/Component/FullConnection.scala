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
case class WeightInterface(dataWidth: Int, inputSize: Int, outputSize: Int) extends Bundle with IMasterSlave {
  val weights = Vec(SInt(dataWidth bits), inputSize * outputSize)
  val bias = Vec(SInt(dataWidth bits), outputSize)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = WeightInterface(dataWidth, inputSize, outputSize)
}

/**
 * Weight loader for dynamic weight updates
 */
class WeightLoader(dataWidth: Int, inputSize: Int, outputSize: Int) extends Component {
  val io = new Bundle {
    val load = in Bool()
    val addr = in UInt(log2Up(inputSize * outputSize) bits)
    val weight_data = in SInt(dataWidth bits)
    val bias_addr = in UInt(log2Up(outputSize) bits)
    val bias_data = in SInt(dataWidth bits)
    val weights = master(WeightInterface(dataWidth, inputSize, outputSize))
  }

  // Weight memory
  val weightMem = Mem(SInt(dataWidth bits), inputSize * outputSize)
  val biasMem = Mem(SInt(dataWidth bits), outputSize)

  // Load weights
  when(io.load) {
    weightMem(io.addr) := io.weight_data
    biasMem(io.bias_addr) := io.bias_data
  }

  // Connect to output, synchronously
  for (i <- 0 until inputSize * outputSize) {
    io.weights.weights(i) := weightMem.readAsync(U(i, log2Up(inputSize * outputSize) bits))
  }
  for (i <- 0 until outputSize) {
    io.weights.bias(i) := biasMem.readAsync(U(i, log2Up(outputSize) bits))
  }
}

/**
 * Pipelined Full Connection for better performance
 */
class PipelinedFullConnection(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(Vec(SInt(inputWidth bits), inputSize)))
    val post = master(Stream(Vec(SInt(outputWidth bits), outputSize)))
    val weights = slave(WeightInterface(weightWidth, inputSize, outputSize))
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
        val weight = io.weights.weights(i * inputSize + j)
        weight * stage1_input(j)
      }).reduce(_ + _)

      val acc = dotProduct.resize(outputWidth)

      // Add bias if enabled
      val withBias = if (useBias) {
        acc + io.weights.bias(i).resize(outputWidth)
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
 * Full Connection module
 */
class FullConnection(config: FullConnectionConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(Vec(SInt(inputWidth bits), inputSize)))
    val post = master(Stream(Vec(SInt(outputWidth bits), outputSize)))
    val weights = slave(WeightInterface(weightWidth, inputSize, outputSize))
  }

  // Ready signal
  io.pre.ready := True

  // Matrix multiplication: Y = W * X + b
  def matrixMultiply(input: Vec[SInt], weights: Vec[SInt]): Vec[SInt] = {
    val result = Vec(SInt(outputWidth bits), outputSize)

    for (i <- 0 until outputSize) {
      val sum = Reg(SInt(outputWidth bits)) init(0)
      val acc = SInt(outputWidth bits)

      // Compute dot product for output neuron i
      val dotProduct = (0 until inputSize).map(j => {
        val weight = weights(i * inputSize + j)
        weight * input(j)
      }).reduce(_ + _)

      acc := dotProduct.resize(outputWidth)

      // Add bias if enabled
      val withBias = if (useBias) {
        acc + io.weights.bias(i).resize(outputWidth)
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

  val outputValues = matrixMultiply(io.pre.payload, io.weights.weights)

  // Stream output logic
  io.post.valid := Mux(io.EN, io.pre.valid, io.pre.valid)
  io.post.payload := Mux(io.EN, outputValues, Vec(io.pre.payload.take(outputSize)))
}


object FullConnectionGen {
  def main(args: Array[String]): Unit = {
    println("Generating FullConnection modules...")

    // Weight loader
    SpinalConfig(targetDirectory = "rtl").generateVerilog(
      new WeightLoader(8, 784, 10)
    ).printPruned()

    // // Basic full connection layer
    // SpinalConfig(targetDirectory = "rtl").generateVerilog(
    //   new FullConnection(FullConnectionConfig(
    //     inputWidth = 8,
    //     outputWidth = 16,
    //     weightWidth = 8,
    //     biasWidth = 8,
    //     inputSize = 784,  // 28x28 flattened
    //     outputSize = 10,  // 10 classes
    //     useBias = true,
    //     signed = false,
    //     quantization = false))
    // ).printPruned()

    // // Pipelined full connection
    // SpinalConfig(targetDirectory = "rtl").generateVerilog(
    //   new PipelinedFullConnection(FullConnectionConfig(
    //     inputWidth = 8,
    //     outputWidth = 16,
    //     weightWidth = 8,
    //     biasWidth = 8,
    //     inputSize = 256,
    //     outputSize = 128,
    //     useBias = true,
    //     signed = false,
    //     quantization = true))
    // ).printPruned()

    println("FullConnection modules generated successfully!")
  }
}
