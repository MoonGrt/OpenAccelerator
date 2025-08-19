package cnn

import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* -------------------------------- Shift Ram -------------------------------- */
/* --------------------------------------------------------------------------- */
// ShiftRam is a (dynamic) shift register with wrap-around behavior
class ShiftRam(dataWidth: Int, lineLength: Int = 0, lineLengthDyn: Boolean = false) extends Component {
  val io = new Bundle {
    val CE = in Bool()
    val D  = in Bits(dataWidth bits)
    val Q  = out Bits(dataWidth bits)
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  val mem     = Mem(Bits(dataWidth bits), (lineLength - 1))
  val wrPtr = RegInit(U(0, log2Up((lineLength - 1)) bits))
  val rdPtr = RegInit(U(0, log2Up((lineLength - 1)) bits))

  io.Q := mem.readSync(rdPtr)
  when(io.CE) {
    mem(wrPtr) := io.D
    if (lineLengthDyn) {
      when(wrPtr === io.linewidth - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
      when(rdPtr === io.linewidth - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
    } else {
      when(wrPtr === (lineLength - 1 ) - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
      when(rdPtr === (lineLength - 1 ) - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* ------------------------------ Shift Column ------------------------------ */
/* -------------------------------------------------------------------------- */
/**
 * ShiftColumn2x2Interface is a bundle for the shift column output signals.
 */
case class ShiftColumn2x2Interface(dataWidth: Int) extends Bundle with IMasterSlave {
  val c1 = SInt(dataWidth bits)
  val c2 = SInt(dataWidth bits)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = ShiftColumn2x2Interface(dataWidth)

  def << (that: ShiftColumn2x2Interface): Unit = {
    this.c1 := that.c1
    this.c2 := that.c2
  }
}

/* ---------------------------------------------------------------------------- */
/* ---------------------------------- Matrix ---------------------------------- */
/* ---------------------------------------------------------------------------- */
/**
 * Matrix builds a kernelSizexkernelSize neighborhood for a single channel.
 * The output matrix signals are valid two cycles after input pre_de (matches Verilog behavior).
 */
// General Matrix Interface
case class MatrixInterface(dataWidth: Int, kernelSize: Int) extends Bundle with IMasterSlave {
  val de = Bool()
  val m = Vec(Vec(SInt(dataWidth bits), kernelSize), kernelSize)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = MatrixInterface(dataWidth, kernelSize)

  def << (that: MatrixInterface): Unit = {
    this.de := that.de
    for(i <- 0 until kernelSize; j <- 0 until kernelSize){
      this.m(i)(j) := that.m(i)(j)
    }
  }
}

// Matrix Configuration
case class MatrixConfig(
  dataWidth     : Int,           // bits per pixel
  lineLength    : Int,           // number of pixels per line
  kernelSize    : Int,           // kernel size
  padding       : Int = 0,       // padding size
  stride        : Int = 2,       // stride size
  lineLengthDyn : Boolean = true // dynamic line length
)

// Matrix Component
class Matrix(config: MatrixConfig) extends Component {
  import config._
  val io = new Bundle {
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix = master(MatrixInterface(dataWidth, kernelSize))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True
  // A total of (kernelSize - 1) line buffers are required.
  val lineBuffers = Array.fill(kernelSize-1)(new ShiftRam(dataWidth, lineLength, lineLengthDyn))
  // Row data stream, row(0) is the latest row, row(kernelSize-1) is the oldest row.
  val rows = Array.fill(kernelSize)(Reg(Bits(dataWidth bits)) init(0))

  // Write the input pixels sequentially into lineBuffer
  lineBuffers.zipWithIndex.foreach { case (ram, idx) =>
    ram.io.CE := io.pre.valid
    if (idx == 0) {
      ram.io.D := io.pre.payload.asBits
    } else {
      ram.io.D := rows(idx).asBits
    }
    if (lineLengthDyn) { ram.io.linewidth := io.linewidth }
    rows(idx + 1) := ram.io.Q
  }
  rows(0) := io.pre.payload.asBits

  // Column shift register
  val shiftRegs = Array.fill(kernelSize, kernelSize)(Reg(Bits(dataWidth bits)) init(0))
  when(io.pre.valid){
    for(i <- 0 until kernelSize){
      for(j <- 0 until kernelSize){
        if (j == kernelSize-1){
          shiftRegs(i)(j) := rows(i)
        } else {
          shiftRegs(i)(j) := shiftRegs(i)(j+1)
        }
      }
    }
  }

  // Two-cycle delay de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)
  val raw_matrix_de = pre_de_r(1)

  // Stride count
  val strideCounter = Reg(UInt(log2Up(stride) bits)) init(0)
  when(raw_matrix_de){
    if (stride > 1){
      when(strideCounter === stride - 1){
        strideCounter := 0
      } otherwise {
        strideCounter := strideCounter + 1
      }
    }
  }

  io.matrix.de := raw_matrix_de && (strideCounter === 0)

  // Output control logic - only output when not padding and stride is valid
  for(i <- 0 until kernelSize){
    for(j <- 0 until kernelSize){
      io.matrix.m(i)(j) := shiftRegs(i)(j).asSInt
    }
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object MatrixGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Matrix(MatrixConfig(
//         dataWidth = 8,
//         lineLength = 28,
//         kernelSize = 5,
//         padding = 0,
//         stride = 1,
//         lineLengthDyn = false))
//     ).printPruned()
//   }
// }
