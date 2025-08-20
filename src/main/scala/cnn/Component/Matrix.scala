package cnn

import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* -------------------------------- Shift Ram -------------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * ShiftRam is a (dynamic) shift register with wrap-around behavior
 */
class ShiftRam(dataWidth: Int, rowNumDyn: Boolean = false, rowNum: Int = 8) extends Component {
  val io = new Bundle {
    val CE = in Bool()
    val D  = in Bits(dataWidth bits)
    val Q  = out Bits(dataWidth bits)
    val rownum = if (rowNumDyn) in UInt(log2Up((rowNum - 1)) bits) else null
  }

  val mem = Mem(Bits(dataWidth bits), (rowNum - 1))
  val wrPtr = RegInit(U(0, log2Up((rowNum - 1)) bits))
  val rdPtr = RegInit(U(0, log2Up((rowNum - 1)) bits))

  io.Q := mem.readSync(rdPtr)
  when(io.CE) {
    mem(wrPtr) := io.D
    if (rowNumDyn) {
      when(wrPtr === io.rownum - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
      when(rdPtr === io.rownum - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
    } else {
      when(wrPtr === (rowNum - 1 ) - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
      when(rdPtr === (rowNum - 1 ) - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* ------------------------------ Shift Column ------------------------------ */
/* -------------------------------------------------------------------------- */
/**
 * ShiftColumnInterface is a bundle for the shift column output signals.
 */
// ShiftColumn Interface
case class ShiftColumnInterface(dataWidth: Int, kernelSize: Int) extends Bundle with IMasterSlave {
  val de = Bool()
  val c = Vec(SInt(dataWidth bits), kernelSize)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = ShiftColumnInterface(dataWidth, kernelSize)

  def << (that: ShiftColumnInterface): Unit = {
    this.de := that.de
    for(i <- 0 until kernelSize){
      this.c(i) := that.c(i)
    }
  }
}

// ShiftColumn Configuration
case class ShiftColumnConfig(
  dataWidth  : Int,            // bits per pixel
  kernelSize : Int,            // kernel size
  rowNumDyn  : Boolean = true, // dynamic line length
  rowNum     : Int = 8         // number of pixels per row
)

// ShiftColumn Component
class ShiftColumn(config: ShiftColumnConfig) extends Component {
  import config._
  val io = new Bundle {
    val pre = slave(Stream(SInt(dataWidth bits)))
    val column = master(ShiftColumnInterface(dataWidth, kernelSize))
    val rownum = if (rowNumDyn) in UInt(log2Up((rowNum - 1)) bits) else null
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True
  // A total of (kernelSize - 1) line buffers are required.
  val lineBuffers = Array.fill(kernelSize - 1)(new ShiftRam(dataWidth, rowNumDyn, rowNum))
  // Row data stream, row(0) is the latest row, row(kernelSize - 1) is the oldest row.
  val row0 = Reg(Bits(dataWidth bits)) init(0)
  val rows = Array.fill(kernelSize - 1)(Bits(dataWidth bits))

  // Write the input pixels sequentially into lineBuffer
  row0 := io.pre.payload.asBits
  lineBuffers.zipWithIndex.foreach { case (ram, idx) =>
    ram.io.CE := io.pre.valid
    if (idx == 0) {
      ram.io.D := row0
    } else {
      ram.io.D := rows(idx - 1).asBits
    }
    if (rowNumDyn) { ram.io.rownum := io.rownum }
    rows(idx) := ram.io.Q
  }

  // Output control logic
  io.column.de := RegNext(io.pre.valid)
  io.column.c(0) := row0.asSInt
  for(i <- 0 until kernelSize - 1){
    io.column.c(i + 1) := rows(i).asSInt
  }
}


/* ---------------------------------------------------------------------------- */
/* ---------------------------------- Matrix ---------------------------------- */
/* ---------------------------------------------------------------------------- */
/**
 * Matrix builds a kernelSize x kernelSize neighborhood.
 */
// Matrix Interface
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
  dataWidth  : Int,      // bits per pixel
  kernelSize : Int,      // kernel size
  padding    : Int = 0,  // padding num
  stride     : Int = 1,  // stride num
  rowNum     : Int = 24, // dynamic row length
  colNum     : Int = 24  // dynamic column length
)

// Matrix Component
class Matrix(config: MatrixConfig) extends Component {
  import config._
  require(padding <= (kernelSize - 1) / 2, "Padding should be less than (kernelSize - 1) / 2")
  val io = new Bundle {
    val column = slave(ShiftColumnInterface(dataWidth, kernelSize))
    val matrix = master(MatrixInterface(dataWidth, kernelSize))
  }

  // Column cache: (kernelSize - 1) x kernelSize
  val cols = Vec(Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize), kernelSize - 1)
  // Horizontal stride counter
  val strideCntH = if (stride > 1) Reg(UInt(log2Up(stride) bits)).init(0) else null
  // Vertical stride counter
  val strideCntV = if (stride > 1) Reg(UInt(log2Up(stride) bits)).init(0) else null
  // Row and column counters (padding judgment)
  val rowCnt = Reg(UInt(log2Up(rowNum + kernelSize - 1) bits)) init 0
  val colCnt = Reg(UInt(log2Up(colNum + kernelSize - 1) bits)) init 0

  // Column input promotion
  when(io.column.de) {
    // Newest column goes into cols(0)
    for (i <- 0 until kernelSize) {
      cols(0)(i) := io.column.c(i)
    }
    // Move old columns to the right
    for (j <- (kernelSize - 2) downto 1) {
      for (i <- 0 until kernelSize) {
        cols(j)(i) := cols(j - 1)(i)
      }
    }
  }
  // Padding & Stride
  val paddingValid = (colCnt >= (kernelSize - 1 + padding * 2)) && (rowCnt >= (kernelSize - 1 + padding * 2))
  when(io.column.de) {
    rowCnt := rowCnt + 1
    // Horizontal stride update
    if (stride > 1) {
      when (paddingValid) {
        strideCntH := strideCntH + 1
        when(strideCntH === (stride - 1)) { strideCntH := 0 }
      }
    }
    when(rowCnt === (rowNum - 1)) {
      rowCnt := 0
      colCnt := colCnt + 1
      // Vertical stride update
      if (stride > 1) {
        when (paddingValid) {
          strideCntV := strideCntV + 1
          when(strideCntV === (stride - 1)) { strideCntV := 0 }
        }
      }
    }
  }

  // Output valid conditions: horizontal stride + vertical stride + passed the padding zone
  if (stride > 1) {
    val strideValid = (strideCntH === 0) && (strideCntV === 0)
    io.matrix.de := io.column.de && strideValid && paddingValid
  } else {
    io.matrix.de := io.column.de && paddingValid
  }
  // Output matrix construction
  for (i <- 0 until kernelSize) {
    for (j <- 0 until kernelSize) {
      if (j == 0) {
        // The newest column comes directly from input
        io.matrix.m(i)(j) := io.column.c(i)
      } else {
        // Historical columns from cache
        io.matrix.m(i)(j) := cols(j - 1)(i)
      }
    }
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object ShiftColumnGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new ShiftColumn(ShiftColumnConfig(
//         dataWidth = 8,
//         rowNum = 28,
//         colNum = 28,
//         kernelSize = 5,
//         rowNumDyn = false))
//     ).printPruned()
//   }
// }

// object MatrixGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Matrix(MatrixConfig(
//         dataWidth = 8,
//         kernelSize = 5,
//         padding = 0,
//         stride = 1,
//         rowNum = 28,
//         colNum = 28,))
//     ).printPruned()
//   }
// }
