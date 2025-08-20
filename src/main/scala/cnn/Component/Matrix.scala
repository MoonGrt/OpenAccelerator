package cnn

import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* -------------------------------- Shift Ram -------------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * ShiftRam is a (dynamic) shift register with wrap-around behavior
 */
class ShiftRam(dataWidth: Int, lineLengthDyn: Boolean = false, lineLength: Int = 8) extends Component {
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
  dataWidth     : Int,            // bits per pixel
  kernelSize    : Int,            // kernel size
  lineLengthDyn : Boolean = true, // dynamic line length
  lineLength    : Int = 8         // number of pixels per line
)

// ShiftColumn Component
class ShiftColumn(config: ShiftColumnConfig) extends Component {
  import config._
  val io = new Bundle {
    val pre = slave(Stream(SInt(dataWidth bits)))
    val column = master(ShiftColumnInterface(dataWidth, kernelSize))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True
  // A total of (kernelSize - 1) line buffers are required.
  val lineBuffers = Array.fill(kernelSize - 1)(new ShiftRam(dataWidth, lineLengthDyn, lineLength))
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

  // Two-cycle delay de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)

  // Output control logic
  io.column.de := pre_de_r(1)
  for(i <- 0 until kernelSize){
    io.column.c(i) := rows(i).asSInt
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
  dataWidth  : Int,     // bits per pixel
  kernelSize : Int,     // kernel size
  padding    : Int = 0, // padding num
  stride     : Int = 2  // stride num
)

// Matrix Component
class Matrix(config: MatrixConfig) extends Component {
  import config._
  val io = new Bundle {
    val column = slave(ShiftColumnInterface(dataWidth, kernelSize))
    val matrix = master(MatrixInterface(dataWidth, kernelSize))
  }

  // 列缓存：一共 kernelSize 个列，每列 kernelSize 个像素
  val cols = Vec(Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize), kernelSize)
  // 控制 stride 的水平步进
  // val strideCnt = Reg(UInt(log2Up(stride) bits)) init(0)
  val strideCnt = if (stride > 1) {
      Reg(UInt(log2Up(stride) bits)) init 0
    } else {
      U(0, 0 bits)
    }

  // 输入有效时，更新 strideCnt
  if (stride > 1) {
    when(io.column.de) {
      strideCnt := strideCnt + 1
      when(strideCnt === (stride - 1)) { strideCnt := 0 }
    }
  }

  // 输入有效 且 stride 对齐时，移位矩阵
  when(io.column.de && (strideCnt === 0)) {
    // 右移旧列
    for (j <- (kernelSize - 1) downto 1) {
      for (i <- 0 until kernelSize) {
        cols(j)(i) := cols(j - 1)(i)
      }
    }
    // 新列写入 cols(0)
    for (i <- 0 until kernelSize) {
      cols(0)(i) := io.column.c(i)
    }
  }

  // // Padding 处理：
  // // 当在边缘时，部分窗口元素应该输出 0。
  // // 为简单起见，这里只在外部控制阶段考虑 padding 区域（即用 cols 中的数据 + 额外补零逻辑）。
  // def getWithPadding(i: Int, j: Int): SInt = {
  //   val inPadRow = (i < padding) || (i >= kernelSize - padding)
  //   val inPadCol = (j < padding) || (j >= kernelSize - padding)
  //   val value = S(dataWidth bits)
  //   value := cols(j)(i)
  //   when(inPadRow || inPadCol) { value := 0}
  //   value
  // }

  // 输出
  io.matrix.de := io.column.de && (strideCnt === 0)
  for (i <- 0 until kernelSize) {
    for (j <- 0 until kernelSize) {
      io.matrix.m(i)(j) := cols(j)(i)
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
//         lineLength = 28,
//         kernelSize = 5,
//         lineLengthDyn = false))
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
//         stride = 1))
//     ).printPruned()
//   }
// }
