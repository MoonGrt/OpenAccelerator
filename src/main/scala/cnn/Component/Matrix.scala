package cnn

import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* -------------------------------- Shift Ram -------------------------------- */
/* --------------------------------------------------------------------------- */
// ShiftRamDyn is a dynamic shift register with wrap-around behavior
class ShiftRamDyn(dataWidth: Int, lineLengthBits: Int) extends Component {
  val io = new Bundle {
    val CE = in Bool()
    val D  = in Bits(dataWidth bits)
    val Q  = out Bits(dataWidth bits)
    val LINE_LENGTH = in UInt(lineLengthBits bits)
  }

  // Parameterize a max length by using a Mem; LINE_LENGTH controls wrap mask
  val mem = Mem(Bits(dataWidth bits), 1 << lineLengthBits)
  val wrPtr = RegInit(U(0, lineLengthBits bits))
  val rdPtr = RegInit(U(0, lineLengthBits bits))

  io.Q := mem.readSync(rdPtr)
  when(io.CE) {
    // write then read (same cycle behavior as original Verilog)
    mem(wrPtr) := io.D
    // increment with wrap
    when(wrPtr === (io.LINE_LENGTH - 1) - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
    when(rdPtr === (io.LINE_LENGTH - 1) - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
  }
}

// ShiftRam is a simple shift register with wrap-around behavior
class ShiftRam(dataWidth: Int, lineLength: Int) extends Component {
  val io = new Bundle {
    val CE = in Bool()
    val D  = in Bits(dataWidth bits)
    val Q  = out Bits(dataWidth bits)
  }

  // Parameterize a max length by using a Mem; LINE_LENGTH controls wrap mask
  val mem = Mem(Bits(dataWidth bits), (lineLength - 1))
  val wrPtr = RegInit(U(0, log2Up((lineLength - 1)) bits))
  val rdPtr = RegInit(U(0, log2Up((lineLength - 1)) bits))

  io.Q := mem.readSync(rdPtr)
  when(io.CE) {
    // write then read (same cycle behavior as original Verilog)
    mem(wrPtr) := io.D
    // increment with wrap
    when(wrPtr === (lineLength - 1 ) - 1) { wrPtr := U(0) } .otherwise { wrPtr := wrPtr + 1 }
    when(rdPtr === (lineLength - 1 ) - 1) { rdPtr := U(0) } .otherwise { rdPtr := rdPtr + 1 }
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
 * Matrix2x2 builds a 2x2 neighborhood for a single channel.
 * The output matrix signals are valid two cycles after input pre_de (matches Verilog behavior).
 */
case class Matrix2x2Interface(dataWidth: Int) extends Bundle with IMasterSlave {
  val m11 = SInt(dataWidth bits)
  val m12 = SInt(dataWidth bits)
  val m21 = SInt(dataWidth bits)
  val m22 = SInt(dataWidth bits)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = Matrix2x2Interface(dataWidth)

  def << (that: Matrix2x2Interface): Unit = {
    this.m11 := that.m11
    this.m12 := that.m12
    this.m21 := that.m21
    this.m22 := that.m22
  }
}

class Matrix2x2Dyn(
  dataWidth: Int,
  lineLengthBits: Int,
  padding: Int,
  stride: Int
) extends Component {
  val io = new Bundle {
    val LINEWIDTH = in UInt(lineLengthBits bits)
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix_de = out Bool()
    val matrix = master(Matrix2x2Interface(dataWidth))
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True

  // row shift: row1 is the straight input, row1 and row2 come from two line buffers
  val row1 = Bits(dataWidth bits)
  val row2 = Reg(Bits(dataWidth bits))
  when(io.pre.valid) { row2 := io.pre.payload.asBits }

  val ram1 = new ShiftRamDyn(dataWidth, lineLengthBits)
  ram1.io.CE := io.pre.valid
  ram1.io.D  := row2
  ram1.io.LINE_LENGTH := io.LINEWIDTH
  row1 := ram1.io.Q

  // two-cycle sync for de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)
  io.matrix_de := pre_de_r(1)

  // 2x2 shift registers per row
  val m11 = Reg(Bits(dataWidth bits)) init(0)
  val m12 = Reg(Bits(dataWidth bits)) init(0)
  val m21 = Reg(Bits(dataWidth bits)) init(0)
  val m22 = Reg(Bits(dataWidth bits)) init(0)

  when(io.pre.valid) {
    m11 := m12; m12 := m12
    m21 := m22; m22 := m22
  }

  io.matrix.m11 := m11.asSInt
  io.matrix.m12 := m12.asSInt
  io.matrix.m21 := m21.asSInt
  io.matrix.m22 := m22.asSInt
}

class Matrix2x2(
  dataWidth: Int,
  lineLength: Int,
  padding: Int = 1,
  stride: Int = 1
) extends Component {
  val kernelSize = 2
  val io = new Bundle {
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix_de = out Bool()
    val matrix = master(Matrix2x2Interface(dataWidth))
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True

  // row shift: row1 is the straight input, row1 and row2 come from two line buffers
  val row1 = Bits(dataWidth bits)
  val row2 = Reg(Bits(dataWidth bits))
  row2 := io.pre.payload.asBits

  val ram1 = new ShiftRam(dataWidth, lineLength)
  ram1.io.CE := io.pre.valid
  ram1.io.D  := row2
  row1 := ram1.io.Q

  // two-cycle sync for de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)
  val raw_matrix_de = pre_de_r(1)

  // Position tracking for padding and stride
  val pixelX = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val pixelY = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val lineCount = Reg(UInt(log2Up(lineLength) bits)) init(0)

  // Update position counters
  when(io.pre.valid && io.pre.ready) {
    when(pixelX === lineLength - 1) {
      pixelX := 0
      when(lineCount === lineLength - 1) {
        lineCount := 0
        pixelY := 0
      } otherwise {
        lineCount := lineCount + 1
        pixelY := pixelY + 1
      }
    } otherwise {
      pixelX := pixelX + 1
    }
  }

  // Convolution window center position (2 cycles delay to match matrix_de)
  val convX = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val convY = Reg(UInt(log2Up(lineLength) bits)) init(0)
  when(raw_matrix_de) {
    // Convolution window center is at current pixel position minus 2 (for 2x2 kernel)
    convX := pixelX - 2
    convY := pixelY - 2
  }

  // Padding logic: check if convolution window center is within valid area
  val isPadding = Bool()
  isPadding := (convX > (kernelSize - padding * 2 + 1)) || (convY > (kernelSize - padding * 2 + 1))

  // Stride logic: only output every stride-th convolution
  val strideCounter = Reg(UInt(log2Up(stride) bits)) init(0)
  val strideValid = Bool()
  strideValid := (strideCounter === 0)

  when(raw_matrix_de) {
    if (stride > 1) {
      when(strideCounter === stride - 1) {
        strideCounter := 0
      } otherwise {
        strideCounter := strideCounter + 1
      }
    }
  }

  // Output control logic - only output when not padding and stride is valid
  val shouldOutput = Bool()
  shouldOutput := raw_matrix_de && !isPadding && strideValid
  io.matrix_de := shouldOutput

  // 2x2 shift registers per row
  val m11, m12 = Reg(Bits(dataWidth bits)) init(0)
  val m21, m22 = Reg(Bits(dataWidth bits)) init(0)
  when(io.pre.valid) {
    m11 := m12; m12 := row1
    m21 := m22; m22 := row2
  }

  io.matrix.m11 := m11.asSInt
  io.matrix.m12 := m12.asSInt
  io.matrix.m21 := m21.asSInt
  io.matrix.m22 := m22.asSInt
}

/**
 * Matrix3x3 builds a 3x3 neighborhood for a single channel.
 * The output matrix signals are valid two cycles after input pre_de (matches Verilog behavior).
 */
case class Matrix3x3Interface(dataWidth: Int) extends Bundle with IMasterSlave {
  val m11 = SInt(dataWidth bits)
  val m12 = SInt(dataWidth bits)
  val m13 = SInt(dataWidth bits)
  val m21 = SInt(dataWidth bits)
  val m22 = SInt(dataWidth bits)
  val m23 = SInt(dataWidth bits)
  val m31 = SInt(dataWidth bits)
  val m32 = SInt(dataWidth bits)
  val m33 = SInt(dataWidth bits)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = Matrix3x3Interface(dataWidth)

  def << (that: Matrix3x3Interface): Unit = {
    this.m11 := that.m11
    this.m12 := that.m12
    this.m13 := that.m13
    this.m21 := that.m21
    this.m22 := that.m22
    this.m23 := that.m23
    this.m31 := that.m31
    this.m32 := that.m32
    this.m33 := that.m33
  }
}

class Matrix3x3Dyn(
  dataWidth: Int,
  lineLengthBits: Int,
  padding: Int,
  stride: Int
) extends Component {
  var kernelSize = 3
  require(padding <= (kernelSize - 1 ) / 2, "padding must be less than (kernelSize - 1 ) / 2")

  val io = new Bundle {
    val LINEWIDTH = in UInt(lineLengthBits bits)
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix_de = out Bool()
    val matrix = master(Matrix3x3Interface(dataWidth))
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True

  // row shift: row1 is the straight input, row1 and row2 come from two line buffers
  val row1 = Bits(dataWidth bits)
  val row2 = Bits(dataWidth bits)
  val row3 = Reg(Bits(dataWidth bits))
  when(io.pre.valid) { row3 := io.pre.payload.asBits }

  val ram2 = new ShiftRamDyn(dataWidth, lineLengthBits)
  ram2.io.CE := io.pre.valid
  ram2.io.D  := row3
  ram2.io.LINE_LENGTH := io.LINEWIDTH
  row2 := ram2.io.Q

  val ram1 = new ShiftRamDyn(dataWidth, lineLengthBits)
  ram1.io.CE := io.pre.valid
  ram1.io.D  := row2
  ram1.io.LINE_LENGTH := io.LINEWIDTH
  row1 := ram1.io.Q

  // two-cycle sync for de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)
  io.matrix_de := pre_de_r(1)

  // 3x3 shift registers per row
  val m11 = Reg(Bits(dataWidth bits)) init(0)
  val m12 = Reg(Bits(dataWidth bits)) init(0)
  val m13 = Reg(Bits(dataWidth bits)) init(0)
  val m21 = Reg(Bits(dataWidth bits)) init(0)
  val m22 = Reg(Bits(dataWidth bits)) init(0)
  val m23 = Reg(Bits(dataWidth bits)) init(0)
  val m31 = Reg(Bits(dataWidth bits)) init(0)
  val m32 = Reg(Bits(dataWidth bits)) init(0)
  val m33 = Reg(Bits(dataWidth bits)) init(0)
  when(io.pre.valid) {
    m11 := m12; m12 := m13; m13 := row1
    m21 := m22; m22 := m23; m23 := row2
    m31 := m32; m32 := m33; m33 := row3
  }
  io.matrix.m11 := m11.asSInt
  io.matrix.m12 := m12.asSInt
  io.matrix.m13 := m13.asSInt
  io.matrix.m21 := m21.asSInt
  io.matrix.m22 := m22.asSInt
  io.matrix.m23 := m23.asSInt
  io.matrix.m31 := m31.asSInt
  io.matrix.m32 := m32.asSInt
  io.matrix.m33 := m33.asSInt
}

class Matrix3x3(
  dataWidth: Int,
  lineLength: Int,
  padding: Int = 1,
  stride: Int = 1
) extends Component {
  var kernelSize = 3
  require(padding <= (kernelSize - 1 ) / 2, "padding must be less than (kernelSize - 1 ) / 2")

  val io = new Bundle {
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix_de = out Bool()
    val matrix = master(Matrix3x3Interface(dataWidth))
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True

  // row shift: row1 is the straight input, row1 and row2 come from two line buffers
  val row1 = Bits(dataWidth bits)
  val row2 = Bits(dataWidth bits)
  val row3 = Reg(Bits(dataWidth bits))
  row3 := io.pre.payload.asBits

  val ram2 = new ShiftRam(dataWidth, lineLength)
  ram2.io.CE := io.pre.valid
  ram2.io.D  := row3
  row2 := ram2.io.Q

  val ram1 = new ShiftRam(dataWidth, lineLength)
  ram1.io.CE := io.pre.valid
  ram1.io.D  := row2
  row1 := ram1.io.Q

  // two-cycle sync for de
  val pre_de_r = Reg(Bits(2 bits)) init(0)
  pre_de_r := (pre_de_r(0) ## io.pre.valid)
  val raw_matrix_de = pre_de_r(1)

  // Position tracking for padding and stride
  val pixelX = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val pixelY = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val lineCount = Reg(UInt(log2Up(lineLength) bits)) init(0)

  // Update position counters
  when(io.pre.valid && io.pre.ready) {
    when(pixelX === lineLength - 1) {
      pixelX := 0
      when(lineCount === lineLength - 1) {
        lineCount := 0
        pixelY := 0
      } otherwise {
        lineCount := lineCount + 1
        pixelY := pixelY + 1
      }
    } otherwise {
      pixelX := pixelX + 1
    }
  }

  // Convolution window center position (2 cycles delay to match matrix_de)
  val convX = Reg(UInt(log2Up(lineLength) bits)) init(0)
  val convY = Reg(UInt(log2Up(lineLength) bits)) init(0)
  when(raw_matrix_de) {
    // Convolution window center is at current pixel position minus 2 (for 3x3 kernel)
    convX := pixelX - 2
    convY := pixelY - 2
  }

  // Padding logic: check if convolution window center is within valid area
  val isPadding = Bool()
  isPadding := (convX > (kernelSize - padding * 2 + 1)) || (convY > (kernelSize - padding * 2 + 1))

  // Stride logic: only output every stride-th convolution
  val strideCounter = Reg(UInt(log2Up(stride) bits)) init(0)
  val strideValid = Bool()
  strideValid := (strideCounter === 0)

  when(raw_matrix_de) {
    if (stride > 1) {
      when(strideCounter === stride - 1) {
        strideCounter := 0
      } otherwise {
        strideCounter := strideCounter + 1
      }
    }
  }

  // Output control logic - only output when not padding and stride is valid
  val shouldOutput = Bool()
  shouldOutput := raw_matrix_de && !isPadding && strideValid
  io.matrix_de := shouldOutput

  // 3x3 shift registers per row
  val m11, m12, m13 = Reg(Bits(dataWidth bits)) init(0)
  val m21, m22, m23 = Reg(Bits(dataWidth bits)) init(0)
  val m31, m32, m33 = Reg(Bits(dataWidth bits)) init(0)
  when(io.pre.valid) {
    m11 := m12; m12 := m13; m13 := row1
    m21 := m22; m22 := m23; m23 := row2
    m31 := m32; m32 := m33; m33 := row3
  }
  io.matrix.m11 := m11.asSInt
  io.matrix.m12 := m12.asSInt
  io.matrix.m13 := m13.asSInt
  io.matrix.m21 := m21.asSInt
  io.matrix.m22 := m22.asSInt
  io.matrix.m23 := m23.asSInt
  io.matrix.m31 := m31.asSInt
  io.matrix.m32 := m32.asSInt
  io.matrix.m33 := m33.asSInt
}




// General Matrix Interface
case class MatrixInterface(dataWidth: Int, kernelSize: Int) extends Bundle with IMasterSlave {
  val m = Vec(Vec(SInt(dataWidth bits), kernelSize), kernelSize)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = MatrixInterface(dataWidth, kernelSize)

  def << (that: MatrixInterface): Unit = {
    for(i <- 0 until kernelSize; j <- 0 until kernelSize){
      this.m(i)(j) := that.m(i)(j)
    }
  }
}

// General dynamic matrix
class MatrixDyn(
  dataWidth: Int,
  lineLengthBits: Int,
  kernelSize: Int,
  padding: Int = 0,
  stride: Int = 1
) extends Component {
  val io = new Bundle {
    val LINEWIDTH = in UInt(lineLengthBits bits)
    val pre = slave(Stream(SInt(dataWidth bits)))
    val matrix_de = out Bool()
    val matrix = master(MatrixInterface(dataWidth, kernelSize))
  }

  // Ready signal - always ready to accept input when not processing or when bypassing
  io.pre.ready := True

  // A total of (kernelSize - 1) line buffers are required.
  val lineBuffers = Array.fill(kernelSize-1)(new ShiftRamDyn(dataWidth, lineLengthBits))

  // Row data stream, row(0) is the latest row, row(kernelSize-1) is the oldest row.
  val rows = Array.fill(kernelSize)(Reg(Bits(dataWidth bits)) init(0))

  // Write the input pixels sequentially into lineBuffer
  lineBuffers.zipWithIndex.foreach { case (ram, idx) =>
    ram.io.CE := io.pre.valid
    if(idx == 0) {
      ram.io.D := io.pre.payload.asBits
    } else {
      ram.io.D := rows(idx).asBits
    }
    ram.io.LINE_LENGTH := io.LINEWIDTH
    rows(idx+1) := ram.io.Q
  }
  rows(0) := io.pre.payload.asBits

  // Column shift register
  val shiftRegs = Array.fill(kernelSize, kernelSize)(Reg(Bits(dataWidth bits)) init(0))

  when(io.pre.valid){
    for(i <- 0 until kernelSize){
      for(j <- 0 until kernelSize){
        if(j == kernelSize-1){
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
  val strideValid = Bool()
  strideValid := (strideCounter === 0)

  when(raw_matrix_de){
    if(stride > 1){
      when(strideCounter === stride - 1){
        strideCounter := 0
      } otherwise {
        strideCounter := strideCounter + 1
      }
    }
  }

  io.matrix_de := raw_matrix_de && strideValid

  // Output control logic - only output when not padding and stride is valid
  for(i <- 0 until kernelSize){
    for(j <- 0 until kernelSize){
      io.matrix.m(i)(j) := shiftRegs(i)(j).asSInt
    }
  }
}

// object MatrixDynGen {
//   def main(args: Array[String]): Unit = {
//     println("Generating MatrixDyn modules...")

//     // 3x3 max pooling with stride 3
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new MatrixDyn(
//         dataWidth = 8,
//         lineLengthBits = log2Up(28),
//         kernelSize = 7,
//         padding = 0,
//         stride = 1)
//     ).printPruned()

//     println("MatrixDyn modules generated successfully!")
//   }
// }
