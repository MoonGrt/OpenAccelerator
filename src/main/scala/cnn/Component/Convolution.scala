package cnn

import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* ------------------------------- Convolution ------------------------------- */
/* --------------------------------------------------------------------------- */

/**
 * Conv2D3x3 with Padding and Stride Support
 *
 * This module implements a 3x3 2D convolution with configurable padding and stride.
 *
 * Padding:
 * - padding = 0: No padding, output size = input_size - 2
 * - padding = 1: Same padding, output size = input_size (for 3x3 kernel)
 * - padding > 1: Extended padding, output size = input_size + 2*(padding-1)
 *
 * Stride:
 * - stride = 1: No stride, process every pixel
 * - stride = 2: Skip every other pixel, output size = input_size/2
 * - stride = n: Skip n-1 pixels, output size = input_size/n
 *
 * Output Size Formula:
 * output_size = (input_size - 2*padding) / stride + 1
 *
 * Example:
 * - Input: 8x8 image
 * - Padding: 1, Stride: 2
 * - Output: (8 - 2*1) / 2 + 1 = 4x4
 */
case class Conv2D3x3ParamsInterface(dataWidth: Int) extends Bundle with IMasterSlave {
  val k11 = SInt(dataWidth bits)
  val k12 = SInt(dataWidth bits)
  val k13 = SInt(dataWidth bits)
  val k21 = SInt(dataWidth bits)
  val k22 = SInt(dataWidth bits)
  val k23 = SInt(dataWidth bits)
  val k31 = SInt(dataWidth bits)
  val k32 = SInt(dataWidth bits)
  val k33 = SInt(dataWidth bits)

  override def asMaster() = this.asOutput()
  override def asSlave() = this.asInput()
  override def clone = new Conv2D3x3ParamsInterface(dataWidth)

  def << (that: Conv2D3x3ParamsInterface): Unit = {
    this.k11 := that.k11
    this.k12 := that.k12
    this.k13 := that.k13
    this.k21 := that.k21
    this.k22 := that.k22
    this.k23 := that.k23
    this.k31 := that.k31
    this.k32 := that.k32
    this.k33 := that.k33
  }
}

case class Conv2DConfig(
  dataWidth    : Int,             // bits per pixel
  convWidth    : Int,             // bits per convolution output
  lineLength   : Int,             // number of pixels per line
  kernel       : Seq[Int],        // elements of kernel (3x3, 5x5, etc.)
  kernelShift  : Int = 4,         // right shift to divide by kernel sum (e.g. 16 -> shift 4)
  kernelSize   : Int = 3,         // size of kernel (3x3, 5x5, etc.)
  insigned     : Boolean = true,  // input signed or unsigned
  absolute     : Boolean = false, // absolute value of convolution output
  leftresize   : Boolean = true,  // left shift to increase precision
  padding      : Int = 1,         // padding size (0=no padding, 1=same padding for (3x3, 5x5, etc) kernel)
  stride       : Int = 1          // stride size (1=no stride, 2=skip every other pixel, etc.)
)

class Conv2D3x3(config: Conv2DConfig) extends Component {
  import config._
  require(padding <= (kernelSize - 1 ) / 2, "padding must be less than (kernelSize - 1 ) / 2")

  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
  }

  // instantiate matrix3x3 module with padding and stride support
  val m = new Matrix3x3(dataWidth, lineLength, padding, stride)
  m.io.pre <> io.pre

  // convolution compute for each channel
  def conv(m: Matrix3x3, kernel: Seq[Int]) = {
    // compute sum = sum(kernel[i]*pixel[i])
    val convm = if (insigned) {
      Matrix3x3Interface(dataWidth)
    } else {
      Matrix3x3Interface(dataWidth + 1)
    }
    convm << m.io.matrix.resized
    val pixels = Seq(
      convm.m11, convm.m12, convm.m13,
      convm.m21, convm.m22, convm.m23,
      convm.m31, convm.m32, convm.m33
    )
    val acc = pixels.zip(kernel).map { case (p, k) => p * k }.reduce(_ + _).resize(32)
    // arithmetic right shift by kernelShift
    val shifted = (acc >> kernelShift)
    val clipped = shifted.resize(convWidth)
    clipped
  }
  val convolution = conv(m, kernel)

  // delay de by one cycle to match data latency and support EN bypass
  val matrix_de = Reg(Bool()) init(False)
  matrix_de := m.io.matrix_de

  // Stream output logic - Matrix3x3 already handles padding and stride
  io.post.valid := Mux(io.EN, matrix_de, io.pre.valid)
  io.post.payload := Mux(io.EN, convolution, io.pre.payload.resized)
}

class Conv2DDyn(config: Conv2DConfig) extends Component {
  import config._
  require(kernel.length == kernelSize * kernelSize, s"kernel length must match kernelSize*kernelSize")
  require(padding <= (kernelSize - 1) / 2, "padding must be less than (kernelSize - 1)/2")

  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
    val LINEWIDTH = in UInt(log2Up(lineLength) bits)
  }

  // instantiate dynamic matrix module
  val m = new MatrixDyn(dataWidth, log2Up(lineLength), kernelSize, padding, stride)
  m.io.pre <> io.pre
  m.io.LINEWIDTH := io.LINEWIDTH

  // convolution compute for each channel
  def conv(m: MatrixDyn, kernel: Seq[Int]) = {
    val convm = if (insigned) {
      MatrixInterface(dataWidth, kernelSize)
    } else {
      MatrixInterface(dataWidth + 1, kernelSize)
    }
    convm << m.io.matrix.resized
    val pixels = (0 until kernelSize).flatMap { i =>
      (0 until kernelSize).map { j =>
        convm.m(i)(j)
      }
    }
    val acc = pixels.zip(kernel).map { case (p, k) => p * k }.reduce(_ + _).resize(32)
    val shifted = (acc >> kernelShift)
    val clipped = shifted.resize(convWidth)
    // if(absolute) clipped.abs else clipped
    clipped
  }

  val convolution = conv(m, kernel)

  // delay de by one cycle to match data latency and support EN bypass
  val matrix_de = Reg(Bool()) init(False)
  matrix_de := m.io.matrix_de

  // Stream output logic
  io.post.valid := Mux(io.EN, matrix_de, io.pre.valid)
  io.post.payload := Mux(io.EN, convolution, io.pre.payload.resized)
}


// object Conv2DGen {
//   def main(args: Array[String]): Unit = {
//     val meanKernel = Seq(1,1,1, 1,1,1, 1,1,1) // Identity kernel: 1 1 1 / 1 1 1 / 1 1 1
//     val gaussianKernel = Seq(1,2,1, 2,4,2, 1,2,1) // Gaussian kernel: 1 2 1 / 2 4 2 / 1 2 1
//     val sharpenKernel = Seq(0,-1,0, -1,5,-1, 0,-1,0) // Sharpen kernel: 0 -1 0 / -1 5 -1 / 0 -1 0
//     val meanKernel5x5 = Seq(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1)
//     val gaussianKernel5x5 = Seq(1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1) // Gaussian kernel: 1 4 6 4 1 / 4 16 24 16 4 / 6 24 36 24 6 / 4 16 24 16 4 / 1 4 6 4 1
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2D3x3(Conv2DConfig(
//     //     dataWidth = 8,
//     //     convWidth = 10,
//     //     lineLength = 480,
//     //     kernel = gaussianKernel,
//     //     kernelShift = 4,
//     //     insigned = false,
//     //     padding = 1,
//     //     stride = 1))
//     // ).printPruned()
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DDyn(Conv2DConfig(
//         dataWidth = 8,
//         convWidth = 10,
//         lineLength = 28,
//         kernel = gaussianKernel5x5,
//         kernelShift = 4,
//         kernelSize = 5,
//         insigned = false,
//         padding = 1,
//         stride = 1))
//     ).printPruned()
//   }
// }
