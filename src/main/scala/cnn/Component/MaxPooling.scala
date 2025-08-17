package cnn

import spinal.core._
import spinal.lib._

/**
 * 2x2 MaxPooling with Padding and Stride Support
 *
 * This module implements 2D max pooling with 2x2 kernel, configurable padding and stride.
 *
 * Padding:
 * - padding = 0: No padding
 * - padding = 1: Same padding (for 2x2 kernel)
 *
 * Stride:
 * - stride = 1: No stride, process every pixel
 * - stride = 2: Skip every other pixel
 *
 * Output Size Formula:
 * output_size = (input_size - 2 + 2*padding) / stride + 1
 */
case class MaxPool2x2Config(
  dataWidth  : Int,     // bits per pixel
  lineLength : Int,     // number of pixels per line
  padding    : Int = 0, // padding size
  stride     : Int = 2  // stride size
)

/**
 * 2x2 MaxPooling module
 */
class MaxPooling2x2(config: MaxPool2x2Config) extends Component {
  import config._

  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(dataWidth bits)))
  }

  // Instantiate matrix builder
  val Matrix = new Matrix2x2(dataWidth, lineLength, padding, stride)
  Matrix.io.pre <> io.pre

  // Max pooling computation for 2x2
  def maxPool2x2(matrix: Matrix2x2Interface): SInt = {
    val values = Seq(matrix.m11, matrix.m12, matrix.m21, matrix.m22)
    values.reduce((a, b) => Mux(a > b, a, b))
  }
  val maxValue = maxPool2x2(Matrix.io.matrix).resized

  // Delay de by one cycle
  val matrix_de = Reg(Bool()) init(False)
  matrix_de := Matrix.io.matrix_de

  // Stream output logic
  io.post.valid := Mux(io.EN, matrix_de, io.pre.valid)
  io.post.payload := Mux(io.EN, maxValue, io.pre.payload)
}

// object MaxPool2x2Gen {
//   def main(args: Array[String]): Unit = {
//     println("Generating 2x2 MaxPooling modules...")

//     // 2x2 max pooling with stride 2
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new MaxPooling2x2(MaxPool2x2Config(
//         dataWidth = 8,
//         lineLength = 480,
//         padding = 0,
//         stride = 2))
//     ).printPruned()

//     println("2x2 MaxPooling modules generated successfully!")
//   }
// }


/**
 * 3x3 MaxPooling with Padding and Stride Support
 *
 * This module implements 2D max pooling with 3x3 kernel, configurable padding and stride.
 *
 * Padding:
 * - padding = 0: No padding
 * - padding = 1: Same padding (for 3x3 kernel)
 *
 * Stride:
 * - stride = 1: No stride, process every pixel
 * - stride = 2: Skip every other pixel
 *
 * Output Size Formula:
 * output_size = (input_size - 3 + 2*padding) / stride + 1
 */
case class MaxPool3x3Config(
  dataWidth  : Int,     // bits per pixel
  lineLength : Int,     // number of pixels per line
  padding    : Int = 0, // padding size
  stride     : Int = 3  // stride size
)

/**
 * 3x3 MaxPooling module
 */
class MaxPooling3x3(config: MaxPool3x3Config) extends Component {
  import config._

  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(dataWidth bits)))
  }

  // Instantiate matrix builder
  val Matrix = new Matrix3x3(dataWidth, lineLength, padding, stride)
  Matrix.io.pre <> io.pre

  // Max pooling computation for 3x3
  def maxPool3x3(matrix: Matrix3x3Interface): SInt = {
    val values = Seq(
      matrix.m11, matrix.m12, matrix.m13,
      matrix.m21, matrix.m22, matrix.m23,
      matrix.m31, matrix.m32, matrix.m33
    )
    values.reduce((a, b) => Mux(a > b, a, b))
  }
  val maxValue = maxPool3x3(Matrix.io.matrix).resized

  // Delay de by one cycle
  val matrix_de = Reg(Bool()) init(False)
  matrix_de := Matrix.io.matrix_de

  // Stream output logic
  io.post.valid := Mux(io.EN, matrix_de, io.pre.valid)
  io.post.payload := Mux(io.EN, maxValue, io.pre.payload)
}

// object MaxPool3x3Gen {
//   def main(args: Array[String]): Unit = {
//     println("Generating 3x3 MaxPooling modules...")

//     // 3x3 max pooling with stride 3
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new MaxPooling3x3(MaxPool3x3Config(
//         dataWidth = 8,
//         lineLength = 480,
//         padding = 0,
//         stride = 3))
//     ).printPruned()

//     println("3x3 MaxPooling modules generated successfully!")
//   }
// }
