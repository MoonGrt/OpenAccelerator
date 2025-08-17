package cnn

import spinal.core._
import spinal.lib._

/**
 * MaxPooling with Padding and Stride Support
 *
 * This module implements 2D max pooling with kernel, configurable padding and stride.
 *
 * Padding:
 * - padding = 0: No padding
 * - padding >= 1: Same padding
 *
 * Stride:
 * - stride = 1: No stride, process every pixel
 * - stride >= 2: Skip every other pixel
 *
 * Output Size Formula:
 * output_size = (input_size - 2 + 2 * padding) / stride + 1
 */
// MaxPool Configuration
case class MaxPoolConfig(
  dataWidth     : Int,           // bits per pixel
  lineLength    : Int,           // number of pixels per line
  kernelSize    : Int,           // kernel size
  padding       : Int = 0,       // padding size
  stride        : Int = 2,       // stride size
  lineLengthDyn : Boolean = true // dynamic line length
)

// MaxPool Component
class MaxPool(config: MaxPoolConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // Instantiate matrix builder
  val m = new Matrix(MatrixConfig(dataWidth, lineLength, kernelSize, padding, stride, lineLengthDyn))
  m.io.pre <> io.pre
  if (lineLengthDyn) { m.io.linewidth := io.linewidth }
  // Max pooling computation
  def maxPool(matrix: MatrixInterface): SInt = {
    val elems = for(i <- 0 until kernelSize; j <- 0 until kernelSize) yield matrix.m(i)(j)
    elems.reduceBalancedTree((a, b) => (a > b) ? a | b)
  }
  val maxValue = maxPool(m.io.matrix).resized

  // Stream output logic
  io.post.valid := Mux(io.EN, RegNext(m.io.matrix.de), io.pre.valid)
  io.post.payload := Mux(io.EN, maxValue, io.pre.payload)
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object MaxPoolGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new MaxPool(MaxPoolConfig(
//         dataWidth = 8,
//         lineLength = 24,
//         kernelSize = 2,
//         padding = 0,
//         stride = 2,
//         lineLengthDyn = true))
//     ).printPruned()
//   }
// }
