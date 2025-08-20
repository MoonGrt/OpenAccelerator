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
  dataWidth     : Int,            // bits per pixel
  kernelSize    : Int,            // kernel size
  padding       : Int = 0,        // padding size
  stride        : Int = 2,        // stride size
  lineLengthDyn : Boolean = true, // dynamic line length
  lineLength    : Int = 8         // number of pixels per line
)

// MaxPool Component
class MaxPool(config: MaxPoolConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // --- ShiftColumn ---
  val shiftCol = new ShiftColumn(ShiftColumnConfig(
    dataWidth     = dataWidth,
    kernelSize    = kernelSize,
    lineLength    = lineLength,
    lineLengthDyn = lineLengthDyn
  ))
  shiftCol.io.pre <> io.pre
  if (lineLengthDyn) {
    shiftCol.io.linewidth := io.linewidth
  }

  // --- Matrix ---
  val matrix = new Matrix(MatrixConfig(
    dataWidth  = dataWidth,
    kernelSize = kernelSize,
    padding    = padding,
    stride     = stride
  ))
  matrix.io.column << shiftCol.io.column

  // --- Max pooling ---
  def maxPool(matrix: MatrixInterface): SInt = {
    val elems = for(i <- 0 until kernelSize; j <- 0 until kernelSize) yield matrix.m(i)(j)
    elems.reduceBalancedTree((a, b) => (a > b) ? a | b)
  }
  val maxValue = maxPool(matrix.io.matrix).resized

  // --- 输出 ---
  io.post.valid   := matrix.io.matrix.de && io.EN
  io.post.payload := maxValue
}


/* --------------------------------------------------------------------------- */
/* ------------------------------ MaxPool Layer ------------------------------ */
/* --------------------------------------------------------------------------- */
case class MaxPoolLayerConfig(
  maxpoolNum: Int,
  maxpoolConfig: MaxPoolConfig
)

class MaxPoolLayer(layerCfg: MaxPoolLayerConfig) extends Component {
  import layerCfg._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(maxpoolConfig.dataWidth bits), maxpoolNum)))
    val post = master(Stream(Vec(SInt(maxpoolConfig.dataWidth bits), maxpoolNum)))
    val linewidth = if (maxpoolConfig.lineLengthDyn) in UInt(log2Up((maxpoolConfig.lineLength - 1)) bits) else null
  }

  // Multiple MaxPool
  val maxpools = Array.fill(maxpoolNum)(new MaxPool(maxpoolConfig))
  for (i <- 0 until maxpoolNum) {
    maxpools(i).io.EN := io.EN
    maxpools(i).io.pre.payload := io.pre.payload(i)
    maxpools(i).io.pre.valid := io.pre.valid
    maxpools(i).io.post.ready := io.post.ready
    if (maxpoolConfig.lineLengthDyn) {
      maxpools(i).io.linewidth := io.linewidth
    }
  }

  // Output
  io.pre.ready := maxpools.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := maxpools.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until maxpoolNum) {
    io.post.payload(i) := maxpools(i).io.post.payload
  }
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

// object MaxPoolLayerGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new MaxPoolLayer(MaxPoolLayerConfig(
//         maxpoolNum = 2,
//         maxpoolConfig = MaxPoolConfig(
//           dataWidth = 8,
//           lineLength = 24,
//           kernelSize = 2,
//           padding = 0,
//           stride = 2,
//           lineLengthDyn = true))
//       )
//     ).printPruned()
//   }
// }
