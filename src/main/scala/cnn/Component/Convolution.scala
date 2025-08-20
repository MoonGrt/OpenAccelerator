package cnn

import misc._
import spinal.core._
import spinal.lib._

/* --------------------------------------------------------------------------- */
/* ------------------------------- Convolution ------------------------------- */
/* --------------------------------------------------------------------------- */
/**
 * Conv2D with Padding and Stride Support
 *
 * This module implements a 2D convolution with configurable padding and stride.
 *
 * Padding:
 * - padding = 0: No padding, output size = input_size - 2
 * - padding = 1: Same padding, output size = input_size
 * - padding > 1: Extended padding, output size = input_size + 2*(padding-1)
 *
 * Stride:
 * - stride = 1: No stride, process every pixel
 * - stride = 2: Skip every other pixel, output size = input_size/2
 * - stride = n: Skip n-1 pixels, output size = input_size/n
 *
 * Output Size Formula:
 * output_size = (input_size - 2 * padding) / stride + 1
 */
// Conv2D Configuration
case class Conv2DConfig(
  channelNum    : Int = 1,        // number of input channels
  dataWidth     : Int = 8,        // bits per pixel
  convWidth     : Int = 8,        // bits per convolution output
  padding       : Int = 1,        // padding size (0=no padding, 1=same padding for (3x3, 5x5, etc) kernel)
  stride        : Int = 1,        // stride size (1=no stride, 2=skip every other pixel, etc.)
  lineLengthDyn : Boolean = true, // dynamic line length
  lineLength    : Int = 24,       // number of pixels per line
  kernelSize    : Int = 3,        // size of kernel (3x3, 5x5, etc.)
  kernelShift   : Int = 4,        // right shift to divide by kernel sum (e.g. 16 -> shift 4)
  kernel        : Seq[Int] = Seq(1,2,1, 2,4,2, 1,2,1)
) {
  require(padding <= (kernelSize - 1) / 2, "padding must be less than (kernelSize - 1)/2")
}

class Convolution(config: Conv2DConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val column = slave(ShiftColumnInterface(dataWidth, kernelSize))
    val post = master(Stream(SInt(convWidth bits)))
  }

  // --- Matrix ---
  val matrix = new Matrix(MatrixConfig(dataWidth, kernelSize, padding, stride))
  matrix.io.column << io.column
  // --- Kernel ---
  val kernelRegs = Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize * kernelSize)
  val idx = Reg(UInt(log2Up(kernelSize * kernelSize) bits)) init(0)
  when(io.kernel.valid) {
    kernelRegs(idx) := io.kernel.payload
    idx := idx + 1
  }
  io.kernel.ready := (idx < kernelSize * kernelSize)
  // ---Convolution ---
  val pixels = (0 until kernelSize).flatMap(i => (0 until kernelSize).map(j => matrix.io.matrix.m(i)(j)))
  val acc = (pixels, kernelRegs).zipped.map(_ * _).reduce(_ + _).resize(32)
  val shifted = (acc >> kernelShift)
  val convolution = shifted.resize(convWidth)
  // ---Output ---
  io.post.valid := RegNext(matrix.io.matrix.de)
  io.post.payload := convolution
}

class Conv2D(config: Conv2DConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // --- ShiftColumn ---
  val col = new ShiftColumn(ShiftColumnConfig(dataWidth, kernelSize, lineLengthDyn, lineLength))
  if (lineLengthDyn) { col.io.linewidth := io.linewidth }
  col.io.pre <> io.pre
  // --- Convolution ---
  val c = new Convolution(config)
  c.io.EN := io.EN
  c.io.column << col.io.column
  c.io.kernel <> io.kernel
  c.io.post <> io.post
}

class Conv2DMultiChannel(config: Conv2DConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(dataWidth bits), channelNum)))
    val post = master(Stream(SInt(convWidth bits)))
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // --- StreamMap ---
  val streamMap = new StreamMap(StreamMapConfig(dataWidth, Seq.fill(channelNum)(kernelSize * kernelSize)))
  streamMap.io.kernelIn <> io.kernel
  // --- ShiftColumn --- 
  val cols = (0 until channelNum).map { ch =>
    val col = new ShiftColumn(ShiftColumnConfig(dataWidth, kernelSize, lineLengthDyn, lineLength))
    if (lineLengthDyn) { col.io.linewidth := io.linewidth }
    col.io.pre.valid := io.pre.valid
    col.io.pre.payload := io.pre.payload(ch)
    col
  }
  io.pre.ready := cols.map(_.io.pre.ready).reduce(_ && _)
  // --- Convolution ---
  val convs = (0 until channelNum).map { ch =>
    val conv = new Convolution(config)
    conv.io.EN := io.EN
    conv.io.column <> cols(ch).io.column
    conv.io.kernel <> streamMap.io.kernelOut(ch)
    conv
  }
  // ---Output ---
  val merged = Stream(SInt(convWidth bits))
  merged.valid := convs.map(_.io.post.valid).reduce(_ || _)
  merged.payload := convs.map(_.io.post.payload).reduce(_ + _)
  convs.zipWithIndex.foreach { case (conv, idx) => conv.io.post.ready := io.post.ready }
  io.post <> merged
}


/* --------------------------------------------------------------------------- */
/* ---------------------------- Convolution Layer ---------------------------- */
/* --------------------------------------------------------------------------- */
case class Conv2DLayerConfig(
  convNum: Int,
  convConfig: Conv2DConfig
)

class Conv2DLayer(layerCfg: Conv2DLayerConfig) extends Component {
  import layerCfg._
  import convConfig._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(Vec(SInt(convWidth bits), convNum)))
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // --- StreamMap ---
  val streamMap = new StreamMap(StreamMapConfig(dataWidth, Seq.fill(convNum)(kernelSize * kernelSize)))
  streamMap.io.kernelIn <> io.kernel
  // --- ShiftColumn ---
  val col = new ShiftColumn(ShiftColumnConfig(dataWidth, kernelSize, lineLengthDyn, lineLength))
  if (lineLengthDyn) { col.io.linewidth := io.linewidth }
  col.io.pre <> io.pre
  // --- Convolution ---
  val convs = (0 until convNum).map { ch =>
    val conv = new Convolution(convConfig)
    conv.io.column <> col.io.column
    conv.io.kernel <> streamMap.io.kernelOut(ch)
    conv
  }
  // ---Output ---
  io.post.valid := convs.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until convNum) {
    io.post.payload(i) := convs(i).io.post.payload
  }
}

class Conv2DLayerMultiChannel(layerCfg: Conv2DLayerConfig) extends Component {
  import layerCfg._
  import convConfig._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(dataWidth bits), channelNum)))
    val post = master(Stream(Vec(SInt(convWidth bits), convNum)))
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
  }

  // --- StreamMap ---
  val streamMap = new StreamMap(StreamMapConfig(dataWidth, Seq.fill(convNum)(kernelSize * kernelSize * channelNum)))
  streamMap.io.kernelIn <> io.kernel
  // --- Conv2DLayer ---
  val convLayers = (0 until channelNum).map { ch =>
    val conv = new Conv2DLayer(layerCfg)
    conv.io.EN <> io.EN
    if (lineLengthDyn) { conv.io.linewidth := io.linewidth }
    conv.io.pre.valid := io.pre.valid
    conv.io.pre.payload := io.pre.payload(ch)
    conv.io.kernel <> streamMap.io.kernelOut(ch)
    conv
  }
  // --- Output ---
  io.post.valid := convLayers.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until convNum) {
    io.post.payload(i) := convLayers.map(_.io.post.payload(i)).reduce(_ + _)
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object Conv2DGen {
//   def main(args: Array[String]): Unit = {
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2D(Conv2DConfig(
//     //     dataWidth = 8,
//     //     convWidth = 10,
//     //     padding = 1,
//     //     stride = 1,
//     //     lineLengthDyn = true,
//     //     kernelSize = 5))
//     // ).printPruned()
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DMultiChannel(Conv2DConfig(
//         channelNum = 3,
//         dataWidth = 8,
//         convWidth = 10,
//         padding = 1,
//         stride = 1,
//         lineLengthDyn = true,
//         kernelSize = 5))
//     ).printPruned()
//   }
// }

// object Conv2DLayerGen {
//   def main(args: Array[String]): Unit = {
//     val config = Conv2DLayerConfig(
//       convNum = 6,
//       convConfig = Conv2DConfig(
//         channelNum = 3,
//         dataWidth = 8,
//         convWidth = 10,
//         padding = 1,
//         stride = 1,
//         lineLengthDyn = true,
//         kernelSize = 5))
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DLayer(config)
//     ).printPruned()
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2DLayerMultiChannel(config)
//     // ).printPruned()
//   }
// }
