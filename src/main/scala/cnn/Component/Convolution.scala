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
case class KernelInterface(dataWidth: Int, kernelSize: Int) extends Bundle with IMasterSlave {
  val de = Bool()
  val k = Vec(Vec(SInt(dataWidth bits), kernelSize), kernelSize)

  override def asMaster(): Unit = this.asOutput()
  override def asSlave(): Unit = this.asInput()
  override def clone = KernelInterface(dataWidth, kernelSize)

  def <<(that: KernelInterface): Unit = {
    this.de := that.de
    for(i <- 0 until kernelSize; j <- 0 until kernelSize){
      this.k(i)(j) := that.k(i)(j)
    }
  }
}

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
    val matrix = slave(MatrixInterface(dataWidth, kernelSize))
    val kernel = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
  }

  // kernel dynamic streaming
  val kernelRegs = Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize * kernelSize)
  val idx = Reg(UInt(log2Up(kernelSize * kernelSize) bits)) init(0)
  when(io.kernel.valid) {
    kernelRegs(idx) := io.kernel.payload
    idx := idx + 1
  }
  io.kernel.ready := (idx < kernelSize * kernelSize)

  // Convolution
  val pixels = (0 until kernelSize).flatMap(i => (0 until kernelSize).map(j => io.matrix.m(i)(j)))
  val acc = (pixels, kernelRegs).zipped.map(_ * _).reduce(_ + _).resize(32)
  val shifted = (acc >> kernelShift)
  val convolution = shifted.resize(convWidth)

  // Output
  io.post.valid := RegNext(io.matrix.de)
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

  // Matrix
  val m = new Matrix(MatrixConfig(dataWidth, lineLength, kernelSize, padding, stride, lineLengthDyn))
  if (lineLengthDyn) { m.io.linewidth := io.linewidth }
  m.io.pre <> io.pre
  // Convolution
  val c = new Convolution(config)
  c.io.matrix <> m.io.matrix
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

  // StreamMap: Split kernel input into each channel
  val streamMap = new StreamMap(StreamMapConfig(dataWidth, Seq.fill(channelNum)(kernelSize * kernelSize)))
  streamMap.io.kernelIn <> io.kernel

  // Matrix per channel
  val m = (0 until channelNum).map { ch =>
    val mat = new Matrix(MatrixConfig(dataWidth, lineLength, kernelSize, padding, stride, lineLengthDyn))
    if (lineLengthDyn) mat.io.linewidth := io.linewidth
    mat.io.pre.valid := io.pre.valid
    mat.io.pre.payload := io.pre.payload(ch)
    mat
  }
  io.pre.ready := m.map(_.io.pre.ready).reduce(_ && _)

  // Convolution per channel
  val c = (0 until channelNum).map { ch =>
    val conv = new Convolution(config)
    conv.io.matrix <> m(ch).io.matrix
    conv.io.kernel <> streamMap.io.kernelOut(ch)
    conv
  }

  // Merge each channel output
  val merged = Stream(SInt(convWidth bits))
  merged.valid := c.map(_.io.post.valid).reduce(_ || _)
  merged.payload := c.map(_.io.post.payload).reduce(_ + _)
  c.zipWithIndex.foreach { case (conv, idx) => conv.io.post.ready := io.post.ready }
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
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(convConfig.dataWidth bits)))
    val post = master(Stream(Vec(SInt(convConfig.convWidth bits), convNum)))
    val kernel = slave(Stream(SInt(convConfig.dataWidth bits)))
    val linewidth = if (convConfig.lineLengthDyn) in UInt(log2Up((convConfig.lineLength - 1)) bits) else null
  }

  // ---------------- Matrix 共用 ----------------
  val m = new Matrix(MatrixConfig(
    convConfig.dataWidth,
    convConfig.lineLength,
    convConfig.kernelSize,
    convConfig.padding,
    convConfig.stride,
    convConfig.lineLengthDyn
  ))
  m.io.pre <> io.pre
  if (convConfig.lineLengthDyn) { m.io.linewidth := io.linewidth }

  // ---------------- Kernel streaming ----------------
  // 每个卷积核一份寄存器
  val kernelRegs = Vec(
    Vec(Reg(SInt(convConfig.dataWidth bits)) init(0), convConfig.kernelSize * convConfig.kernelSize),
    convNum
  )
  // 每个卷积核的 index
  val idxs = Vec(Reg(UInt(log2Up(convConfig.kernelSize * convConfig.kernelSize) bits)) init(0), convNum)

  // 内部 FSM 控制：依次装载 convNum 个 kernel
  val kernelSel = Reg(UInt(log2Up(convNum) bits)) init(0)
  when(io.kernel.valid) {
    kernelRegs(kernelSel)(idxs(kernelSel)) := io.kernel.payload
    idxs(kernelSel) := idxs(kernelSel) + 1
    when(idxs(kernelSel) === (convConfig.kernelSize*convConfig.kernelSize - 1)) {
      // 完成一个 kernel 的加载 -> 切换下一个
      kernelSel := kernelSel + 1
    }
  }
  io.kernel.ready := (kernelSel < convNum)

  // ---------------- Convolution ----------------
  val pixels = (0 until convConfig.kernelSize).flatMap(i =>
    (0 until convConfig.kernelSize).map(j => m.io.matrix.m(i)(j))
  )

  // 为每个卷积核做卷积计算
  val convResults = Vec(SInt(convConfig.convWidth bits), convNum)
  for (k <- 0 until convNum) {
    val acc = (pixels, kernelRegs(k)).zipped.map(_ * _).reduce(_ + _).resize(32)
    val shifted = (acc >> convConfig.kernelShift)
    convResults(k) := shifted.resize(convConfig.convWidth)
  }

  // ---------------- Output ----------------
  io.post.valid := RegNext(m.io.matrix.de)
  io.post.payload := convResults
}

class Conv2DLayerMultiChannel(layerCfg: Conv2DLayerConfig) extends Component {
  import layerCfg._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(convConfig.dataWidth bits), convConfig.channelNum)))
    val post = master(Stream(Vec(SInt(convConfig.convWidth bits), convNum)))
    val kernel = slave(Stream(SInt(convConfig.dataWidth bits)))
    val linewidth = if (convConfig.lineLengthDyn) in UInt(log2Up((convConfig.lineLength - 1)) bits) else null
  }

  // ---------------- Matrix per channel ----------------
  val m = (0 until convConfig.channelNum).map { ch =>
    val mat = new Matrix(MatrixConfig(
      convConfig.dataWidth,
      convConfig.lineLength,
      convConfig.kernelSize,
      convConfig.padding,
      convConfig.stride,
      convConfig.lineLengthDyn
    ))
    mat.io.pre.valid   := io.pre.valid
    mat.io.pre.payload := io.pre.payload(ch)
    if (convConfig.lineLengthDyn) { mat.io.linewidth := io.linewidth }
    mat
  }
  io.pre.ready := m.map(_.io.pre.ready).reduce(_ && _)

  // ---------------- Kernel registers ----------------
  // 维度：convNum × channelNum × (kernelSize*kernelSize)
  val kernelRegs = Vec(
    Vec(
      Vec(Reg(SInt(convConfig.dataWidth bits)) init(0), convConfig.kernelSize * convConfig.kernelSize),
      convConfig.channelNum
    ),
    convNum
  )

  // 动态 kernel 流加载
  val idx = Reg(UInt(log2Up(convConfig.kernelSize * convConfig.kernelSize * convConfig.channelNum * convNum) bits)) init(0)
  when(io.kernel.valid) {
    val conv  = (idx / (convConfig.channelNum * convConfig.kernelSize * convConfig.kernelSize)).resized
    val ch    = ((idx / (convConfig.kernelSize * convConfig.kernelSize)) % convConfig.channelNum).resized
    val pos   = (idx % (convConfig.kernelSize * convConfig.kernelSize)).resized
    kernelRegs(conv)(ch)(pos) := io.kernel.payload
    idx := idx + 1
  }
  io.kernel.ready := (idx < convNum * convConfig.channelNum * convConfig.kernelSize * convConfig.kernelSize)

  // ---------------- Convolution ----------------
  // 对每个卷积核
  val convResults = Vec(SInt(convConfig.convWidth bits), convNum)
  for (k <- 0 until convNum) {
    // 每个卷积核累加所有通道的结果
    val channelResults = for (ch <- 0 until convConfig.channelNum) yield {
      val pixels = (0 until convConfig.kernelSize).flatMap(i =>
        (0 until convConfig.kernelSize).map(j => m(ch).io.matrix.m(i)(j))
      )
      (pixels, kernelRegs(k)(ch)).zipped.map(_ * _).reduce(_ + _)
    }
    val acc = channelResults.reduce(_ + _).resize(32)
    val shifted = (acc >> convConfig.kernelShift)
    convResults(k) := shifted.resize(convConfig.convWidth)
  }

  // ---------------- Output ----------------
  io.post.valid := RegNext(m.map(_.io.matrix.de).reduce(_ && _))
  io.post.payload := convResults
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
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2DLayer(config)
//     // ).printPruned()
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DLayerMultiChannel(config)
//     ).printPruned()
//   }
// }
