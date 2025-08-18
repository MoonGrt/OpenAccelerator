package cnn

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
  dataWidth     : Int,            // bits per pixel
  convWidth     : Int,            // bits per convolution output
  lineLength    : Int,            // number of pixels per line
  kernelSize    : Int = 3,        // size of kernel (3x3, 5x5, etc.)
  kernel        : Seq[Int] = Seq(1,2,1, 2,4,2, 1,2,1),
  kernelShift   : Int = 4,        // right shift to divide by kernel sum (e.g. 16 -> shift 4)
  padding       : Int = 1,        // padding size (0=no padding, 1=same padding for (3x3, 5x5, etc) kernel)
  stride        : Int = 1,        // stride size (1=no stride, 2=skip every other pixel, etc.)
  lineLengthDyn : Boolean = true, // dynamic line length
  kernelDyn     : Boolean = true, // dynamic kernel size
  insigned      : Boolean = true, // input signed or unsigned
  leftresize    : Boolean = true  // left shift to increase precision
) {
  require(padding <= (kernelSize - 1) / 2, "padding must be less than (kernelSize - 1)/2")
}

// Conv2D Configuration
class Conv2D(config: Conv2DConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
    val kernel = if (kernelDyn) slave(KernelInterface(dataWidth, kernelSize)) else null
  }

  // Matrix
  val m = new Matrix(MatrixConfig(dataWidth, lineLength, kernelSize, padding, stride, lineLengthDyn))
  m.io.pre <> io.pre
  if (lineLengthDyn) { m.io.linewidth := io.linewidth }

  val pixels = (0 until kernelSize).flatMap(i => (0 until kernelSize).map(j => m.io.matrix.m(i)(j)))
  val kernelRegs = Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize * kernelSize)
  if (kernelDyn) {
    // ========== kernel bus interface ==========
    val bus = io.kernel
    for(i <- 0 until kernelSize; j <- 0 until kernelSize) {
      kernelRegs(i*kernelSize + j) := bus.k(i)(j)
    }
  } else {
    // ========== static kernel ==========
    for ((k, idx) <- kernel.zipWithIndex) {
      kernelRegs(idx) := S(k, dataWidth bits)
    }
  }

  // Convolution
  val acc = (pixels, kernelRegs).zipped.map(_ * _).reduce(_ + _).resize(32)
  val shifted = (acc >> kernelShift)
  val convolution = shifted.resize(convWidth)

  // Output
  io.post.valid := io.EN ? RegNext(m.io.matrix.de) | io.pre.valid
  io.post.payload := io.EN ? convolution | io.pre.payload.resized
}

class Conv2DStream(config: Conv2DConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(convWidth bits)))
    val linewidth = if (lineLengthDyn) in UInt(log2Up((lineLength - 1)) bits) else null
    val kernel = if (kernelDyn) slave(Stream(SInt(dataWidth bits))) else null
  }

  // Matrix
  val m = new Matrix(MatrixConfig(dataWidth, lineLength, kernelSize, padding, stride, lineLengthDyn))
  m.io.pre <> io.pre
  if (lineLengthDyn) { m.io.linewidth := io.linewidth }

  val pixels = (0 until kernelSize).flatMap(i => (0 until kernelSize).map(j => m.io.matrix.m(i)(j)))
  val kernelRegs = Vec(Reg(SInt(dataWidth bits)) init(0), kernelSize * kernelSize)
  if (kernelDyn) {
    // ========== kernel dynamic streaming ==========
    val idx = Reg(UInt(log2Up(kernelSize * kernelSize) bits)) init(0)
    val stream = io.kernel
    when(stream.valid) {
      kernelRegs(idx) := stream.payload
      idx := idx + 1
    }
    stream.ready := True
  } else {
    // ========== static kernel ==========
    for ((k, idx) <- kernel.zipWithIndex) {
      kernelRegs(idx) := S(k, dataWidth bits)
    }
  }

  // Convolution
  val acc = (pixels, kernelRegs).zipped.map(_ * _).reduce(_ + _).resize(32)
  val shifted = (acc >> kernelShift)
  val convolution = shifted.resize(convWidth)

  // Output
  io.post.valid := io.EN ? RegNext(m.io.matrix.de) | io.pre.valid
  io.post.payload := io.EN ? convolution | io.pre.payload.resized
}


/* --------------------------------------------------------------------------- */
/* ---------------------------- Convolution Layer ---------------------------- */
/* --------------------------------------------------------------------------- */
case class Conv2DLayerConfig(
  convNum: Int,
  convConfig: Conv2DConfig
)

class Conv2DLayerStream(layerCfg: Conv2DLayerConfig) extends Component {
  import layerCfg._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(SInt(convConfig.dataWidth bits)))
    val post = master(Stream(Vec(SInt(convConfig.convWidth bits), convNum)))
    val linewidth = if (convConfig.lineLengthDyn) in UInt(log2Up((convConfig.lineLength - 1)) bits) else null
    val kernel = if (convConfig.kernelDyn) slave(Stream(SInt(convConfig.dataWidth bits))) else null
  }

  // Multiple Conv2DStream
  val convs = Array.fill(convNum)(new Conv2DStream(convConfig))
  for (i <- 0 until convNum) {
    convs(i).io.EN := io.EN
    convs(i).io.pre.payload := io.pre.payload
    convs(i).io.pre.valid := io.pre.valid
    convs(i).io.post.ready := io.post.ready
    if (convConfig.lineLengthDyn) {
      convs(i).io.linewidth := io.linewidth
    }
  }

  // Kernel distributor
  if (convConfig.kernelDyn) {
    val totalKernelSize = convConfig.kernelSize * convConfig.kernelSize
    val idx = Reg(UInt(log2Up(totalKernelSize) bits)) init(0)
    val kId = Reg(UInt(log2Up(convNum) bits)) init(0)
    // One-hot select the currently active convolution kernel
    for (i <- 0 until convNum) {
      convs(i).io.kernel.valid   := io.kernel.valid && (kId === i)
      convs(i).io.kernel.payload := io.kernel.payload
    }
    io.kernel.ready := convs.map(c => c.io.kernel.ready && (kId === convs.indexOf(c))).reduce(_ || _)
    // Control distribution order
    when(io.kernel.fire) {
      idx := idx + 1
      when(idx === (totalKernelSize - 1)) {
        idx := 0
        kId := kId + 1
      }
    }
  }
  // Output
  io.pre.ready := convs.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := convs.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until convNum) {
    io.post.payload(i) := convs(i).io.post.payload
  }
}

class Conv2DLayerStreamMultiIn(layerCfg: Conv2DLayerConfig) extends Component {
  import layerCfg._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(convConfig.dataWidth bits), convNum)))
    val post = master(Stream(Vec(SInt(convConfig.convWidth bits), convNum)))
    val linewidth = if (convConfig.lineLengthDyn) in UInt(log2Up((convConfig.lineLength - 1)) bits) else null
    val kernel = if (convConfig.kernelDyn) slave(Stream(SInt(convConfig.dataWidth bits))) else null
  }

  // Multiple Conv2DStream
  val convs = Array.fill(convNum)(new Conv2DStream(convConfig))
  // Dynamic lineWidth
  for (i <- 0 until convNum) {
    convs(i).io.EN := io.EN
    convs(i).io.pre.payload := io.pre.payload(i)
    convs(i).io.pre.valid := io.pre.valid
    convs(i).io.post.ready := io.post.ready
    if (convConfig.lineLengthDyn) {
      convs(i).io.linewidth := io.linewidth
    }
  }

  // kernel distributor
  if (convConfig.kernelDyn) {
    val totalKernelSize = convConfig.kernelSize * convConfig.kernelSize
    val idx = Reg(UInt(log2Up(totalKernelSize) bits)) init(0)
    val kId = Reg(UInt(log2Up(convNum) bits)) init(0)
    // One-hot select the currently active convolution kernel
    for (i <- 0 until convNum) {
      convs(i).io.kernel.valid   := io.kernel.valid && (kId === i)
      convs(i).io.kernel.payload := io.kernel.payload
    }
    io.kernel.ready := convs.map(c => c.io.kernel.ready && (kId === convs.indexOf(c))).reduce(_ || _)
    // Control distribution order
    when(io.kernel.fire) {
      idx := idx + 1
      when(idx === (totalKernelSize - 1)) {
        idx := 0
        kId := kId + 1
      }
    }
  }
  // Output
  io.pre.ready := convs.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := convs.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until convNum) {
    io.post.payload(i) := convs(i).io.post.payload
  }
}


/* --------------------------------------------------------------------------- */
/* -------------------------- Convolution Layer Map -------------------------- */
/* --------------------------------------------------------------------------- */
case class Conv2DLayerStreamMapConfig(
  dataWidth: Int,
  layerKernelSize: Seq[Int] // 每层需要的kernel元素数量
)

class Conv2DLayerStreamMap(mapCfg: Conv2DLayerStreamMapConfig) extends Component {
  import mapCfg._
  val io = new Bundle {
    val kernelIn = slave(Stream(SInt(dataWidth bits)))
    val kernelOut = Vec(master(Stream(SInt(dataWidth bits))), layerKernelSize.length)
  }

  val layerKernelSizesHW = Vec(layerKernelSize.map(x => U(x, 32 bits)))
  val layerIdx = Reg(UInt(log2Up(layerKernelSize.length) bits)) init(0)
  val cnt = Reg(UInt(32 bits)) init(0)

  // Data routing
  io.kernelIn.ready := io.kernelOut(layerIdx).ready // Enter ready to default to the current layer's ready
  for (i <- 0 until layerKernelSize.length) { // Default each output invalid
    io.kernelOut(i).valid := False
    io.kernelOut(i).payload := io.kernelIn.payload
  }
  when(io.kernelIn.valid && io.kernelIn.ready) {
    io.kernelOut(layerIdx).valid := True
    cnt := cnt + 1
    when(cnt === (layerKernelSizesHW(layerIdx) - 1)) {
      cnt := 0
      when(layerIdx === (layerKernelSizesHW.length - 1)) {
        layerIdx := 0
      } otherwise {
        layerIdx := layerIdx + 1
      }
    }
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object Conv2DGen {
//   def main(args: Array[String]): Unit = {
//     val meanKernel = Seq(1,1,1, 1,1,1, 1,1,1) // Identity kernel: 1 1 1 / 1 1 1 / 1 1 1
//     val gaussianKernel = Seq(1,2,1, 2,4,2, 1,2,1) // Gaussian kernel: 1 2 1 / 2 4 2 / 1 2 1
//     val sharpenKernel = Seq(0,-1,0, -1,5,-1, 0,-1,0) // Sharpen kernel: 0 -1 0 / -1 5 -1 / 0 -1 0
//     val meanKernel5x5 = Seq(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1)
//     val gaussianKernel5x5 = Seq(1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1) // Gaussian kernel: 1 4 6 4 1 / 4 16 24 16 4 / 6 24 36 24 6 / 4 16 24 16 4 / 1 4 6 4 1
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2D(Conv2DConfig(
//     //     dataWidth = 8,
//     //     convWidth = 10,
//     //     lineLength = 28,
//     //     kernelSize = 5,
//     //     kernel = gaussianKernel5x5,
//     //     kernelShift = 4,
//     //     lineLengthDyn = true,
//     //     kernelDyn = true,
//     //     padding = 1,
//     //     stride = 1))
//     // ).printPruned()
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DStream(Conv2DConfig(
//         dataWidth = 8,
//         convWidth = 10,
//         lineLength = 28,
//         kernelSize = 5,
//         kernelShift = 4,
//         lineLengthDyn = true,
//         kernelDyn = true,
//         padding = 1,
//         stride = 1))
//     ).printPruned()
//   }
// }

// object Conv2DLayerStreamGen {
//   def main(args: Array[String]): Unit = {
//     val config = Conv2DLayerConfig(
//       convNum = 6,
//       convConfig = Conv2DConfig(
//         dataWidth = 8,
//         convWidth = 10,
//         lineLength = 28,
//         kernelSize = 5,
//         kernelShift = 4,
//         lineLengthDyn = true,
//         kernelDyn = true,
//         padding = 1,
//         stride = 1))
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new Conv2DLayerStream(config)
//     // ).printPruned()
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DLayerStreamMultiIn(config)
//     ).printPruned()
//   }
// }

// object Conv2DLayerStreamMapGen {
//   def main(args: Array[String]): Unit = {
//     val kernelMapCfg = Conv2DLayerStreamMapConfig(
//       dataWidth = 8,
//       layerKernelSize = Seq(
//         6 * 5 * 5,
//         12 * 5 * 5
//       )
//     )
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Conv2DLayerStreamMap(kernelMapCfg)
//     )
//   }
// }
