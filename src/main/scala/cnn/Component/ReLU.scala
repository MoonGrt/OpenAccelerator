package cnn

import spinal.core._
import spinal.lib._

/**
 * ReLU (Rectified Linear Unit) Activation Function
 *
 * This module implements various activation functions:
 * - ReLU: f(x) = max(0, x)
 * - Leaky ReLU: f(x) = max(αx, x) where α is a small positive number
 * - Parametric ReLU: f(x) = max(αx, x) where α is learnable
 * - ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
 */
case class ReLUConfig(
  dataWidth      : Int,             // bits per pixel
  activationType : String = "relu", // "relu", "leaky_relu", "parametric_relu", "elu"
  alpha          : Double = 0.01    // slope for negative values (leaky/parametric relu, elu)
)

/**
 * ReLU module
 */
class ReLU(config: ReLUConfig) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(SInt(dataWidth bits)))
    val post = master(Stream(SInt(dataWidth bits)))
  }

  // Ready signal
  io.pre.ready := True

  // Activation function computation
  def activate(input: SInt) = {
    val result = SInt(dataWidth bits)

    activationType match {
      case "relu" => {
        // For signed data, use comparison with zero
        result := Mux(input > 0, input, S(0, dataWidth bits))
      }
      case "leaky_relu" => {
        val alphaValue = U((alpha * ((1 << dataWidth) - 1)).toInt, dataWidth bits)
        val scaledInput = (input * alphaValue.asSInt) >> dataWidth
        result := Mux(input > 0, input, scaledInput)
      }
      case "parametric_relu" => { // TODO: input alphaReg signal
        // For parametric ReLU, alpha is typically learned and stored in a register
        val alphaValue = U((alpha * ((1 << dataWidth) - 1)).toInt, dataWidth bits)
        val alphaReg = Reg(SInt(dataWidth bits)) init(alphaValue.asSInt)
        val scaledInput = (input * alphaReg) >> dataWidth
        result := Mux(input > 0, input, scaledInput)
      }
      case "elu" => {
        val alphaValue = U((alpha * ((1 << dataWidth) - 1)).toInt, dataWidth bits)
        // ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
        // For hardware implementation, we approximate e^x - 1
        val isNegative = (input <= 0)
        val absInput = Mux(isNegative, -input, input)
        // Simple approximation: e^x ≈ 1 + x + x^2/2 for small x
        val expApprox = S(1, dataWidth bits) + absInput + (absInput * absInput >> 1)
        val eluResult = (expApprox - S(1, dataWidth bits)) * alphaValue.asSInt >> dataWidth
        result := Mux(isNegative, eluResult.resized, input)
      }
      case _ => {
        // Default to ReLU
        result := Mux(input > 0, input, S(0, dataWidth bits))
      }
    }
    result
  }

  // Apply activation function to input signal
  val activatedValue = activate(io.pre.payload)

  // Stream output logic
  io.post.valid := Mux(io.EN, io.pre.valid, io.pre.valid)
  io.post.payload := Mux(io.EN, activatedValue, io.pre.payload)
}

/**
 * Batch ReLU for processing multiple channels with only one ready and valid signal
 */
class BatchReLU(config: ReLUConfig, numChannels: Int) extends Component {
  import config._
  val io = new Bundle {
    val EN = in Bool()
    val pre = slave(Stream(Vec(SInt(dataWidth bits), numChannels)))
    val post = master(Stream(Vec(SInt(dataWidth bits), numChannels)))
  }

  // Ready signal
  io.pre.ready := True

  // Apply ReLU to each channel
  val activatedChannels = Vec(SInt(dataWidth bits), numChannels)
  for (i <- 0 until numChannels) {
    val relu = new ReLU(config)
    relu.io.EN := io.EN
    relu.io.pre.valid := io.pre.valid
    relu.io.pre.payload := io.pre.payload(i)
    activatedChannels(i) := relu.io.post.payload
  }

  // Stream output logic
  io.post.valid := Mux(io.EN, io.pre.valid, io.pre.valid)
  io.post.payload := Mux(io.EN, activatedChannels, io.pre.payload)
}


/* -------------------------------------------------------------------------- */
/* ------------------------------- ReLU Layer ------------------------------- */
/* -------------------------------------------------------------------------- */
case class ReLULayerConfig(
  reluNum: Int,
  reluConfig: ReLUConfig
)

class ReLULayer(layerCfg: ReLULayerConfig) extends Component {
  import layerCfg._
  val io = new Bundle {
    val EN   = in Bool()
    val pre  = slave(Stream(Vec(SInt(reluConfig.dataWidth bits), reluNum)))
    val post = master(Stream(Vec(SInt(reluConfig.dataWidth bits), reluNum)))
  }

  // Multiple ReLU
  val relus = Array.fill(reluNum)(new ReLU(reluConfig))
  // Dynamic lineWidth
  for (i <- 0 until reluNum) {
    relus(i).io.EN := io.EN
    relus(i).io.pre.payload := io.pre.payload(i)
    relus(i).io.pre.valid := io.pre.valid
    relus(i).io.post.ready := io.post.ready
  }

  // Output
  io.pre.ready := relus.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := relus.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until reluNum) {
    io.post.payload(i) := relus(i).io.post.payload
  }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object ReLUGen {
//   def main(args: Array[String]): Unit = {
//     // Standard ReLU
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new ReLU(ReLUConfig(
//         dataWidth = 8,
//         activationType = "relu"))
//     ).printPruned()
//     // // Leaky ReLU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     dataWidth = 8,
//     //     activationType = "leaky_relu",
//     //     alpha = 0.01))
//     // ).printPruned()
//     // // Parametric ReLU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     dataWidth = 8,
//     //     activationType = "parametric_relu",
//     //     alpha = 0.01))
//     // ).printPruned()
//     // // ELU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     dataWidth = 8,
//     //     activationType = "elu",
//     //     alpha = 1.0))
//     // ).printPruned()
//     // // Batch ReLU for 3 channels
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new BatchReLU(ReLUConfig(
//     //     dataWidth = 8,
//     //     activationType = "relu"), 3)
//     // ).printPruned()
//   }
// }

// object ReLULayerGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new ReLULayer(ReLULayerConfig(
//         reluNum = 3,
//         reluConfig = ReLUConfig(
//           dataWidth = 8,
//           activationType = "relu"))
//       )
//     ).printPruned()
//   }
// }
