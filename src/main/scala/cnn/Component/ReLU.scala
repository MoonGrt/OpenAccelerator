package cnn

import spinal.core._
import spinal.lib._

/**
 * ReLU (Rectified Linear Unit) Activation Function
 *
 * This module implements various activation functions:
 * - ReLU: f(x) = indataWidth(0, x)
 * - Leaky ReLU: f(x) = indataWidth(αx, x) where α is a small positive number
 * - Parametric ReLU: f(x) = indataWidth(αx, x) where α is learnable
 * - ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
 */
case class ReLUConfig(
  indataWidth    : Int = 8,         // bits per pixel
  outdataWidth   : Int = 8,         // bits per pixel
  shift          : Int = 0,         // shift value for fixed-point data
  activationType : String = "relu", // "relu", "leaky_relu", "parametric_relu", "elu"
  alpha          : Double = 0.01    // slope for negative values (leaky/parametric relu, elu)
)

/**
 * ReLU module
 */
class ReLU(config: ReLUConfig) extends Component {
  import config._
  val io = new Bundle {
    val pre = slave(Stream(SInt(indataWidth bits)))
    val post = master(Stream(SInt(outdataWidth bits)))
  }

  // Ready signal
  io.pre.ready := True

  // Activation function computation
  def activate(input: SInt) = {
    val result = SInt(indataWidth bits)

    activationType match {
      case "relu" => {
        // For signed data, use comparison with zero
        result := Mux(input > 0, input, S(0, indataWidth bits))
      }
      case "leaky_relu" => {
        val alphaValue = U((alpha * ((1 << indataWidth) - 1)).toInt, indataWidth bits)
        val scaledInput = (input * alphaValue.asSInt) >> indataWidth
        result := Mux(input > 0, input, scaledInput)
      }
      case "parametric_relu" => { // TODO: input alphaReg signal
        // For parametric ReLU, alpha is typically learned and stored in a register
        val alphaValue = U((alpha * ((1 << indataWidth) - 1)).toInt, indataWidth bits)
        val alphaReg = Reg(SInt(indataWidth bits)) init(alphaValue.asSInt)
        val scaledInput = (input * alphaReg) >> indataWidth
        result := Mux(input > 0, input, scaledInput)
      }
      case "elu" => {
        val alphaValue = U((alpha * ((1 << indataWidth) - 1)).toInt, indataWidth bits)
        // ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
        // For hardware implementation, we approximate e^x - 1
        val isNegative = (input <= 0)
        val absInput = Mux(isNegative, -input, input)
        // Simple approximation: e^x ≈ 1 + x + x^2/2 for small x
        val expApprox = S(1, indataWidth bits) + absInput + (absInput * absInput >> 1)
        val eluResult = (expApprox - S(1, indataWidth bits)) * alphaValue.asSInt >> indataWidth
        result := Mux(isNegative, eluResult.resized, input)
      }
      case _ => {
        // Default to ReLU
        result := Mux(input > 0, input, S(0, indataWidth bits))
      }
    }
    result
  }

  // Apply activation function to input signal
  val activatedValue = activate(io.pre.payload) >> shift

  // Stream output logic
  io.post.valid := io.pre.valid
  io.post.payload := activatedValue.resized
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
  import reluConfig._
  val io = new Bundle {
    val pre = slave(Stream(Vec(SInt(indataWidth bits), reluNum)))
    val post = master(Stream(Vec(SInt(outdataWidth bits), reluNum)))
  }

  // Multiple ReLU
  val relus = Array.fill(reluNum)(new ReLU(reluConfig))
  // Dynamic lineWidth
  for (i <- 0 until reluNum) {
    relus(i).io.pre.payload := io.pre.payload(i)
    relus(i).io.pre.valid := io.pre.valid
    relus(i).io.post.ready := io.post.ready
  }

  // Output
  io.pre.ready := relus.map(_.io.pre.ready).reduce(_ && _)
  io.post.valid := relus.map(_.io.post.valid).reduce(_ && _)
  for (i <- 0 until reluNum) { io.post.payload(i) := relus(i).io.post.payload }
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object ReLUGen {
//   def main(args: Array[String]): Unit = {
//     // Standard ReLU
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new ReLU(ReLUConfig(
//         indataWidth = 8,
//         outdataWidth = 8,
//         shift = 1,
//         activationType = "relu"))
//     ).printPruned()
//     // // Leaky ReLU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     indataWidth = 8,
//     //     outdataWidth = 8,
//     //     shift = 1,
//     //     activationType = "leaky_relu",
//     //     alpha = 0.01))
//     // ).printPruned()
//     // // Parametric ReLU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     indataWidth = 8,
//     //     outdataWidth = 8,
//     //     shift = 1,
//     //     activationType = "parametric_relu",
//     //     alpha = 0.01))
//     // ).printPruned()
//     // // ELU
//     // SpinalConfig(targetDirectory = "rtl").generateVerilog(
//     //   new ReLU(ReLUConfig(
//     //     indataWidth = 8,
//     //     outdataWidth = 8,
//     //     shift = 1,
//     //     activationType = "elu",
//     //     alpha = 1.0))
//     // ).printPruned()
//   }
// }

// object ReLULayerGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new ReLULayer(ReLULayerConfig(
//         reluNum = 3,
//         reluConfig = ReLUConfig(
//           indataWidth = 8,
//           outdataWidth = 8,
//           shift = 1,
//           activationType = "relu"))
//       )
//     ).printPruned()
//   }
// }
