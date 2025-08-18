// systolic_spinal.scala
// SpinalHDL implementation of a parameterizable 2D output-stationary systolic array
// Includes: PE (MAC + forwarding), SystolicArray2D, Verilog generator and a simple SpinalSim testbench.

package cnn

import spinal.core._
import spinal.lib._
import spinal.sim._
import scala.util.Random

case class SystolicConfig(
  M: Int = 4,
  N: Int = 4,
  K: Int = 8,
  DW: Int = 8,
  WW: Int = 8,
  ACCW: Int = 32
)

// -----------------------------
// Processing Element (PE)
// Output-stationary: each PE holds an accumulator. A flows right, B flows down.
// -----------------------------
class PeMac(config: SystolicConfig) extends Component {
  val io = new Bundle {
    val clk    = in Bool() // kept for clarity; Spinal doesn't use explicit clk IO usually
    val rstn   = in Bool()
    val valid_in = in Bool()
    val clear  = in Bool()
    val a_in   = in SInt(config.DW bits)
    val b_in   = in SInt(config.WW bits)

    val valid_out = out Bool()
    val a_out  = out SInt(config.DW bits)
    val b_out  = out SInt(config.WW bits)
    val acc_out = out SInt(config.ACCW bits)
  }

  // registers
  val acc = Reg(SInt(config.ACCW bits)) init(0)
  val a_reg = Reg(SInt(config.DW bits)) init(0)
  val b_reg = Reg(SInt(config.WW bits)) init(0)
  val v_reg = Reg(Bool()) init(False)

  // forward the operands (registered)
  a_reg := io.a_in
  b_reg := io.b_in
  v_reg := io.valid_in

  io.a_out := a_reg
  io.b_out := b_reg
  io.valid_out := v_reg

  when(io.clear) {
    acc := 0
  } otherwise {
    when(io.valid_in) {
      // multiply and accumulate, extend/resize correctly
      val prod = (io.a_in * io.b_in).resize(config.ACCW bits)
      acc := (acc + prod).resize(config.ACCW bits)
    }
  }

  io.acc_out := acc
}

// -----------------------------
// 2D Systolic Array (Output-Stationary)
// - M rows, N cols of PEs
// - a_left: Vec(M) inputs injected at left boundary
// - b_top: Vec(N) inputs injected at top boundary
// - start_tile: clears accumulators and starts a K-cycle accumulation
// - valid_in pulses while K elements are being streamed
// -----------------------------
class SystolicArray2D(config: SystolicConfig) extends Component {
  val io = new Bundle {
    val clk = in Bool()
    val rstn = in Bool()

    val a_left = in Vec(SInt(config.DW bits), config.M)
    val b_top  = in Vec(SInt(config.WW bits), config.N)
    val valid_in = in Bool()
    val start_tile = in Bool()

    val busy = out Bool()
    val c_tile = out Vec(Vec(SInt(config.ACCW bits), config.N), config.M) // M x N
    val c_valid = out Bool()
  }

// instantiate PEs
val pes = Array.tabulate(config.M, config.N){ (i, j) => new PeMac(config) }

// wiring between PEs
for(i <- 0 until config.M) {
  for(j <- 0 until config.N) {
    val pe = pes(i)(j)

    pe.io.clk   := io.clk
    pe.io.rstn  := io.rstn

    val a_in = if(j == 0) io.a_left(i) else pes(i)(j-1).io.a_out
    val b_in = if(i == 0) io.b_top(j)  else pes(i-1)(j).io.b_out
    val v_in = if(i == 0 || j == 0) io.valid_in else pes(i-1)(j-1).io.valid_out

    pe.io.a_in     := a_in
    pe.io.b_in     := b_in
    pe.io.valid_in := v_in
    pe.io.clear    := io.start_tile

    io.c_tile(i)(j) := pe.io.acc_out
  }
}


  // busy / k counter
  val running = Reg(Bool()) init(False)
  val kCnt = Reg(UInt(log2Up(config.K + 1) bits)) init(0)
  io.c_valid := False

  when(!io.rstn) {
    running := False
    kCnt := 0
  } elsewhen(io.start_tile && !running) {
    running := True
    kCnt := 0
  } elsewhen(running && io.valid_in) {
    when(kCnt === (config.K - 1)) {
      running := False
      io.c_valid := True
      kCnt := 0
    } otherwise {
      kCnt := kCnt + 1
      io.c_valid := False
    }
  }

  io.busy := running
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// -----------------------------
// Generate Verilog and simple simulation
// -----------------------------
// object SystolicArrayGen {
//   def main(args: Array[String]): Unit = {
//     val cfg = SystolicConfig(M=4, N=4, K=8, DW=8, WW=8, ACCW=32)
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new SystolicArray2D(cfg)
//     )
//   }
// }

// // -----------------------------
// // Simple SpinalSim test: stream two small matrices and verify C = A*B (tile-wise)
// // This test uses output-stationary semantics: for each k we feed a column of B and a row of A
// // -----------------------------
// object RunSim {
//   def main(args: Array[String]): Unit = {
//     val cfg = SystolicConfig(M=3, N=3, K=5, DW=8, WW=8, ACCW=32)
//     val dut = new SystolicArray2D(cfg)

//     SimConfig.withWave.doSim(dut) { dutSim =>
//       // helper to poke vectors
//       def pokeVec(vec: Seq[SInt], target: Seq[UInt]) = {}

//       // random small test
//       val rand = new Random(42)
//       val A = Array.fill(cfg.M, cfg.K){ (rand.nextInt(1<< (cfg.DW-1)) - (1<<(cfg.DW-2))).toLong }
//       val B = Array.fill(cfg.K, cfg.N){ (rand.nextInt(1<< (cfg.WW-1)) - (1<<(cfg.WW-2))).toLong }

//       // compute golden result
//       val Cgold = Array.ofDim[Long](cfg.M, cfg.N)
//       for(i <- 0 until cfg.M; j <- 0 until cfg.N; k <- 0 until cfg.K) {
//         Cgold(i)(j) += A(i)(k) * B(k)(j)
//       }

//       // reset
//       dutSim.clockDomain.assertReset()
//       dutSim.clockDomain.waitSampling()
//       dutSim.clockDomain.deassertReset()
//       dutSim.clockDomain.waitSampling()

//       // start tile
//       dutSim.io.start_tile #= true
//       dutSim.io.valid_in #= false
//       dutSim.clockDomain.waitSampling()
//       dutSim.io.start_tile #= false

//       // feed K cycles: on each cycle send a_left (row elements for that k) and b_top (col elements for that k)
//       for(k <- 0 until cfg.K) {
//         // a_left: A[:, k]
//         for(i <- 0 until cfg.M) dutSim.io.a_left(i) #= A(i)(k)
//         // b_top: B[k, :]
//         for(j <- 0 until cfg.N) dutSim.io.b_top(j) #= B(k)(j)

//         dutSim.io.valid_in #= true
//         dutSim.clockDomain.waitSampling()
//       }

//       // finish streaming (valid low)
//       dutSim.io.valid_in #= false

//       // wait a few cycles for pipeline to drain
//       for(_ <- 0 until (cfg.M + cfg.N + 2)) dutSim.clockDomain.waitSampling()

//       // read out c_tile and compare
//       for(i <- 0 until cfg.M) {
//         for(j <- 0 until cfg.N) {
//           val got = dutSim.io.c_tile(i)(j).toLong
//           val expect = Cgold(i)(j)
//           println(s"C[$i][$j] = $got (expect $expect)")
//           assert(got == expect, s"Mismatch at ($i,$j): got $got expect $expect")
//         }
//       }

//       println("Simulation passed: results match golden reference.")
//     }
//   }
// }
