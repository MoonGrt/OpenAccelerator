package misc

import spinal.core._
import spinal.lib._

/* -------------------------------------------------------------------------- */
/* ------------------------------- Stream Map ------------------------------- */
/* -------------------------------------------------------------------------- */
case class StreamMapConfig(
  dataWidth: Int,
  streamSize: Seq[Int]
)

class StreamMap(mapCfg: StreamMapConfig) extends Component {
  import mapCfg._
  val io = new Bundle {
    val kernelIn = slave(Stream(SInt(dataWidth bits)))
    val kernelOut = Vec(master(Stream(SInt(dataWidth bits))), streamSize.length)
  }

  val streamSizesHW = Vec(streamSize.map(x => U(x, 32 bits)))
  val layerIdx = Reg(UInt(log2Up(streamSize.length) bits)) init(0)
  val cnt = Reg(UInt(32 bits)) init(0)

  // Data routing
  io.kernelIn.ready := io.kernelOut(layerIdx).ready // Enter ready to default to the current layer's ready
  for (i <- 0 until streamSize.length) { // Default each output invalid
    io.kernelOut(i).valid := False
    io.kernelOut(i).payload := io.kernelIn.payload
  }
  when(io.kernelIn.valid && io.kernelIn.ready) {
    io.kernelOut(layerIdx).valid := True
    cnt := cnt + 1
    when(cnt === (streamSizesHW(layerIdx) - 1)) {
      cnt := 0
      when(layerIdx === (streamSizesHW.length - 1)) {
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
// object StreamMapGen {
//   def main(args: Array[String]): Unit = {
//     val kernelMapCfg = StreamMapConfig(
//       dataWidth = 8,
//       streamSize = Seq(
//         6 * 5 * 5,
//         12 * 5 * 5
//       )
//     )
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new StreamMap(kernelMapCfg)
//     )
//   }
// }
