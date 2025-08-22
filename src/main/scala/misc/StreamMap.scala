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
  require(streamSize.length > 1, "StreamMap requires at least two output streams")
  val io = new Bundle {
    val streamIn = slave(Stream(SInt(dataWidth bits)))
    val streamOut = Vec(master(Stream(SInt(dataWidth bits))), streamSize.length)
  }

  // Registers
  val streamSizes = Vec(streamSize.map(x => U(x, 32 bits)))
  val streamIdx = Reg(UInt(log2Up(streamSize.length) bits)) init(0)
  val streamCnt = Reg(UInt(32 bits)) init(0)
  // Data routing
  io.streamIn.ready := io.streamOut(streamIdx).ready // Enter ready to default to the current stream's ready
  for (i <- 0 until streamSize.length) { // Default each output invalid
    io.streamOut(i).valid := False
    io.streamOut(i).payload := io.streamIn.payload
  }
  when(io.streamIn.valid && io.streamIn.ready) {
    io.streamOut(streamIdx).valid := True
    streamCnt := streamCnt + 1
    when(streamCnt === (streamSizes(streamIdx) - 1)) {
      streamCnt := 0
      when(streamIdx === (streamSizes.length - 1)) {
        streamIdx := 0
      } otherwise {
        streamIdx := streamIdx + 1
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
