package misc
import spinal.core._

class Max(dataWidth: Int, dataNum: Int, outValue: Boolean = false, outIndex: Boolean = false) extends Component {
  val io = new Bundle {
    val data  = in Vec(SInt(dataWidth bits), dataNum)
    val max = if (outValue) out SInt(dataWidth bits) else null
    val idx = if (outIndex) out UInt(log2Up(dataNum) bits) else null
  }

  val (maxVal, maxIdx) =
    (1 until dataNum).foldLeft((io.data(0), U(0, log2Up(dataNum) bits))) {
      case ((mv, mi), i) =>
        val take = io.data(i) > mv
        (Mux(take, io.data(i), mv),
         Mux(take, U(i, mi.getWidth bits), mi))
    }
  if (outValue) io.max := maxVal
  if (outIndex) io.idx := maxIdx
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
// object MaxGen {
//   def main(args: Array[String]): Unit = {
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new Max(8, 4, true, true)
//     ).printPruned()
//   }
// }
