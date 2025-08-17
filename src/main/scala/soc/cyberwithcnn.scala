package soc

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb._
import spinal.lib.bus.amba4.axi._
import spinal.lib.com.uart.{UartCtrlGenerics}
import spinal.lib.com.jtag.Jtag
import spinal.lib.cpu.riscv.impl.Utils.BR
import spinal.lib.cpu.riscv.impl.build.RiscvAxi4
import spinal.lib.cpu.riscv.impl.extension.{
  BarrelShifterFullExtension,
  DivExtension,
  MulExtension
}
import spinal.lib.cpu.riscv.impl._
import spinal.lib.io.{TriStateArray, InOutWrapper}
import spinal.lib.system.debugger.{JtagAxi4SharedDebugger, SystemDebuggerConfig}
import spinal.lib.misc.{HexTools, InterruptCtrl, Timer, Prescaler}


case class cyberConfig(
    axiFrequency: HertzNumber,
    onChipRamSize: BigInt,
    onChipRamHexFile: String,
    cpu: RiscvCoreConfig,
    iCache: InstructionCacheConfig
)

object cyberConfig {
  def default = {
    val config = cyberConfig(
      axiFrequency = 100 MHz,
      onChipRamSize = 4 KiB,
      onChipRamHexFile = null,
      cpu = RiscvCoreConfig(
        pcWidth = 32,
        addrWidth = 32,
        startAddress = 0x00000000,
        regFileReadyKind = sync,
        branchPrediction = dynamic,
        bypassExecute0 = true,
        bypassExecute1 = true,
        bypassWriteBack = true,
        bypassWriteBackBuffer = true,
        collapseBubble = false,
        fastFetchCmdPcCalculation = true,
        dynamicBranchPredictorCacheSizeLog2 = 7
      ),
      iCache = InstructionCacheConfig(
        cacheSize = 4096,
        bytePerLine = 32,
        wayCount = 1, // Can only be one for the moment
        wrappedMemAccess = true,
        addressWidth = 32,
        cpuDataWidth = 32,
        memDataWidth = 32
      )
    )
    // The CPU has a systems of plugin which allow to add new feature into the core.
    // Those extension are not directly implemented into the core, but are kind of additive logic patch defined in a separated area.
    config.cpu.add(new MulExtension)
    config.cpu.add(new DivExtension)
    config.cpu.add(new BarrelShifterFullExtension)

    config
  }
}

class cyber(config: cyberConfig) extends Component {
  def this(axiFrequency: HertzNumber) {
    this(cyberConfig.default.copy(axiFrequency = axiFrequency))
  }

  import config._
  val debug = true
  val interruptCount = 16

  val io = new Bundle {
    // Clocks / reset
    val clk = in Bool ()
    val rstn = in Bool ()
    // Main components IO
    val jtag = slave(Jtag())
  }

  val resetCtrlClockDomain = ClockDomain(
    clock = io.clk,
    config = ClockDomainConfig(
      resetKind = BOOT
    )
  )

  val resetCtrl = new ClockingArea(resetCtrlClockDomain) {
    val axiResetUnbuffered = False
    val coreResetUnbuffered = False
    // Implement an counter to keep the reset axiResetOrder high 64 cycles
    // Also this counter will automaticly do a reset when the system boot.
    val axiResetCounter = Reg(UInt(6 bits)) init (0)
    when(axiResetCounter =/= U(axiResetCounter.range -> true)) {
      axiResetCounter := axiResetCounter + 1
      axiResetUnbuffered := True
    }
    when(BufferCC(~io.rstn)) {
      axiResetCounter := 0
    }
    // When an axiResetOrder happen, the core reset will as well
    when(axiResetUnbuffered) {
      coreResetUnbuffered := True
    }
    // Create all reset used later in the design
    val axiReset = RegNext(axiResetUnbuffered)
    val coreReset = RegNext(coreResetUnbuffered)
  }

  val axiClockDomain = ClockDomain(
    clock = io.clk,
    reset = resetCtrl.axiReset,
    frequency = FixedFrequency(axiFrequency)
  )

  val coreClockDomain = ClockDomain(
    clock = io.clk,
    reset = resetCtrl.coreReset
  )

  val jtagClockDomain = ClockDomain(
    clock = io.jtag.tck
  )

  val axi = new ClockingArea(axiClockDomain) {
    val core = coreClockDomain {
      new RiscvAxi4(
        coreConfig = config.cpu,
        // iCacheConfig = config.iCache, // 目前上板有问题，仿真正常
        iCacheConfig = null,
        dCacheConfig = null,
        debug = debug,
        interruptCount = interruptCount
      )
    }

    val ram = Axi4SharedOnChipRam(
      dataWidth = 32,
      byteCount = onChipRamSize,
      idWidth = 4
    )
    HexTools.initRam(ram.ram, "test/cybewithcnn/build/mem/demo.bin", 0x80000000l)

    val jtagCtrl = JtagAxi4SharedDebugger(
      SystemDebuggerConfig(
        memAddressWidth = 32,
        memDataWidth = 32,
        remoteCmdWidth = 1
      )
    )

    val apbBridge = Axi4SharedToApb3Bridge(
      addressWidth = 20,
      dataWidth = 32,
      idWidth = 4
    )

    val axiCrossbar = Axi4CrossbarFactory()

    axiCrossbar.addSlaves(
      ram.io.axi -> (0x00000000L, onChipRamSize),
      apbBridge.io.axi -> (0xf0000000L, 1 MiB)
    )

    axiCrossbar.addConnections(
      core.io.i -> List(ram.io.axi),
      core.io.d -> List(ram.io.axi, apbBridge.io.axi),
      jtagCtrl.io.axi -> List(ram.io.axi, apbBridge.io.axi)
    )

    axiCrossbar.addPipelining(apbBridge.io.axi)((crossbar, bridge) => {
      crossbar.sharedCmd.halfPipe() >> bridge.sharedCmd
      crossbar.writeData.halfPipe() >> bridge.writeData
      crossbar.writeRsp << bridge.writeRsp
      crossbar.readRsp << bridge.readRsp
    })

    axiCrossbar.build()

    val apbDecoder = Apb3Decoder(
      master = apbBridge.io.apb,
      slaves = List(
        // cnnCtrl.io.apb -> (0x00000, 64 KiB),
        core.io.debugBus -> (0xf0000, 64 KiB)
      )
    )

    if (interruptCount != 0) {
      core.io.interrupt := (
        (default -> false)
      )
    }

    if (debug) {
      core.io.debugResetIn := resetCtrl.axiReset
      resetCtrl.coreResetUnbuffered setWhen (core.io.debugResetOut)
    }
  }
  io.jtag <> axi.jtagCtrl.io.jtag
}


/* ----------------------------------------------------------------------------- */
/* ---------------------------------- Demo Gen --------------------------------- */
/* ----------------------------------------------------------------------------- */
object cyber {
  def main(args: Array[String]) {
    val config =
      SpinalConfig(verbose = true, targetDirectory = "rtl").dumpWave()
    val report = config.generateVerilog(
      InOutWrapper(
        new cyber(
          cyberConfig.default.copy(
            onChipRamSize = 32 kB,
            onChipRamHexFile = "test/cybewithcnn/build/mem/demo.bin"
          )
        )
      )
    )
  }
}
