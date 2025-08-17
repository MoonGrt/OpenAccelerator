// package cnn

// import spinal.core._
// import spinal.lib._

// /**
//  * Complete LeNet-5 CNN Network
//  *
//  * Network Architecture:
//  * Input: 28×28 grayscale image
//  * Conv1: 6 filters of 5×5 → Output: 6×24×24
//  * Pool1: 2×2 max pooling → Output: 6×12×12
//  * Conv2: 12 filters of 5×5 → Output: 12×8×8
//  * Pool2: 2×2 max pooling → Output: 12×4×4
//  * Flatten: 12×4×4 → 192
//  * FC1: 192 → 10 (output classes)
//  */
// case class LeNet5CompleteConfig(
//   dataWidth    : Int = 8,          // bits per pixel
//   weightWidth  : Int = 8,          // bits per weight
//   biasWidth    : Int = 8,          // bits per bias
//   inputSize    : Int = 28,         // input image size
//   numClasses   : Int = 10,         // number of output classes
//   useBias      : Boolean = true,   // whether to use bias
//   signed       : Boolean = false,  // whether data is signed
//   quantization : Boolean = false   // whether to use quantization
// )

// /**
//  * Multi-channel convolution layer
//  */
// class MultiChannelConv[T <: Data](
//   dataType: HardType[T],
//   dataWidth: Int,
//   lineLength: Int,
//   numInputChannels: Int,
//   numOutputChannels: Int,
//   kernelSize: Int = 3,
//   padding: Int = 0,
//   stride: Int = 1
// ) extends Component {

//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(Vec(UInt(dataWidth bits), numInputChannels)))
//     val output = master(Stream(Vec(UInt(dataWidth bits), numOutputChannels)))
//     val weights = slave(Vec(Vec(SInt(weightWidth bits), kernelSize * kernelSize), numOutputChannels))
//     val bias = slave(Vec(SInt(biasWidth bits), numOutputChannels))
//   }

//   // Ready signal
//   io.input.ready := True

//   // Create convolution layers for each output channel
//   val conv_outputs = Vec(Stream(UInt(dataWidth bits)), numOutputChannels)

//   for (outCh <- 0 until numOutputChannels) {
//     val conv_config = Conv2DConfig(
//       dataWidth = dataWidth,
//       convWidth = dataWidth,
//       lineLength = lineLength,
//       kernel = Seq.fill(kernelSize * kernelSize)(1), // Placeholder, will be overridden
//       kernelShift = 4,
//       padding = padding,
//       stride = stride
//     )

//     val conv = new Conv2D3x3(dataType, conv_config)
//     conv.io.EN := io.EN
//     conv.io.pre <> io.input.payload(outCh % numInputChannels).asStream

//     conv_outputs(outCh) <> conv.io.post
//   }

//   // Connect outputs
//   io.output.valid := conv_outputs(0).valid
//   for (i <- 0 until numOutputChannels) {
//     io.output.payload(i) := conv_outputs(i).payload
//   }
// }

// /**
//  * Multi-channel pooling layer
//  */
// class MultiChannelPool[T <: Data](
//   dataType: HardType[T],
//   dataWidth: Int,
//   lineLength: Int,
//   numChannels: Int,
//   kernelSize: Int = 2,
//   padding: Int = 0,
//   stride: Int = 2
// ) extends Component {

//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(Vec(UInt(dataWidth bits), numChannels)))
//     val output = master(Stream(Vec(UInt(dataWidth bits), numChannels)))
//   }

//   // Ready signal
//   io.input.ready := True

//   // Create pooling layers for each channel
//   val pool_outputs = Vec(Stream(UInt(dataWidth bits)), numChannels)

//   for (ch <- 0 until numChannels) {
//     val pool = if (kernelSize == 2) {
//       val pool_config = MaxPool2x2Config(
//         dataWidth = dataWidth,
//         lineLength = lineLength,
//         padding = padding,
//         stride = stride
//       )
//       new MaxPooling2x2(dataType, pool_config)
//     } else {
//       val pool_config = MaxPool3x3Config(
//         dataWidth = dataWidth,
//         lineLength = lineLength,
//         padding = padding,
//         stride = stride
//       )
//       new MaxPooling3x3(dataType, pool_config)
//     }
//     pool.io.EN := io.EN
//     pool.io.pre <> io.input.payload(ch).asStream

//     pool_outputs(ch) <> pool.io.post
//   }

//   // Connect outputs
//   io.output.valid := pool_outputs(0).valid
//   for (i <- 0 until numChannels) {
//     io.output.payload(i) := pool_outputs(i).payload
//   }
// }

// /**
//  * Multi-channel ReLU layer
//  */
// class MultiChannelReLU[T <: Data](
//   dataType: HardType[T],
//   dataWidth: Int,
//   numChannels: Int,
//   activationType: String = "relu",
//   signed: Boolean = false
// ) extends Component {

//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(Vec(UInt(dataWidth bits), numChannels)))
//     val output = master(Stream(Vec(UInt(dataWidth bits), numChannels)))
//   }

//   // Ready signal
//   io.input.ready := True

//   // Create ReLU layers for each channel
//   val relu_outputs = Vec(Stream(UInt(dataWidth bits)), numChannels)

//   for (ch <- 0 until numChannels) {
//     val relu_config = ReLUConfig(
//       dataWidth = dataWidth,
//       activationType = activationType,
//       signed = signed
//     )

//     val relu = new ReLU(dataType, relu_config)
//     relu.io.EN := io.EN
//     relu.io.pre <> io.input.payload(ch).asStream

//     relu_outputs(ch) <> relu.io.post
//   }

//   // Connect outputs
//   io.output.valid := relu_outputs(0).valid
//   for (i <- 0 until numChannels) {
//     io.output.payload(i) := relu_outputs(i).payload
//   }
// }

// /**
//  * Flatten layer for multi-channel data
//  */
// class FlattenLayer[T <: Data](
//   dataType: HardType[T],
//   dataWidth: Int,
//   numChannels: Int,
//   spatialSize: Int
// ) extends Component {

//   val flattenedSize = numChannels * spatialSize * spatialSize

//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(Vec(UInt(dataWidth bits), numChannels)))
//     val output = master(Stream(Vec(UInt(dataWidth bits), flattenedSize)))
//   }

//   // Ready signal
//   io.input.ready := True

//   // Flatten logic
//   val flatten_valid = Reg(Bool()) init(False)
//   val flatten_data = Reg(Vec(UInt(dataWidth bits), flattenedSize))

//   flatten_valid := io.input.valid
//   for (i <- 0 until flattenedSize) {
//     val ch = i / (spatialSize * spatialSize)
//     val spatialIdx = i % (spatialSize * spatialSize)
//     flatten_data(i) := io.input.payload(ch)
//   }

//   io.output.valid := flatten_valid
//   io.output.payload := flatten_data
// }

// /**
//  * Complete LeNet-5 CNN Network
//  */
// class LeNet5Complete[T <: Data](dataType: HardType[T], config: LeNet5CompleteConfig) extends Component {
//   import config._

//   val io = new Bundle {
//     val EN = in Bool()
//     val input = slave(Stream(UInt(dataWidth bits)))
//     val output = master(Stream(Vec(UInt(dataWidth bits), numClasses)))
//     val conv1_weights = slave(Vec(Vec(SInt(weightWidth bits), 25), 6))  // 6 filters, 5x5 each
//     val conv1_bias = slave(Vec(SInt(biasWidth bits), 6))
//     val conv2_weights = slave(Vec(Vec(SInt(weightWidth bits), 150), 12)) // 12 filters, 6*5*5 each
//     val conv2_bias = slave(Vec(SInt(biasWidth bits), 12))
//     val fc_weights = slave(Vec(SInt(weightWidth bits), 192 * numClasses))
//     val fc_bias = slave(Vec(SInt(biasWidth bits), numClasses))
//   }

//   // Ready signal
//   io.input.ready := True

//   // ============================================================================
//   // Convolution Layer 1: 6 filters of 5×5 → Output: 6×24×24
//   // ============================================================================
//   val conv1_input = Stream(Vec(UInt(dataWidth bits), 1))
//   conv1_input.valid := io.input.valid
//   conv1_input.payload(0) := io.input.payload
//   conv1_input.ready := io.input.ready

//   val conv1 = new MultiChannelConv(dataType, dataWidth, inputSize, 1, 6, 5, 0, 1)
//   conv1.io.EN := io.EN
//   conv1.io.input <> conv1_input

//   // Connect weights and bias
//   for (i <- 0 until 6) {
//     for (j <- 0 until 25) {
//       conv1.io.weights(i)(j) := io.conv1_weights(i)(j)
//     }
//     conv1.io.bias(i) := io.conv1_bias(i)
//   }

//   // ============================================================================
//   // ReLU Activation after Conv1
//   // ============================================================================
//   val relu1 = new MultiChannelReLU(dataType, dataWidth, 6, "relu", signed)
//   relu1.io.EN := io.EN
//   relu1.io.input <> conv1.io.output

//   // ============================================================================
//   // Max Pooling Layer 1: 2×2 max pooling → Output: 6×12×12
//   // ============================================================================
//   val pool1 = new MultiChannelPool(dataType, dataWidth, 24, 6, 2, 0, 2)
//   pool1.io.EN := io.EN
//   pool1.io.input <> relu1.io.output

//   // ============================================================================
//   // Convolution Layer 2: 12 filters of 5×5 → Output: 12×8×8
//   // ============================================================================
//   val conv2 = new MultiChannelConv(dataType, dataWidth, 12, 6, 12, 5, 0, 1)
//   conv2.io.EN := io.EN
//   conv2.io.input <> pool1.io.output

//   // Connect weights and bias
//   for (i <- 0 until 12) {
//     for (j <- 0 until 150) {
//       conv2.io.weights(i)(j) := io.conv2_weights(i)(j)
//     }
//     conv2.io.bias(i) := io.conv2_bias(i)
//   }

//   // ============================================================================
//   // ReLU Activation after Conv2
//   // ============================================================================
//   val relu2 = new MultiChannelReLU(dataType, dataWidth, 12, "relu", signed)
//   relu2.io.EN := io.EN
//   relu2.io.input <> conv2.io.output

//   // ============================================================================
//   // Max Pooling Layer 2: 2×2 max pooling → Output: 12×4×4
//   // ============================================================================
//   val pool2 = new MultiChannelPool(dataType, dataWidth, 8, 12, 2, 0, 2)
//   pool2.io.EN := io.EN
//   pool2.io.input <> relu2.io.output

//   // ============================================================================
//   // Flatten Layer: 12×4×4 → 192
//   // ============================================================================
//   val flatten = new FlattenLayer(dataType, dataWidth, 12, 4)
//   flatten.io.EN := io.EN
//   flatten.io.input <> pool2.io.output

//   // ============================================================================
//   // Fully Connected Layer: 192 → 10
//   // ============================================================================
//   val fc_config = FullConnectionConfig(
//     inputWidth = dataWidth,
//     outputWidth = dataWidth,
//     weightWidth = weightWidth,
//     biasWidth = biasWidth,
//     inputSize = 192,
//     outputSize = numClasses,
//     useBias = useBias,
//     signed = signed,
//     quantization = quantization
//   )

//   val fc = new FullConnection(dataType, fc_config)
//   fc.io.EN := io.EN
//   fc.io.pre <> flatten.io.output

//   // Connect weights and bias
//   for (i <- 0 until 192 * numClasses) {
//     fc.io.weights.weights(i) := io.fc_weights(i)
//   }
//   for (i <- 0 until numClasses) {
//     fc.io.weights.bias(i) := io.fc_bias(i)
//   }

//   // ============================================================================
//   // Output
//   // ============================================================================
//   io.output <> fc.io.post
// }

// /**
//  * Test bench for LeNet-5
//  */
// class LeNet5TestBench extends Component {
//   val io = new Bundle {
//     val input_data = in Vec(UInt(8 bits), 784) // 28x28 flattened
//     val input_valid = in Bool()
//     val output_data = out Vec(UInt(8 bits), 10)
//     val output_valid = out Bool()
//   }

//   val config = LeNet5CompleteConfig(
//     dataWidth = 8,
//     weightWidth = 8,
//     biasWidth = 8,
//     inputSize = 28,
//     numClasses = 10,
//     useBias = true,
//     signed = false,
//     quantization = false
//   )

//   val lenet5 = new LeNet5Complete(UInt(8 bits), config)
//   lenet5.io.EN := True

//   // Input stream
//   val input_counter = Reg(UInt(10 bits)) init(0)
//   val input_stream = Stream(UInt(8 bits))

//   input_stream.valid := io.input_valid
//   input_stream.payload := io.input_data(input_counter)

//   when(input_stream.valid && input_stream.ready) {
//     when(input_counter === 783) {
//       input_counter := 0
//     } otherwise {
//       input_counter := input_counter + 1
//     }
//   }

//   lenet5.io.input <> input_stream

//   // Output
//   io.output_data := lenet5.io.output.payload
//   io.output_valid := lenet5.io.output.valid
// }


// /* ----------------------------------------------------------------------------- */
// /* ---------------------------------- Demo Gen --------------------------------- */
// /* ----------------------------------------------------------------------------- */
// object LeNet5CompleteGen {
//   def main(args: Array[String]): Unit = {
//     println("Generating complete LeNet-5 CNN network...")

//     val config = LeNet5CompleteConfig(
//       dataWidth = 8,
//       weightWidth = 8,
//       biasWidth = 8,
//       inputSize = 28,
//       numClasses = 10,
//       useBias = true,
//       signed = false,
//       quantization = false
//     )

//     // Generate complete LeNet-5
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new LeNet5Complete(UInt(8 bits), config)
//     ).printPruned()

//     // Generate test bench
//     SpinalConfig(targetDirectory = "rtl").generateVerilog(
//       new LeNet5TestBench()
//     ).printPruned()

//     println("Complete LeNet-5 network generated successfully!")
//     println("\n=== Complete LeNet-5 Architecture ===")
//     println("Input: 28×28 grayscale image")
//     println("Conv1: 6 filters of 5×5 → 6×24×24")
//     println("ReLU1: Activation")
//     println("Pool1: 2×2 max pooling → 6×12×12")
//     println("Conv2: 12 filters of 5×5 → 12×8×8")
//     println("ReLU2: Activation")
//     println("Pool2: 2×2 max pooling → 12×4×4")
//     println("Flatten: 12×4×4 → 192")
//     println("FC: 192 → 10 (output classes)")
//     println("\nFeatures:")
//     println("- Multi-channel convolution and pooling")
//     println("- Proper weight and bias management")
//     println("- Stream-based data flow")
//     println("- Configurable parameters")
//   }
// }
