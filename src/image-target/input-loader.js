/*

The provided file, input-loader.js, is a JavaScript module that defines a class called InputLoader. This class is responsible for loading and processing input data for a machine learning model using the TensorFlow.js library.

Here's a breakdown of what the file does:

Imports the TensorFlow.js library using the import statement.
Defines the InputLoader class, which takes in the width and height of the input data as constructor parameters.
Initializes properties such as width, height, texShape, context, and program within the InputLoader class.
The buildProgram method is defined within the InputLoader class. This method constructs a WebGL shader program that will be used to process the input data. It uses the width and height parameters to calculate the UV coordinates and applies a grayscale transformation to the input data.
The _compileAndRun method is defined within the InputLoader class. This method compiles and runs the WebGL shader program using the provided program and inputs. It returns a TensorFlow.js tensor object representing the output of the program.
The loadInput method is defined within the InputLoader class. This method takes an input parameter, which is expected to be an instance of HTMLVideoElement or HTMLImageElement. It prepares the input data by resizing and rotating it if necessary, and then uploads the pixel data to a WebGL texture using the TensorFlow.js backend. Finally, it calls the _compileAndRun method to process the input data using the defined shader program and returns the processed output as a TensorFlow.js tensor object.
The InputLoader class is exported from the module, making it available for use in other parts of the codebase.
Overall, this file provides a more efficient implementation for loading and processing input data for a machine learning model using TensorFlow.js. It utilizes WebGL shaders to perform operations on the input data, such as resizing, rotating, and converting to grayscale.The provided file, input-loader.js, is a JavaScript module that defines a class called InputLoader. This class is responsible for loading and processing input data for a machine learning model using the TensorFlow.js library.

*/

import * as tf from '@tensorflow/tfjs';

// More efficient implementation for tf.browser.fromPixels
//   original implementation: /node_modules/@tensorflow/tfjs-backend-webgl/src/kernels/FromPixels.ts
//
// This implementation return grey scale instead of RGBA in the orignal implementation

class InputLoader {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.texShape = [height, width];

    const context = document.createElement('canvas').getContext('2d');
    context.canvas.width = width;
    context.canvas.height = height;
    this.context = context;

    this.program = this.buildProgram(width, height);

    const backend = tf.backend();
    //this.tempPixelHandle = backend.makeTensorInfo(this.texShape, 'int32');
    this.tempPixelHandle = backend.makeTensorInfo(this.texShape, 'float32');
    // warning!!!
    // usage type should be TextureUsage.PIXELS, but tfjs didn't export this enum type, so we hard-coded 2 here
    //   i.e. backend.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
    backend.texData.get(this.tempPixelHandle.dataId).usage = 2;
  }

  // old method
  _loadInput(input) {
    return tf.tidy(() => {
      let inputImage = tf.browser.fromPixels(input);
      inputImage = inputImage.mean(2);
      return inputImage;
    });
  }

  // input is instance of HTMLVideoElement or HTMLImageElement
  loadInput(input) {
    const context = this.context;
    context.clearRect(
      0,
      0,
      this.context.canvas.width,
      this.context.canvas.height
    );

    const isInputRotated =
      input.width === this.height && input.height === this.width;
    if (isInputRotated) {
      // rotate 90 degree and draw
      let x = this.context.canvas.width / 2;
      let y = this.context.canvas.height / 2;
      let angleInDegrees = 90;

      context.save(); // save the current context state
      context.translate(x, y); // move the context origin to the center of the image
      context.rotate((angleInDegrees * Math.PI) / 180); // rotate the context

      // draw the image with its center at the origin
      context.drawImage(input, -input.width / 2, -input.height / 2);
      context.restore(); // restore the context to its original state
    } else {
      this.context.drawImage(input, 0, 0, input.width, input.height);
    }

    const backend = tf.backend();
    backend.gpgpu.uploadPixelDataToTexture(
      backend.getTexture(this.tempPixelHandle.dataId),
      this.context.canvas
    );

    //const res = backend.compileAndRun(this.program, [this.tempPixelHandle]);
    const res = this._compileAndRun(this.program, [this.tempPixelHandle]);
    //const res = this._runWebGLProgram(this.program, [this.tempPixelHandle], 'float32');
    //backend.disposeData(tempPixelHandle.dataId);
    return res;
  }

  buildProgram(width, height) {
    const textureMethod =
      tf.env().getNumber('WEBGL_VERSION') === 2 ? 'texture' : 'texture2D';

    const program = {
      variableNames: ['A'],
      outputShape: this.texShape,
      userCode: `
	void main() {
	  ivec2 coords = getOutputCoords();
	  int texR = coords[0];
	  int texC = coords[1];
	  vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${width}.0, ${height}.0);

	  vec4 values = ${textureMethod}(A, uv);
	  setOutput((0.299 * values.r + 0.587 * values.g + 0.114 * values.b) * 255.0);
	}
      `,
    };
    return program;
  }

  _compileAndRun(program, inputs) {
    const outInfo = tf.backend().compileAndRun(program, inputs);
    return tf
      .engine()
      .makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
  }

  _runWebGLProgram(program, inputs, outputType) {
    const outInfo = tf.backend().runWebGLProgram(program, inputs, outputType);
    return tf
      .engine()
      .makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
  }
}

export { InputLoader };
