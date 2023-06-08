const cocoSsd = require('@tensorflow-models/coco-ssd');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;

// Load the Coco SSD model and image.
Promise.all([cocoSsd.load(), fs.readFile('image.png')])
.then((results) => {
  // COCO-SSD model object presents in the first element of results.
  const model = results[0];
  // Image buffer presents in the second element of results.
  const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
  // Call detect() to run inference.
  return model.detect(imgTensor);
})
.then((predictions) => {
  console.log('predictions: ', JSON.stringify(predictions, null, 2));
  console.log('\n Detected Objects',predictions.map(prediction => prediction.class));
});