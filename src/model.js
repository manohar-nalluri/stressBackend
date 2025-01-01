import ort from 'onnxruntime-node';
import path from 'path'
import { fileURLToPath } from 'url';
import fs from 'fs'
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
function normalize(hr, eda) {
  return [
    (eda - 0) / (6 - 0),
    (hr - 50) / (180 - 50),
  ];
}
async function predictEdaHr({hr,eda}) {
  try {
    const modelPath = path.resolve(__dirname, '../physio_model.onnx');
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at ${modelPath}`);
    }
    console.log('data of hr eda',hr,eda)
    const session = await ort.InferenceSession.create(modelPath);
    const input = normalize(hr, eda);

    const tensor = new ort.Tensor('float32', new Float32Array(input), [1, 2]);

    const feeds = { 'input': tensor };
    console.log('Feeds:', feeds);
    const results = await session.run(feeds);
    console.log('Model results:', results);
    const prediction = results['output_label'].data[0];
    return prediction;
  } catch (error) {
    console.error('Error running the ONNX model:', error);
    throw error;
  }
}

export default predictEdaHr
