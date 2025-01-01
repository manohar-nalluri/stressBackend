import natural,{TfIdf} from 'natural';
import stopword from 'stopword';
import fs from 'fs';
import ort from 'onnxruntime-node'
const stopword = require('stopword');

const vocab = JSON.parse(fs.readFileSync('../tfidf_vocab.json', 'utf-8'));

const tfidf = new TfIdf();
Object.keys(vocab).forEach(word => tfidf.addDocument(word));

const preprocessText=(text)=>{
  text = text.toLowerCase().replace(/[^a-z\s]/g, ' ');

  const tokenizer = new natural.WordTokenizer();
  let tokens = tokenizer.tokenize(text);

  tokens = stopword.removeStopwords(tokens);

  const lemmatizer = new natural.LancasterStemmer();
  tokens = tokens.map(word => lemmatizer.stem(word));

  return tokens.join(' ');
}

const vectorizeText=(text)=>{
  const vector = new Array(Object.keys(vocab).length).fill(0);
  tfidf.tfidfs(text, (i, measure) => {
    vector[i] = measure;
  });
  return vector;
}



async function loadModel() {
  const session = await ort.InferenceSession.create('../text_model.onnx');
  return session;
}

export const predictText=async (text, confidence)=> {
  const session = await loadModel();

  const preprocessedText = preprocessText(text);
  const tfidfVector = vectorizeText(preprocessedText);

  tfidfVector.push(confidence);

  const tensor = new ort.Tensor('float32', new Float32Array(tfidfVector), [1, 5001]);

  const feeds = { input: tensor };
  const results = await session.run(feeds);

  return results.output.data[0];
}

const userText = "Today the day is sunny and bright";
const confidence = 0.8;

predictText(userText, confidence)
  .then(prediction => {
    console.log('Prediction:', prediction);
  })
  .catch(err => {
    console.error('Error:', err);
  });
