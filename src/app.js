import express from 'express'
import cors from 'cors';
import { predictHR } from './hrPredictorNode.js';
import { predictText } from './textPredictorNode.js';
import { predictImage } from './imagePredictorNode.js';
import multer from 'multer';
import path from 'path';
const app=express()
app.use(cors()); 

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage: storage });
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/predict',upload.single('image'),async(req,res)=>{
  const {hr,eda,text}=JSON.parse(req.body.data)
  let prediction=0
  let textPrediction=0
  let imagePrediction=0
  let availableWeights = [0.05, 0.46, 0.49];
  if(hr && eda){
    prediction=await predictHR(eda,hr)
    prediction=prediction*10*availableWeights[Math.floor(Math.random()*3)]
  }else{
    availableWeights[0]=0
  }
  if(text){
    textPrediction=await predictText(text,0.85)
    textPrediction=textPrediction.probability[1]
  }else{
    availableWeights[1]=0
  }
  if(req.file){
    imagePrediction=await predictImage(req.file.path)
  }else{
    availableWeights[2]=0
  }
  let normalizedWeights = availableWeights.map(weight => weight / availableWeights.reduce((a, b) => a + b, 0));
  let finalPrediction = normalizedWeights[0] * prediction + normalizedWeights[1] * textPrediction + normalizedWeights[2] * imagePrediction;
  res.status(200).json({status:true,prediction:finalPrediction})
})
export default app
