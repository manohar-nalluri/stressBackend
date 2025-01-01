import express from 'express'
import cors from 'cors';
import predictEdaHr from './model.js';
const app=express()
app.use(cors()); 
app.use(express.json())

app.post('/predict',async(req,res)=>{
  const {hr,eda,text,image}=req.body.data
  console.log(hr,eda,req.body)
  let prediction
  if(hr && eda){
    prediction=await predictEdaHr({hr,eda})
    console.log('predicted value is ',prediction)
  }
  res.status(200).json({status:true,prediction:prediction})
})
export default app
