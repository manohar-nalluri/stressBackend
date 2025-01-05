import sys
import onnxruntime as ort
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

vocab_path = 'tfidf_vocab.json'
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

def vectorize_text(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            vector[vocab[word]] = 1 
    return vector

def predict_text(text, confidence):
    model_path = 'text_model.onnx'
    session = ort.InferenceSession(model_path)

    processed_text = preprocess_text(text)
    vectorized_input = vectorize_text(processed_text, vocab)
    vectorized_input_with_confidence = np.append(vectorized_input, confidence)

    input_tensor = np.array([vectorized_input_with_confidence], dtype=np.float32)

    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)

    output_label = outputs[0][0] 
    output_probability = outputs[1][0] 

    result = {
        "label": int(output_label),
        "probability": output_probability
    }

    return result

if __name__ == "__main__":
    input_text = sys.argv[1]
    confidence = float(sys.argv[2])

    result = predict_text(input_text, confidence)
    
    print(json.dumps(result))
