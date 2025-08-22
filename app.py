from typing import List, Union
import torch
import random
from datasets import load_dataset
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import argparse
import PyPDF2
import io
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
import secrets

# Assuming train.py is in the same directory
from train import CNN, CNNRNN, byte_level_tokenize

# Import Lightning model
try:
    from train_lightning import LanguageClassificationModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

NUM_CLASSES = 235 # Forgot to save -> Hardcode
VOCAB_SIZE = 257
MAX_LEN = 128

PASSWORD = open('passwd', 'r').read().strip()
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

app = FastAPI()
security = HTTPBasic()

# Global variable to hold the model
model = None
ds = None

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not secrets.compare_digest(credentials.password, PASSWORD):
        raise HTTPException(status_code=401, detail="Wrong password")
    return True

def load_model(model_path):
    global model
    
    # Check if it's a Lightning checkpoint
    if model_path.endswith('.ckpt') and LIGHTNING_AVAILABLE:
        print(f"Loading Lightning checkpoint from {model_path}")
        model = LanguageClassificationModule.load_from_checkpoint(model_path)
        model.eval()
        print(f"Loaded Lightning model: {model.hparams.model_type}")
        return
    
    # Original PyTorch checkpoint loading
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    args = checkpoint['args']
    
    if not hasattr(args, "model") or args['model'] == 'CNN':
        model = CNN(
            num_classes=NUM_CLASSES,
            max_len=MAX_LEN,
            hidden_dim=checkpoint['args']['hidden_dim'],
            vocab_size=VOCAB_SIZE,
            num_layers=checkpoint['args']['num_layers']
        )
    elif args['model'] == 'CNNRNN':
        model = CNNRNN(
            num_classes=NUM_CLASSES,
            max_len=MAX_LEN,
            hidden_dim=checkpoint['args']['hidden_dim'],
            vocab_size=VOCAB_SIZE,
            num_layers=checkpoint['args']['num_layers']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

@app.get("/", response_class=FileResponse)
async def read_root(auth: bool = Depends(check_auth)):
    return 'index.html'

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    # TODO: For now just file ending, this is not safe. But only a demo so I guess it's fine.
    if not file.filename.endswith('.pdf'):
        return {"error": "Invalid file type. Please upload a PDF."}

    try:
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text = text.split('\n')
        # TODO: I am sure this could be more efficient and better heuristics could be used
        text = [x for x in text if 
                not any (c.isdigit() for c in x) 
                and len(x) >= 30 
                and x.count(',') <= 2 
                and x.count('.') <= 2 
                and x.count('(') <= 2
        ]
        
        sampled_sentences = random.sample(text, 10)
        print(f"Sampled sentences: {sampled_sentences}")
        
    except Exception as e:
        return {"error": f"Error processing PDF file: {e}"}
    
    return await predict(sampled_sentences)

@app.post("/predict")
async def predict(text: Union[str, List[str]] = Form(...), auth: bool = Depends(check_auth)):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Handle both single string and list of strings
    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = text
    
    all_probabilities = []
    
    # Process each text
    for single_text in text_list:
        tokenized_input = byte_level_tokenize({'sentence': [single_text], 'label': [0]})['input_ids']
        input_tensor = torch.tensor(tokenized_input)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=-1)
            all_probabilities.append(probabilities[0])
    
    # Average all probability distributions
    avg_probabilities = torch.stack(all_probabilities).mean(dim=0)
    top10_probs, top10_indices = torch.topk(avg_probabilities, 10)
    
    results = []
    for i in range(10):
        confidence = top10_probs[i].item()
        prediction_index = top10_indices[i].item()
        str_label = ds['train'].features['label'].int2str(prediction_index)
        results.append({"language": str_label, "confidence": confidence})
        
    print(text_list)
    for text in text_list:
        print(f"Text: {text}")

    return {
        "top_10_predictions": results,
        "num_texts_processed": len(text_list)
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Identification Web App')
    parser.add_argument('--model_path', type=str, default='runs/lightning_v3/best-checkpoint-epoch=97-val_acc=0.8411.ckpt', help='Path to the trained model')
    args = parser.parse_args()
    
    ds = load_dataset("MartinThoma/wili_2018")

    load_model(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)
