import torch
from datasets import load_dataset
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import argparse

# Assuming train.py is in the same directory
from train import CNN, CNNRNN, byte_level_tokenize

NUM_CLASSES = 235 # Forgot to save -> Hardcode
VOCAB_SIZE = 257
MAX_LEN = 128

app = FastAPI()

# Global variable to hold the model
model = None
ds = None

def load_model(model_path):
    global model
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Language Identification</title>
        </head>
        <body>
            <h1>Language Identification</h1>
            <form action="/predict" method="post">
                <textarea name="text" rows="10" cols="50"></textarea>
                <br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(text: str = Form(...)):
    if model is None:
        return {"error": "Model not loaded"}

    tokenized_input = byte_level_tokenize({'sentence': [text], 'label': [0]})['input_ids']
    input_tensor = torch.tensor(tokenized_input)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=-1)
        top10_probs, top10_indices = torch.topk(probabilities, 10)

    results = []
    for i in range(10):
        confidence = top10_probs[0][i].item()
        prediction_index = top10_indices[0][i].item()
        str_label = ds['train'].features['label'].int2str(prediction_index)
        results.append({"language": str_label, "confidence": confidence})
        
    return {"top_10_predictions": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Identification Web App')
    parser.add_argument('--model_path', type=str, default='runs/test_data_aug/best_model.pth', help='Path to the trained model')
    args = parser.parse_args()
    
    ds = load_dataset("MartinThoma/wili_2018")

    load_model(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)
