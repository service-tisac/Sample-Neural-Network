from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import numpy as np
from typing import List

# Import the training script
from training_script import SimpleNet, train_model

app = FastAPI()

# Load the trained model
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.post("/train")
def train_model_endpoint():
    """
    Train the ML model and save it to 'model.pth'
    """
    train_model()
    return {"message": "Model trained and saved successfully"}

@app.post("/predict")
async def predict(data: List[List[float]]):
    """
    Make predictions using the trained model
    """
    # Convert input data to PyTorch tensor
    input_data = torch.tensor(data, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_data).squeeze().tolist()
    
    # Convert outputs to binary predictions
    predictions = [1 if o > 0.5 else 0 for o in outputs]
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)