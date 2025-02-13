from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import clip
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

app = FastAPI()

@app.post("/clip-encode-image")
async def encode_image(image_file: UploadFile = File(...)):
    try:
        image = Image.open(image_file.file)
        image_preprocessed = clip_preprocess(image).unsqueeze(0).to(device)
        features = clip_model.encode_image(image_preprocessed).detach().cpu().numpy().astype('float32')
        return {"features": features.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/clip-encode-text")
async def encode_text(text: str):
    try:
        text_tokenized = clip.tokenize([text]).to(device)
        features = clip_model.encode_text(text_tokenized).detach().cpu().numpy().astype('float32')
        return {"features": features.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/")
def root():
    return {"message": "CLIP model is running"}