from fastapi import FastAPI, UploadFile, File
from app.schemas.response import PredictionResponse
from app.services.model_engine import predict
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=PredictionResponse)
async def classify_parcel(file: UploadFile = File(...)):

    # 1. Read the raw bytes from the uploaded file
    image_bytes = await file.read()
    
    # 2. Pass those bytes to predict in a thread pool to avoid async issues with PyTorch
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, predict, image_bytes)
    
    return result