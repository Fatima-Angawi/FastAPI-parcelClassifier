import torch
import numpy as np
from PIL import Image
import io

from app.schemas.response import PredictionResponse
from app.core.config import settings
from ultralytics import YOLO

# Load model once at module import using settings path
model = YOLO(settings.MODEL_PATH)

def predict(image_input):
    try:
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            raise ValueError("Unsupported image input type.")
    except Exception:
        raise ValueError("Invalid image input.")

    with torch.no_grad():
        result = model.predict(image)
        if not result:
            raise ValueError("Model prediction returned empty result")
        res = result[0]
    
    # Safely extract probabilities
    if hasattr(res, 'probs') and res.probs is not None:
        try:
            probs = res.probs.data.cpu().numpy()
            top_class_index = int(res.probs.top1)
            top_score = float(res.probs.top1conf)
            top_class_name = res.names[top_class_index]
        except Exception as e:
            raise ValueError(f"Error extracting probabilities: {e}")
    else:
        raise ValueError("Model did not return probability scores")

    probs_dict = {
    name: float(probs[idx]) 
    for idx, name in model.names.items()
}

    return PredictionResponse(
        top_class=top_class_name,
        top_score=top_score,
        class_probabilities=probs_dict
    )
 
