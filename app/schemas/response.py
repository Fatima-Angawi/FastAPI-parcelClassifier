from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    top_class: str
    top_score: float
    class_probabilities: Dict[str, float]