from pydantic import BaseModel
from typing import List

class GestureRequest(BaseModel):
    landmarks: List[float]  # Assuming 1D list of landmarks

class GestureResponse(BaseModel):
    gesture: str
    confidence: float
