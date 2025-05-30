# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import predict  # your existing function

app = FastAPI()

# Enable CORS for frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LandmarkInput(BaseModel):
    landmarks: list[float]  # List of 63 floats (21 x 3)

@app.post("/predict")
def get_prediction(data: LandmarkInput):
    label, confidence = predict(data.landmarks)
    return {"gesture": label, "confidence": confidence}
