from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import predict  
import uvicorn

app = FastAPI()

# Enable CORS for frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def get_prediction(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    landmarks = data.get("landmarks")
    
    if landmarks is None:
        raise HTTPException(status_code=400, detail="Missing landmarks field")

    try:
        label = predict(landmarks)
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {"gesture": label or "No gesture detected"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
