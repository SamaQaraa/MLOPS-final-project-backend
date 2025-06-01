from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time
from predict import predict  
import uvicorn

app = FastAPI()
REQUEST_COUNTER = Counter("http_requests_total", "Total HTTP Requests")
INVALID_INPUT_COUNTER = Counter("invalid_input_requests_total", "Total Requests With Empty Landmarks")
INFERENCE_TIME = Histogram(
    "model_inference_duration_seconds",
    "Model Inference Duration",
    buckets=[0.0001, 0.001, 0.01, 0.1, 1.0]  # Custom buckets
)


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
    REQUEST_COUNTER.inc()  # This works as seen in your dashboard
    
    try:
        data = await request.json()
    except Exception:
        INVALID_INPUT_COUNTER.inc()
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    landmarks = data.get("landmarks")
    
    if not landmarks:
        INVALID_INPUT_COUNTER.inc()
        raise HTTPException(status_code=400, detail="Missing landmarks field")

    # Start measurement
    start_time = time.time()
    try:
        label = predict(landmarks)
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        # Record duration in seconds
        duration = time.time() - start_time
        INFERENCE_TIME.observe(duration)  # This must be called
        print(f"Inference took: {duration:.4f}s")  # Debug output

    return {"gesture": label or "No gesture detected"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
