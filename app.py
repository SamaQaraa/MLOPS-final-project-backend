from fastapi import FastAPI, Request
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
    data = await request.json()
    label = predict(data["landmarks"])
    print(label)
    return {"gesture": label}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
