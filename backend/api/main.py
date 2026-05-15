from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.predict import router as predict_router

app = FastAPI(title="PIGNet Inference API", version="1.0.0")

# Enable CORS for local dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

@app.get("/")
def read_root():
    return {"message": "PIGNet Inference API is running. Check /docs for endpoints."}