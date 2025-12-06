# main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment variables")

# Initialize FastAPI
app = FastAPI()

# Load models from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mixtral-8x7b-instruct")
AVAILABLE_MODELS = [m.strip() for m in os.getenv("AVAILABLE_MODELS", "").split(",") if m.strip()]

# Pydantic models for request
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = None  # optional, will use DEFAULT_MODEL if not provided
    messages: list[Message]

# OpenRouter API call function
def call_openrouter(model, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False  # set True if you handle streaming on frontend
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

# Chat endpoint (POST only)
@app.post("/chat")
def chat(request: ChatRequest):
    model = request.model or DEFAULT_MODEL
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")
    
    messages_payload = [msg.dict() for msg in request.messages]
    response_json = call_openrouter(model, messages_payload)
    return response_json

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "LLMChat backend running"}
