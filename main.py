# main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Function to call OpenRouter API
def call_openrouter(model, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": True  # enable streaming
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    return response

# Initialize FastAPI
app = FastAPI()

# Load models from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mixtral-8x7b-instruct")
AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", "").split(",")

# Pydantic models for request
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = None
    messages: list[Message]

# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    # Use default model if none selected
    model = request.model or DEFAULT_MODEL

    # Check if model is in allowed list
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Model not available")

    # Prepare messages for OpenRouter
    messages_payload = [msg.dict() for msg in request.messages]

    # Call OpenRouter API
    response = call_openrouter(model, messages_payload)

    # Return JSON response
    return response.json()

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "LLMChat backend running"}
