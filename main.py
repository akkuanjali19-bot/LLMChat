from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# Enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default model
DEFAULT_MODEL = "mistralai/mixtral-8x7b-instruct"

# Request schema
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = None
    messages: list[Message]

# Health check
@app.get("/")
def home():
    return {"status": "OK"}

# Handle preflight (browser OPTIONS)
@app.options("/chat")
def options_chat():
    return {}

# Main chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    model = req.model or DEFAULT_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [m.dict() for m in req.messages]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=body
    )

    # Directly return OpenRouter response
    return response.json()
