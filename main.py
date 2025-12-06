import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter(model, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": True  # enable token-by-token streaming
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    return response
