from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    use_cache: bool

@app.get("/")
def home():
    return {"status": "Backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    start_time = time.time()

    if request.use_cache:
        time.sleep(0.5)
    else:
        time.sleep(1.8)

    user_message = request.message.lower()

    if "dragon" in user_message:
        npc_response = "The dragon guards the cave because it protects an ancient crystal that powers the entire valley."
    elif "cave" in user_message:
        npc_response = "The cave is old and filled with glowing stones. Many travelers enter, but not all return."
    elif "castle" in user_message:
        npc_response = "The castle gates are closed because strange lights appeared near the mountain last night."
    elif "treasure" in user_message:
        npc_response = "The real treasure is not gold. It is a memory stone that stores the forgotten history of this world."
    else:
        npc_response = "That is a great question. This world has many secrets, and your choices will shape what happens next."

    latency_ms = round((time.time() - start_time) * 1000)

    return {
        "response": npc_response,
        "latency_ms": latency_ms,
        "tokens_per_second": 55 if request.use_cache else 15,
        "use_cache": request.use_cache
    }