from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    user_input: str

def generate_response(user_input: str) -> str:

    user_input = "User input: '" + user_input + "'"
    messages = [{"role": "user", "content": user_input}]

    response = ollama.chat(model='gemmaBot', messages=messages)
    bot_message = response['message']['content']
    
    return bot_message

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    response = generate_response(user_input)
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Welcome to LLMBot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)