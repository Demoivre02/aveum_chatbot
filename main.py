from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import httpx
import pandas as pd
from datetime import datetime, timedelta
import os
import re

app = FastAPI(title="Gemini Chatbot API with RAG and General LLM Capabilities")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini API configuration
GEMINI_API_KEY = "AIzaSyCJ2wHV2SKnwuxE3SottQ4Hy4Z4LCIDD2w"  # Replace with your actual API key
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Path to local CSV file - adjust this to the location of your CSV file
CSV_PATH = "agricultural_market_data.csv"  # Default to current directory

# In-memory storage for sessions
sessions: Dict[str, Dict] = {}

# Data models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: List[ChatMessage]

class SessionListResponse(BaseModel):
    sessions: List[str]

# Greeting patterns and responses
GREETING_PATTERNS = [
    r'\b(?:hi|hello|hey|greetings|good\s*(?:morning|afternoon|evening)|howdy)\b',
    r'\bhow\s+(?:are\s+you|is\s+it\s+going|are\s+things)\b',
    r'\bwhat\'?s\s+up\b',
    r'\bnice\s+to\s+meet\s+you\b'
]

GREETING_RESPONSES = [
    "Hello! How can I help you with agricultural market information or any other questions today?",
    "Hi there! I'm your assistant. I can help with agricultural data or answer general questions. What would you like to know?",
    "Greetings! I can help you explore agricultural market data or answer other questions. What would you like to know?",
    "Hello! I'm ready to help with your queries - whether about agricultural markets or anything else. What's on your mind?"
]

# Load the CSV dataset once during startup
def load_dataset():
    try:
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH)
        else:
            print(f"Warning: CSV file not found at {CSV_PATH}")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Global dataset variable
dataset = load_dataset()

# Session management
def create_session() -> str:
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "history": []
    }
    return session_id

def get_session(session_id: str) -> Dict:
    if session_id not in sessions:
        # Return a new session instead of raising an error
        new_session_id = create_session()
        return sessions[new_session_id]
    
    sessions[session_id]["last_activity"] = datetime.now()
    return sessions[session_id]

def cleanup_old_sessions():
    cutoff = datetime.now() - timedelta(hours=24)
    expired_sessions = [sid for sid, data in sessions.items() if data["last_activity"] < cutoff]
    for sid in expired_sessions:
        sessions.pop(sid, None)

# Check if a message contains a greeting
def is_greeting(message: str) -> bool:
    message = message.lower()
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, message):
            return True
    return False

# Get a random greeting response
def get_greeting_response() -> str:
    import random
    return random.choice(GREETING_RESPONSES)

# Check if a query is related to agricultural data
def is_agricultural_query(query: str) -> bool:
    if dataset is None:
        return False
    
    # Agricultural related keywords
    ag_keywords = [
        'farm', 'crop', 'agriculture', 'harvest', 'market', 'price', 'commodity',
        'goods', 'supply', 'demand', 'grain', 'wheat', 'corn', 'rice', 'produce',
        'vegetable', 'fruit', 'livestock', 'dairy', 'location', 'factors', 'volume',
        'quality', 'grade', 'date', 'record', 'market data', 'agricultural'
    ]
    
    # Get unique values from dataset columns to use as potential keywords
    if 'goods' in dataset.columns:
        goods_keywords = dataset['goods'].dropna().unique().tolist()
        ag_keywords.extend(goods_keywords)
    
    if 'location' in dataset.columns:
        location_keywords = dataset['location'].dropna().unique().tolist()
        ag_keywords.extend(location_keywords)
    
    # Check if query contains any agricultural keywords
    query_lower = query.lower()
    has_keywords = any(keyword.lower() in query_lower for keyword in ag_keywords)
    
    # Check if query explicitly asks about the dataset or data
    data_related = any(term in query_lower for term in ['dataset', 'csv', 'data', 'information', 'statistics'])
    
    return has_keywords or data_related

# RAG functionality with fallback to general LLM responses
async def generate_gemini_response(message: str, history: List[ChatMessage]) -> str:
    # Check if this is just a greeting
    if is_greeting(message) and len(message.split()) < 5:
        return get_greeting_response()
    
    context = ""
    is_ag_query = is_agricultural_query(message)
    
    # Only perform RAG if it's an agricultural query and dataset is available
    if is_ag_query and dataset is not None:
        keywords = message.lower().split()
        
        # Create a text representation of each row for searching
        dataset['text'] = dataset.apply(
            lambda row: f"Location: {row['location']}, Goods: {row['goods']}, Price: {row['price']}, Factors: {row['price_factors']}, Supply: {row['supply_volume']}, Grade: {row['quality_grade']}, Date: {row['record_date']}",
            axis=1
        )
        
        # Find relevant rows based on keywords
        mask = dataset['text'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
        relevant_rows = dataset[mask].head(5)
        
        if not relevant_rows.empty:
            context = "Agricultural Market Data:\n" + "\n".join(relevant_rows['text'].tolist()) + "\n\n"
    
    # Prepare the conversation history for the API request
    contents = []
    for msg in history[-10:]:  # Limit to last 10 messages to avoid token limits
        role = "user" if msg.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg.content}]})
    
    # Add the current user message with context if it's an agricultural query
    user_message = message
    if context:
        user_message = f"{context}Using the data above if relevant, answer: {message}"
    
    # Add system instructions to guide the model
    system_instruction = """
    You are a helpful assistant that can answer both general questions and specific questions about agricultural market data. 
    If the user asks about agricultural data, use the provided context to give accurate answers.
    For general questions, use your knowledge to provide helpful responses.
    Always be conversational and friendly in your replies.
    """
    
    # Add system message at the beginning of contents
    contents = [{"role": "user", "parts": [{"text": system_instruction}]}, 
                {"role": "model", "parts": [{"text": "I understand. I'll answer both agricultural data questions using the context provided and general questions using my knowledge."}]}] + contents
    
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json={"contents": contents},
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"Error response from Gemini API: {response.text}")
                return "Error processing your request."
                
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Exception in Gemini API call: {e}")
        return "An error occurred while processing your request."

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Create a new session if none provided or if the provided one doesn't exist
    if not request.session_id or request.session_id not in sessions:
        session_id = create_session()
    else:
        session_id = request.session_id
    
    session = sessions[session_id]  # Directly access the session without get_session to avoid errors
    
    user_message = ChatMessage(role="user", content=request.message, timestamp=datetime.now().isoformat())
    session["history"].append(user_message)
    
    response_text = await generate_gemini_response(
        request.message, 
        session["history"]
    )
    
    assistant_message = ChatMessage(role="assistant", content=response_text, timestamp=datetime.now().isoformat())
    session["history"].append(assistant_message)
    
    # Update the last activity timestamp
    session["last_activity"] = datetime.now()
    
    if len(sessions) > 100:
        cleanup_old_sessions()
    
    return ChatResponse(
        session_id=session_id,
        response=response_text,
        history=session["history"]
    )

@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    return SessionListResponse(sessions=list(sessions.keys()))

@app.get("/api/sessions/{session_id}")
async def get_session_history(session_id: str):
    if session_id not in sessions:
        return {"message": "Session not found", "history": []}
    return sessions[session_id]["history"]

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        sessions.pop(session_id)
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

# Add a route to check if the dataset is loaded
@app.get("/api/dataset/status")
async def dataset_status():
    if dataset is None:
        return {"status": "not_loaded", "message": f"Dataset not found at {CSV_PATH}"}
    
    return {
        "status": "loaded",
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "sample": dataset.head(3).to_dict(orient="records")
    }

# Add a route to manually reload the dataset
@app.post("/api/dataset/reload")
async def reload_dataset():
    global dataset
    dataset = load_dataset()
    
    if dataset is None:
        return {"status": "error", "message": f"Failed to load dataset from {CSV_PATH}"}
    
    return {
        "status": "success",
        "rows": len(dataset),
        "columns": list(dataset.columns)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)