import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

class Settings(BaseModel):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Free local, high quality
    llm_model: str = "llama-3.3-70b-versatile"  # Groq free fast model, or mixtral-8x7b-32768
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    cache_ttl: int = 3600  # seconds
    disclaimer: str = "⚠️ This is an AI-generated response for informational purposes only. It is NOT legal advice. Always consult a qualified lawyer or official sources for legal matters. Laws may change; verify with India Code (indiacode.nic.in)."

settings = Settings()

# Validate key
if not settings.groq_api_key:
    print("Warning: GROQ_API_KEY not set. Get free key from console.groq.com")