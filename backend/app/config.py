import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "llama-3.3-70b-versatile"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    disclaimer: str = "This is an AI-generated response for informational purposes only. It is NOT legal advice."

settings = Settings()