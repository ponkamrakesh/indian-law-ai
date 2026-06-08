from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.langgraph_agent import process_legal_query
from app.config import settings
import mlflow
from datetime import datetime

app = FastAPI(title="IndianLawAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    filter_act: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    language: Dict[str, str]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    safety: Dict[str, Any]
    timestamp: str

@app.on_event("startup")
async def startup_event():
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        if "localhost" not in str(settings.mlflow_tracking_uri):
            mlflow.set_experiment("IndianLawAI_RAG_Evaluation")
    except Exception as e:
        print(f"MLflow startup skipped: {str(e)}")
    print("IndianLawAI started successfully")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        result = process_legal_query(request.query)
        return QueryResponse(
            response=result["response"],
            language=result["language"],
            sources=result["sources"],
            metadata=result["metadata"],
            safety=result["safety"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        import traceback
        print("\n========== CHAT ENDPOINT ERROR ==========")
        traceback.print_exc()
        print("=========================================\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}