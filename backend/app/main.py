from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.langgraph_agent import process_legal_query
from app.services.vector_store import get_vectorstore, add_documents
from app.config import settings
import mlflow
from datetime import datetime
import json

app = FastAPI(
    title="IndianLawAI - Production Grade Legal Intelligence Assistant",
    description="Free, modular RAG system for Indian Laws (IPC, CrPC, etc.) with LangGraph, Guardrails, Safety, Caching, MLflow Eval. Deployable on Vercel (frontend) + free backend hosting.",
    version="1.0.0"
)

# CORS for frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    filter_act: Optional[str] = None  # e.g., "IPC" for metadata filter

class QueryResponse(BaseModel):
    response: str
    language: Dict[str, str]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    safety: Dict[str, Any]
    timestamp: str

class IngestRequest(BaseModel):
    documents: List[Dict[str, Any]]  # [{"text": "...", "metadata": {"act": "IPC", "section": "302", "source_url": "..."}}]
    version: str = "1.0"
    collection: str = "indian_law"

class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    version: str
    collection: str


@app.on_event("startup")
async def startup_event():
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Only connect to MLflow server if it's not localhost
        if "localhost" not in str(settings.mlflow_tracking_uri):
            mlflow.set_experiment("IndianLawAI_RAG_Evaluation")
        else:
            print("MLflow: Using local file storage (no server connection)")
            
    except Exception as e:
        print(f"MLflow startup skipped (not critical for production): {str(e)}")
    
    print("IndianLawAI started successfully")

@app.get("/")
async def root():
    return {
        "message": "IndianLawAI - Free Production Grade Indian Law Assistant",
        "features": ["LangGraph Agent", "Guardrails & Safety", "Multi-lang (Indian langs)", "RAG with versioning", "MLflow + RAGAS Eval", "Caching", "Completely Free Stack"],
        "data_sources": "Public: indiacode.nic.in, courtbook.in, legislative.gov.in, devgan.in (scraped dynamically)",
        "deploy": "Frontend: Vercel | Backend: Render/Railway free tier or Vercel Python"
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        result = process_legal_query(request.query)
        
        # Optional metadata filter (future: enhance with filter_act)
        if request.filter_act:
            result["metadata"]["filter_applied"] = request.filter_act
        
        return QueryResponse(
            response=result["response"],
            language=result["language"],
            sources=result["sources"],
            metadata=result["metadata"],
            safety=result["safety"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """Add new legal data with versioning. Supports future DB additions or bulk upload."""
    try:
        vectorstore = get_vectorstore(request.collection)
        num_added = add_documents(vectorstore, request.documents, request.version)
        return IngestResponse(
            status="success",
            chunks_added=num_added,
            version=request.version,
            collection=request.collection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...), version: str = Form("1.0"), collection: str = Form("indian_law")):
    """Upload text/JSON file with documents for ingestion. Supports PDF parsing in future."""
    try:
        content = await file.read()
        if file.filename.endswith('.json'):
            docs = json.loads(content)
        else:
            # Simple text: treat as one doc, or parse
            docs = [{"text": content.decode('utf-8'), "metadata": {"source": file.filename, "act": "Uploaded"}}]
        
        vectorstore = get_vectorstore(collection)
        num = add_documents(vectorstore, docs, version)
        return {"status": "success", "chunks_added": num, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "vectorstore": "Chroma active", "llm": "Groq free tier", "embeddings": "free local BGE"}

# MLflow logging example endpoint (for metrics)
@app.post("/log_evaluation")
async def log_evaluation(metrics: Dict[str, float], run_name: str = "rag_eval"):
    """Log RAGAS or custom metrics to MLflow for tracking."""
    with mlflow.start_run(run_name=run_name):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_param("model", settings.llm_model)
        mlflow.log_param("embedding", settings.embedding_model)
    return {"status": "logged", "metrics": metrics}