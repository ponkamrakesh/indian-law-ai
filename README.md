# IndianLawAI - Production Grade GenAI Legal Assistant

**Completely Free • Enterprise Ready • Modular • Deployable on Vercel**

A sophisticated RAG-based AI assistant for Indian Laws (IPC, CrPC, BNS, etc.) built with modern best practices using only free resources.

## Key Features
- **LangGraph Agentic Workflow**: Multi-step reasoning with language detection, safety guardrails, retrieval, generation
- **Production Guardrails & Safety**: Harmful intent detection, LLM-based safety checks, automatic disclaimers
- **Multi-Language Support**: Auto-detects Hindi, Tamil, Telugu, Bengali, etc. and responds in the same language
- **Dynamic Data Ingestion**: Scrapes real legal data from public internet sources (indiacode.nic.in, courtbook.in, devgan.in) with versioning
- **Versioning & Extensibility**: Add data from any source/DB, track versions of chunks
- **Caching**: Query-level caching for performance
- **MLflow + RAGAS Evaluation**: Precise metrics (faithfulness, answer relevancy, context precision/recall)
- **Modular Architecture**: Easy to swap components (LLM, Vector DB, Embeddings)
- **FastAPI Backend**: Clean REST API
- **Aesthetic UI**: Modern white + #8FDDDF teal design, fully responsive

## Tech Stack (100% Free)
- **LLM**: Groq (Llama 3.3 70B) - generous free tier, no card needed
- **Embeddings**: BAAI/bge-small-en-v1.5 (local via sentence-transformers)
- **Vector DB**: ChromaDB (free, local persist) - easily swap to Pinecone free / Supabase pgvector
- **Agent Framework**: LangGraph + LangChain
- **Backend**: FastAPI
- **Frontend**: Modern Tailwind + JS (deploy to Vercel)
- **Evaluation**: RAGAS + MLflow (free local or Dagshub)
- **Data**: Public domain from Indian government & legal sites

## Project Structure
```
indian_law_ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── config.py
│   │   ├── routers/             # Future expansion
│   │   ├── services/
│   │   │   ├── langgraph_agent.py   # Core agent with guardrails
│   │   │   ├── vector_store.py      # Modular Chroma (swap ready)
│   │   │   └── cache.py
│   │   └── utils/
│   │       ├── lang_detect.py
│   │       └── safety.py
│   ├── data_ingestion/
│   │   └── ingest_laws.py       # Dynamic web scraping pipeline
│   ├── evaluation/
│   │   └── evaluate_rag.py      # RAGAS + MLflow
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html               # Beautiful aesthetic UI (Vercel ready)
└── README.md
```

## Setup & Run (Completely Free)

### 1. Get Free API Key
- Go to https://console.groq.com/keys → Create free account → Copy key (no credit card)

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Create .env
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 3. Ingest Initial Data (Dynamic from Internet)
```bash
python data_ingestion/ingest_laws.py
```
This scrapes real IPC sections from public sites. Expand the scraper for more acts.

### 4. Run Backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Frontend (Vercel Deploy)
- The `frontend/index.html` is a complete standalone modern UI.
- Deploy to Vercel: Drag & drop the `frontend` folder or use `vercel --prod`.
- Update the fetch URL in the HTML script from `http://localhost:8000` to your deployed FastAPI URL.

**Alternative Backend Hosting (Recommended for Production)**:
- Deploy FastAPI to **Render.com** or **Railway.app** (both have excellent free tiers for Python/FastAPI).
- Frontend on Vercel.

## API Endpoints
- `POST /chat` - Main query endpoint (LangGraph powered)
- `POST /ingest` - Add new legal documents with versioning
- `POST /ingest_file` - Upload JSON/text files
- `POST /log_evaluation` - Log RAGAS metrics to MLflow
- `GET /health`

## Evaluation & Metrics
Run:
```bash
cd backend
python evaluation/evaluate_rag.py
```
Logs beautiful metrics to MLflow:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

View at http://localhost:5000 (MLflow UI)

## Production Recommendations
- **Vector DB**: Swap Chroma to Pinecone (free tier) or Supabase pgvector in `vector_store.py`
- **Caching**: Replace SimpleTTLCache with Upstash Redis (free tier)
- **Scaling**: Add rate limiting, auth if needed
- **Data Versioning**: Already built-in via metadata
- **Guardrails**: Extend safety.py with more rules or NeMo Guardrails

## Data Sources Used
- indiacode.nic.in (official)
- courtbook.in
- devgan.in
- legislative.gov.in
- Archive.org public domain texts

All data is public domain / government sourced.

## Why This is Production Grade & Enterprise Ready
- Modular code (easy maintenance & extension)
- Safety & guardrails at every step
- Caching for cost/performance
- Evaluation pipeline with precise metrics
- Versioned data ingestion
- Language detection for India's diversity
- Completely free stack with generous limits
- Clean separation of concerns

This project is ready for real users and can be extended with more acts, PDFs, or external databases.

**Created for you on 2026-06-08** — Enjoy building with completely free resources!