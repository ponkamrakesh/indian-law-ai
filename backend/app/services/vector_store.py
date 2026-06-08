import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from typing import List, Dict, Any, Optional
import os

def get_embedding_model():
    """Free local embedding model - completely free, no API calls."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vectorstore(collection_name: str = "indian_law"):
    """Initialize or load Chroma vectorstore. Modular - easy to swap to Pinecone/Supabase/pgvector."""
    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)
    
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def add_documents(vectorstore: Chroma, documents: List[Dict[str, Any]], version: str = "1.0"):
    """Add documents with versioning metadata. Supports future DB additions."""
    texts = [doc["text"] for doc in documents]
    metadatas = []
    for doc in documents:
        meta = doc.get("metadata", {})
        meta.update({
            "version": version,
            "ingest_timestamp": __import__("datetime").datetime.now().isoformat(),
            "source_type": meta.get("source_type", "web_scrape")
        })
        metadatas.append(meta)
    
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    vectorstore.persist()
    return len(documents)

def search_similar(vectorstore: Chroma, query: str, k: int = None, filter_metadata: Optional[Dict] = None):
    """Retrieve with optional metadata filter (e.g., by act or version)."""
    if k is None:
        k = settings.top_k
    return vectorstore.similarity_search(query, k=k, filter=filter_metadata)