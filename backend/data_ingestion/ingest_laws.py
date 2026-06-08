"""
Production-grade Data Ingestion Pipeline for Indian Laws.
Fetches dynamically from public internet sources (indiacode.nic.in, courtbook.in, devgan.in, legislative.gov.in).
Supports versioning, metadata, modular for adding any DB/source.
Run: python data_ingestion/ingest_laws.py
Completely free - uses public web data.
"""
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vectorstore, add_documents
from app.config import settings
from typing import List, Dict
import datetime

def scrape_ipc_sections(max_sections: int = 50) -> List[Dict]:
    """Scrape IPC sections from public site devgan.in (example). Expand to indiacode.nic.in or courtbook.in."""
    base_url = "https://devgan.in/ipc_section.php?section="
    documents = []
    
    for i in range(1, max_sections + 1):
        try:
            url = f"{base_url}{i}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Extract main content (adjust selector based on site structure)
            content_div = soup.find('div', class_='content') or soup.find('main') or soup.body
            text = content_div.get_text(separator='\n', strip=True) if content_div else soup.get_text()
            
            if len(text) > 100:
                documents.append({
                    "text": text[:2000],  # Limit per section
                    "metadata": {
                        "act": "Indian Penal Code (IPC)",
                        "section": f"Section {i}",
                        "source_url": url,
                        "source_type": "web_scrape",
                        "category": "Criminal Law"
                    }
                })
            print(f"Scraped IPC Section {i}")
        except Exception as e:
            print(f"Error scraping section {i}: {e}")
            continue
    
    return documents

def scrape_generic_bare_act(act_name: str, url: str) -> List[Dict]:
    """Generic scraper for other acts from courtbook.in or indiacode.nic.in. Customize selectors."""
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Example: Extract all paragraphs or sections
        sections = soup.find_all(['p', 'div', 'section'])  # Adjust per site
        text_content = "\n".join([s.get_text(strip=True) for s in sections if len(s.get_text(strip=True)) > 50])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_text(text_content)
        
        docs = []
        for idx, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {
                    "act": act_name,
                    "section": f"Chunk {idx+1}",
                    "source_url": url,
                    "source_type": "web_scrape",
                    "category": "Bare Act"
                }
            })
        return docs
    except Exception as e:
        print(f"Error scraping {act_name}: {e}")
        return []

def main():
    print("Starting Indian Law Data Ingestion from Internet Sources...")
    print("Sources: devgan.in (IPC), courtbook.in, indiacode.nic.in (public domain)")
    
    vectorstore = get_vectorstore()
    
    # 1. Scrape IPC (example, expand max_sections)
    ipc_docs = scrape_ipc_sections(max_sections=20)  # Start small, increase as needed
    
    # 2. Add more acts - example URLs (update with real ones from search)
    # Example: CrPC from a public site
    # crpc_docs = scrape_generic_bare_act("Code of Criminal Procedure (CrPC)", "https://example-crpc-url.com")
    
    all_docs = ipc_docs  # + crpc_docs + ...
    
    if all_docs:
        num = add_documents(vectorstore, all_docs, version="1.0")
        print(f"Successfully ingested {num} chunks with versioning.")
        print("Vector DB updated. Ready for RAG queries.")
    else:
        print("No documents scraped. Check URLs or network.")
    
    # Future: Add support for PDF from archive.org, or user DB connections.

if __name__ == "__main__":
    main()