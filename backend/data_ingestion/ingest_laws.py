"""
Data Ingestion Pipeline for Indian Laws.
"""
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vectorstore, add_documents
from app.config import settings
from typing import List, Dict

def scrape_ipc_sections(max_sections: int = 20) -> List[Dict]:
    base_url = "https://devgan.in/ipc_section.php?section="
    documents = []
    
    for i in range(1, max_sections + 1):
        try:
            url = f"{base_url}{i}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            content_div = soup.find('div', class_='content') or soup.find('main') or soup.body
            text = content_div.get_text(separator='\n', strip=True) if content_div else soup.get_text()
            
            if len(text) > 100:
                documents.append({
                    "text": text[:2000],
                    "metadata": {
                        "act": "Indian Penal Code (IPC)",
                        "section": f"Section {i}",
                        "source_url": url,
                        "source_type": "web_scrape",
                        "category": "Criminal Law"
                    }
                })
        except Exception as e:
            print(f"Error scraping section {i}: {e}")
            continue
    
    return documents

def main():
    print("Starting Indian Law Data Ingestion...")
    vectorstore = get_vectorstore()
    ipc_docs = scrape_ipc_sections(max_sections=20)
    
    if ipc_docs:
        num = add_documents(vectorstore, ipc_docs, version="1.0")
        print(f"Successfully ingested {num} chunks.")
    else:
        print("No documents scraped.")

if __name__ == "__main__":
    main()