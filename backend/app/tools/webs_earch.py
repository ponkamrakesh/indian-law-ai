from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for legal information, case laws, research papers, and articles."""
    search = DuckDuckGoSearchRun()
    return search.run(query)