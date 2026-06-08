from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def legal_research(query: str) -> str:
    """Search for legal research papers, articles, law commission reports, and legal analysis."""
    search = DuckDuckGoSearchRun()
    enhanced_query = f"legal research OR law journal OR law commission report OR legal analysis India {query}"
    return search.run(enhanced_query)