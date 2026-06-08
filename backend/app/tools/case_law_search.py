from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def search_indian_case_law(query: str) -> str:
    """Search for Indian case laws from Supreme Court, High Courts, and tribunals."""
    search = DuckDuckGoSearchRun()
    enhanced_query = f"India case law OR judgment OR supreme court OR high court {query}"
    return search.run(enhanced_query)