from serpapi import GoogleSearch
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()


@tool
def search_patents(query: str) -> list[dict]:
    """
    Uses SerpAPI to search for patent-related information based on the given query.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

if __name__ == "__main__":
    query = "\"aspirin synthesis\" (\"acetic anhydride\" OR \"acetyl chloride\") (\"salicylic acid\" OR \"2-hydroxybenzoic acid\") catalyst H2SO4 \"acetic acid\" heat"
    results = search_patents.invoke({"query": query})

    for item in results:
        print("🔍", item.get("title"))
        print("🔗", item.get("link"))
        print("📄", item.get("snippet"))
        print()
