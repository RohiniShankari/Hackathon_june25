from serpapi import GoogleSearch

params = {
    "engine": "google",
    "q": "TP53 protein patents",
    "api_key": "eb4d219920c6277323e1c520077cf3a9100e5a72ebb2c6fd92e4199bd9c5a326"  # Try hardcoding temporarily
}

search = GoogleSearch(params)
results = search.get_dict()

print(results.get("organic_results", []))
