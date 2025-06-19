import httpx

r = httpx.get("https://agents-mcp-hackathon-mcp-chai1-modal.hf.space")
print("Response status:", r.status_code)
print("Raw text preview:", r.text[:300])  # to see if it’s HTML or empty