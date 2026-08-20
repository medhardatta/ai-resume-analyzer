import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set in this terminal session.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Models available to this API key:\n")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(" -", m.name)
