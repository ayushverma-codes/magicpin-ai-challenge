import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

if not model:
    raise ValueError("GEMINI_MODEL is not set in .env")

print(f"Using model: {model}")
print("Testing Gemini inference...")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini inference is working!"
    )

    output = response.text

    print("\n✅ Gemini inference is working!")
    print("Response:")
    print(output)

except Exception as e:
    print("\n❌ Gemini inference failed!")
    print(f"Error: {type(e).__name__}: {e}")