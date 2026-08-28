import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY1")
model = os.getenv("GROQ_MODEL")

if not api_key:
    raise ValueError("GROQ_API_KEY1 is not set in .env")

if not model:
    raise ValueError("GROQ_MODEL is not set in .env")

print(f"Using model: {model}")
print("Testing Groq inference...")

try:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq inference is working!"
            }
        ],
        temperature=0,
        max_tokens=50,
    )

    output = response.choices[0].message.content

    print("\n✅ Groq inference is working!")
    print("Response:")
    print(output)

except Exception as e:
    print("\n❌ Groq inference failed!")
    print(f"Error: {type(e).__name__}: {e}")

