import os
import asyncio

from groq import AsyncGroq

# Load variables from .env
from dotenv import load_dotenv
load_dotenv()

async def test_groq_connection():
    api_key = os.getenv("GROQ_API_KEY1")
    model = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY1 is not set in .env")

    if not model:
        raise ValueError("GROQ_MODEL is not set in .env")

    print(f"Using model: {model}")
    print("Testing Async Groq inference...")

    try:
        # 1. Initialize the async client WITHOUT the base_url
        client = AsyncGroq(api_key=api_key, base_url="https://api.groq.com",)

        # 2. Await the response using the exact structure that worked
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: Async Groq inference is working!"
                }
            ],
            temperature=0,
            max_tokens=50,
        )

        output = response.choices[0].message.content

        print("\n✅ Connection successful!")
        print("Response:")
        print(output)

    except Exception as e:
        print("\n❌ Groq inference failed!")
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq_connection())