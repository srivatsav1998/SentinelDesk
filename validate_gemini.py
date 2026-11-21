import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def test_gemini_connection():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Usage:")
        print("  export GEMINI_API_KEY='your_api_key_here'")
        print("  python validate_gemini.py")
        sys.exit(1)

    print("Attempting connection to Gemini API...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        start_time = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Respond with only: 'API connection successful.'"
        )
        latency = round((time.perf_counter() - start_time) * 1000, 1)
        
        print(f"SUCCESS: {response.text.strip()}")
        print(f"Roundtrip Latency: {latency} ms")
        print("Your GEMINI_API_KEY is active and ready for live text generation.")
    except Exception as e:
        print(f"FAILED: Encountered error while connecting to Gemini API:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_gemini_connection()
