# test_gemini.py
import os
import sys

try:
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}")
    print("Please install the required dependencies by running:\n")
    print("    python -m pip install langchain-google-genai google-generativeai python-dotenv\n")
    sys.exit(1)

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Please make sure a .env file exists and the API key is properly set."
    )

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

response = llm.invoke(
    "Please reply in one sentence to confirm that the Gemini API connection is successful."
)

print(response.content)

