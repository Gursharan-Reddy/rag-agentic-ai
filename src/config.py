import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment variables.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables.")