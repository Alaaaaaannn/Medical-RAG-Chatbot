import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FLASK_SECRET = os.getenv(
    "FLASK_SECRET",
    "8f3b1c2d9e7a46f5b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b",
)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-rag-chatbot")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.pkl")  
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5") 
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

DENSE_K = int(os.getenv("DENSE_K", "10"))
SPARSE_K = int(os.getenv("SPARSE_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))

USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"     
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"  
GROUNDING_CHECK = os.getenv("GROUNDING_CHECK", "true").lower() == "true" 

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
DAILY_BUDGET = int(os.getenv("DAILY_BUDGET", "500"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1000"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024))) 
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))