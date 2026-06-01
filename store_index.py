from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from src.helper import load_pdfs,filter,text_splitter,embeddings_init
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdfs("./data")
filter_data = filter(extracted_data)
text_chunk = text_splitter(filter_data)

embeddings = embeddings_init()

pinecone_api_key = PINECONE_API_KEY

p_client = Pinecone(api_key=pinecone_api_key)

index_name = "medical-rag-chatbot"

if not p_client.has_index(index_name):
    p_client.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = p_client.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    index_name=index_name,
    embedding=embeddings
)