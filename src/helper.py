from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdfs(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documets = loader.load()
    return documets

from typing import List
from langchain.schema import Document

def filter(docs: List[Document]) -> List[Document]:
    docs_filtered: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        docs_filtered.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return docs_filtered

def text_splitter(docs_filtered):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk = text_splitter.split_documents(docs_filtered)
    return text_chunk

def embeddings_init():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = embeddings_init()