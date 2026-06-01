import pickle
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

from src import config


def load_pdfs(data_dir: str) -> List[Document]:
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_docs(docs: List[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        out.append(
            Document(
                page_content=d.page_content,
                metadata={"source": d.metadata.get("source"), "page": d.metadata.get("page")},
            )
        )
    return out

filter = filter_docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content and len(c.page_content.strip()) > 30]

text_splitter = split_docs


def build_chunks(data_dir: Optional[str] = None) -> List[Document]:
    data_dir = data_dir or config.DATA_DIR
    return split_docs(filter_docs(load_pdfs(data_dir)))


def embeddings_init() -> HuggingFaceBgeEmbeddings:
    return HuggingFaceBgeEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages:",
    )


def save_chunks(chunks: List[Document], path: Optional[str] = None) -> None:
    path = path or config.CHUNKS_PATH
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: Optional[str] = None) -> Optional[List[Document]]:
    path = path or config.CHUNKS_PATH
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None
