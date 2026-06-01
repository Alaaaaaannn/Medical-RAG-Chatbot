from typing import List, Optional

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever

from src import config


def build_base_retriever(docsearch, chunks: Optional[List[Document]]):
    dense = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": config.DENSE_K}
    )
    if not config.USE_HYBRID or not chunks:
        return dense
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.SPARSE_K
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])


def build_reranker() -> CrossEncoderReranker:
    model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    return CrossEncoderReranker(model=model, top_n=config.RERANK_TOP_N)


def build_retriever(docsearch, chunks: Optional[List[Document]], llm):
    base = build_base_retriever(docsearch, chunks)
    if config.USE_MULTI_QUERY:
        base = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    return ContextualCompressionRetriever(
        base_compressor=build_reranker(), base_retriever=base
    )
