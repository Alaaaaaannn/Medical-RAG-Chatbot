from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src import config, helper


def main():
    print("Loading and chunking PDFs from", config.DATA_DIR, "...")
    chunks = helper.build_chunks()
    print(f"  -> {len(chunks)} usable chunks")

    helper.save_chunks(chunks)
    print(f"  -> cached chunks to {config.CHUNKS_PATH}")

    embeddings = helper.embeddings_init()

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if not pc.has_index(config.INDEX_NAME):
        print("Creating index", config.INDEX_NAME, f"(dim={config.EMBED_DIM})")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    print("Uploading embeddings to Pinecone ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.INDEX_NAME,
    )
    print("Done. Index", config.INDEX_NAME, "is ready.")


if __name__ == "__main__":
    main()
