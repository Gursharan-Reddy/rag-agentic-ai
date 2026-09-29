import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

def setup_pinecone_index(index_name: str, dimension: int = 384):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def run_ingestion(pdf_path: str, index_name: str = PINECONE_INDEX_NAME):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}.")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    setup_pinecone_index(index_name, dimension=384)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store

if __name__ == "__main__":
    pdf_file = os.path.join("data", "Ebook-Agentic-AI.pdf")
    print("Starting document ingestion into Pinecone with HuggingFace embeddings...")
    run_ingestion(pdf_file)
    print("Ingestion completed successfully!")