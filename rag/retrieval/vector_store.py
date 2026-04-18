from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def create_vector_store(chunks):

    if not chunks:
        print("[ERROR] No chunks provided to vector store")
        return None

    
    clean_chunks = [c for c in chunks if c.page_content.strip()]

    if not clean_chunks:
        print("[ERROR] All chunks empty")
        return None

    print(f"[DEBUG] Clean chunks: {len(clean_chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"} 
    )

   
    persist_dir = "chroma_db"

    print("[DEBUG] Creating / Loading vector store...")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

  
    vectorstore.add_documents(clean_chunks)

   
    vectorstore.persist()

    print("[DEBUG] Vector store READY")

    return vectorstore
