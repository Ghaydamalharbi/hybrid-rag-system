from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

   
    if not documents:
        print("[ERROR] No pages loaded from PDF")
        return []

  
    clean_docs = []
    for i, doc in enumerate(documents):
        text = doc.page_content.strip()

        if not text:
            continue

        doc.page_content = text
        doc.metadata["page"] = i + 1

        clean_docs.append(doc)

    print(f"[DEBUG] Loaded pages: {len(clean_docs)}")

    return clean_docs
