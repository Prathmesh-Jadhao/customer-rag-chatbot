from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # Initialize free Hugging Face embedding model
    # 'sentence-transformers/all-MiniLM-L6-v2' is fast and accurate
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate vectors and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the database locally
    vector_store.save_local("vectorstore")
    print("Vector database created and saved.")
    return vector_store