from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    # Load the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store from local disk
    vector_store = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever object
    return vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks