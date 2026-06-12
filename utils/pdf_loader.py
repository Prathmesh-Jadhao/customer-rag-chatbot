from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    Loads a PDF and splits it into smaller chunks for the RAG pipeline.
    """
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Define the splitter
    # chunk_size: characters per chunk
    # chunk_overlap: ensures context is kept between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    # 3. Split the documents
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully split PDF into {len(chunks)} chunks.")
    return chunks