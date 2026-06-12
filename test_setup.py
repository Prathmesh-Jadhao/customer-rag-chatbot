from utils.pdf_loader import load_and_split_pdf
from utils.embeddings import create_vector_store

# 1. Provide the path to your PDF
pdf_path = "data/Company_Document_Policy_.pdf"

# 2. Run the loader
chunks = load_and_split_pdf(pdf_path)

# 3. Create the vector store
vector_store = create_vector_store(chunks)

print("Test complete! The 'vectorstore' folder should now be populated.")