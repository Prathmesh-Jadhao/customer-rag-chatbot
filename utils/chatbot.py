
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

def get_chatbot_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant")
    # To run on local device
    """llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048
)"""
    
    prompt = ChatPromptTemplate.from_template("""
You are a friendly and professional customer support assistant for ABC Technologies Pvt. Ltd.

Company Information:
- Company: ABC Technologies Pvt. Ltd.
- CEO: Komal Bhende 
- The company builds AI-powered software solutions, customer support systems, and cloud-based applications.

Your Role:
- Help users with questions about the company, its services, and general support queries.
- Be polite, clear, and concise in your responses.
- If the user greets you (e.g., "Hi", "Hello", "Hey"), respond in a friendly and natural way.

Answering Rules:
- Use ONLY the provided context and company information to answer questions.
- If the answer is not available in the context or company information, respond exactly:
  "I couldn't find that information in the knowledge base."

Context:
{context}

Question:
{input}

Answer:
""")
    
    # This creates the logic to combine retrieved docs with the LLM
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # This creates the full retrieval chain
    return create_retrieval_chain(retriever, combine_docs_chain)
