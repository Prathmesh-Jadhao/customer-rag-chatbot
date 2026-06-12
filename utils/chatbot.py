
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_chatbot_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based only on the following context:
    {context}
    
    Question: {input}
    """)
    
    # This creates the logic to combine retrieved docs with the LLM
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # This creates the full retrieval chain
    return create_retrieval_chain(retriever, combine_docs_chain)