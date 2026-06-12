
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_chatbot_chain(retriever):
    llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=4096
)
    
    prompt = ChatPromptTemplate.from_template("""
You are a friendly customer support assistant for ABC Technologies Pvt. Ltd.

Use the provided context to answer questions.

If the user's message is a greeting (such as "Hi", "Hello", "Hey"), respond naturally and politely.

If the answer is not present in the context, say:
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
