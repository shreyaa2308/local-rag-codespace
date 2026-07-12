import warnings
# Silence all annoying version/deprecation warnings
warnings.filterwarnings("ignore")

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Using the brand new, updated LangChain Ollama imports
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag():
    if not os.path.exists("sample.pdf"):
        print("❌ Error: 'sample.pdf' not found! Please upload a PDF to your folder.")
        return

    print("📖 Loading document...")
    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()

    print("✂️ Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print("🧠 Generating embeddings and building local index...")
    # Updated to new langchain-ollama syntax
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

    print("🤖 Booting Local Brain (Llama 3.2)...")
    # Updated to new langchain-ollama syntax
    llm = OllamaLLM(model="llama3.2", temperature=0)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    print("\n🎉 RAG System Ready! Type 'exit' to quit.")
    print("="*50)

    # LOOP FOREVER TO ACCEPT INTERACTIVE USER INPUT
    while True:
        user_query = input("\n🙋 Ask a question about your PDF: ")
        
        # Check if the user wants to leave
        if user_query.strip().lower() == 'exit':
            print("👋 Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("⏳ Processing (calculating math vectors + generating text)...")
        try:
            response = rag_chain.invoke(user_query)
            print(f"\n🤖 AI Answer:\n{response}")
            print("-" * 40)
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")

if __name__ == "__main__":
    run_rag()