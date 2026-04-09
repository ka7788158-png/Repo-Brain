# chat interface

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

db_path = "./repo_brain_db"

# Load the existing database
print("🔌 Connecting to RepoBrain Database...")
embedding_model = HuggingFaceEmbeddings()
vectorstore = Chroma(
    persist_directory = db_path, 
    embedding = embedding_model
)

# setup the retriever
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

# setup the llm
llm = ChatMistralAI(model="mistral-small-latest")

# giving the prompt 
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", """You are an expert Senior Developer assistant. 
    Use ONLY the provided code context to answer the user's question. 
    If you don't know the answer based on the context, say "I cannot find this in the current codebase."
    
    Context:
    {context}"""),
    ("human", "{question}")
]
)

print("✅ System Ready! Let's talk about your code.")
print("--- Type '0' to exit ---")

# THE CHAT LOOP
while True:
    query = input("/n You: ")

    if query == 0:
        print("Good Bye !")
        break

    # Step A: Retrieve relevant code chunks
    retrieved_docs = retriever.invoke(query)

    # Step B: Combine the chunks into one big context string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step C: Format the prompt with context and user question
    formatted_prompt = ChatPromptTemplate.invoke({
        "context": context_text,
        "question": query
    })

    # Step D: Get the answer from the LLM
    print("🤖 RepoBrain is thinking...")
    response = llm.invoke(formatted_prompt)
    
    print("\n🤖 RepoBrain:")
    print(response.content)
    