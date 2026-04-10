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
    embedding_function = embedding_model
)

# setup the retriever
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

# setup the llm
llm = ChatMistralAI(model="mistral-small-latest")

# giving the prompt 
prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Senior Full-Stack Developer assistant. 
        Your job is to navigate the user's codebase and explain how things work.
        
        CRITICAL RULES:
        1. Use ONLY the provided code context to answer.
        2. Always explicitly state WHICH FILE(S) you found the answer in.
        3. Provide code snippets if it helps explain the logic.
        4. If the answer isn't in the provided context, say "I cannot find this in the currently indexed codebase."
        
        Context:
        {context}"""),
        ("human", "{question}")
    ])

print("✅ System Ready! Let's talk about your code.")
print("--- Type '0' to exit ---")

# THE CHAT LOOP
while True:
    query = input("/n You: ")

    if query == 0:
        print("Good Bye !")
        break
    else :
    # Step A: Retrieve relevant code chunks
        retrieved_docs = retriever.invoke(query)

    # Step B: Combine the chunks into one big context string
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step C: Format the prompt with context and user question
        formatted_prompt = prompt.invoke({
        "context": context_text,
        "question": query
        })  

    # Step D: Get the answer from the LLM
        print("🤖 RepoBrain is thinking...")
        response = llm.invoke(formatted_prompt)
    
    print("\n🤖 RepoBrain:")
    print(response.content)
