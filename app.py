import streamlit as st
import os
from dotenv import load_dotenv

# Swapped OpenAI for HuggingFace based on your request for free embeddings!
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Import our new database builder function at the top of app.py
from create_database import build_database 

# --- 1. Page Setup ---
st.set_page_config(page_title="RepoBrain", page_icon="🧠", layout="centered")
st.title("🧠 RepoBrain: Codebase Navigator")
st.write("Ask questions about your codebase, and the AI will find the answers in your files.")

# --- 2. Sidebar: Settings & Code Ingestion Panel ---
with st.sidebar:
    st.header("🔑 API Settings")
    # type="password" masks the key so it doesn't show on screen
    user_api_key = st.text_input("Enter your API Key (Hugging Face / Mistral):", type="password")
    
    st.markdown("---")
    
    st.header("📂 Load Codebase")
    st.markdown("Paste a local folder path OR a public GitHub URL.")
    
    source_input = st.text_input("Path / URL:", placeholder="https://github.com/user/repo")
    
    if st.button("Process Codebase"):
        # Step A: Block processing if no key is provided
        if not user_api_key:
            st.error("🛑 Bro, please enter your API Key first!")
        elif not source_input:
            st.warning("Please enter a path or URL first.")
        else:
            with st.spinner("Cloning, chunking, and embedding... This takes a minute for large repos!"):
                try:
                    # Trigger the function from create_database.py AND pass the key
                    build_database(source_input, user_api_key)
                    st.success("✅ Database Built Successfully!")
                    
                    # Clear the cache so Streamlit loads the NEW database, not the old one
                    st.cache_resource.clear() 
                except Exception as e:
                    st.error(f"Error processing codebase: {e}")
            
    st.markdown("---")
    st.caption("Note: Large repositories (like thousands of files) will take longer to embed and may consume more API tokens.")

# --- 3. Cache the AI Brain ---
# Added api_key as a parameter so Streamlit knows to rebuild if the key changes
@st.cache_resource
def load_brain(api_key):
    load_dotenv()
    db_path = "./repo_brain_db"
    
    # Initialize Hugging Face Embeddings with the user's key
    embeddings = HuggingFaceEmbeddings ()

    
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    
    # Initialize Mistral with the user's key
    llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=api_key)
    
    # Upgraded prompt so it acts like a Senior Dev and tells you which file it looked at
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
    
    return retriever, llm, prompt_template

# Load the brain only if the database exists AND the user provided a key
if os.path.exists("./repo_brain_db"):
    if user_api_key:
        retriever, llm, prompt_template = load_brain(user_api_key)
    else:
        st.info("👈 Please enter your API key in the sidebar to start chatting!")
        st.stop() # Stops the rest of the app from loading
else:
    st.info("👈 Please enter your API key and load a codebase from the sidebar to get started!")
    st.stop()

# --- 4. Manage Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. The Chat Interface ---
if prompt := st.chat_input("Ask a question about the code..."):
    
    # Add user message to screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Scanning codebase..."):
            # A. Retrieve Documents
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # B. Format and Predict
            formatted_prompt = prompt_template.invoke({"context": context_text, "question": prompt})
            response = llm.invoke(formatted_prompt)
            
            # C. Show Answer
            st.markdown(response.content)
            
            # D. BONUS: Show the source code the AI used!
            with st.expander("🔍 View Source Code Chunks Used"):
                for i, doc in enumerate(docs):
                    # Extract the filename from the metadata
                    source_file = doc.metadata.get("source", "Unknown File")
                    st.caption(f"Source {i+1}: `{source_file}`")
                    st.code(doc.page_content, language="python")

    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
