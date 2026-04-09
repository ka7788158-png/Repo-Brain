import os
from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma

# 1. Setup
load_dotenv()
repo_path = "./my_code_folder" # Your code folder
db_path = "./repo_brain_db"    # Where the database will be saved

def build_database():
    print(f"📂 Loading code from: {repo_path}...")
    
    # 2. Load and Parse Code
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} files.")

    # syntax aware splitting 
    print("splitting code into chunks ...")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language = Language.PYTHON,
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = python_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks.")

    # embedding and storing
    print("🧠 Building Vector Database (this might take a few seconds)...")
    embeddings = OpenAIEmbeddings()
    # Create and persist the database locally
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"💾 Success! Database saved locally at '{db_path}'.")

if __name__ == "__main__":
    build_database()