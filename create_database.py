import os
import tempfile
import shutil
from git import Repo 
from dotenv import load_dotenv

# 👇 NEW BULLETPROOF IMPORTS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings 
from langchain_community.vectorstores import Chroma

load_dotenv()
DB_PATH = "./repo_brain_db"

def build_database(source_input):
    """
    Takes either a local folder path or a GitHub URL and builds the Chroma DB.
    """
    repo_path = ""
    is_temp = False
    
    # 1. Detect if it's a GitHub Link or a Local Folder
    if source_input.startswith("http://") or source_input.startswith("https://"):
        print("🌐 GitHub URL detected! Cloning repository in the background...")
        repo_path = tempfile.mkdtemp()
        Repo.clone_from(source_input, repo_path)
        is_temp = True
    else:
        print("📁 Local folder detected!")
        repo_path = source_input

    print(f"📂 Scanning codebase at: {repo_path}...")
    
    # 👇 2. BULLETPROOF LOADER LOGIC
    documents = []
    # Explicitly list the extensions we care about
    allowed_extensions = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.cpp", "*.c", "*.h", "*.hpp", "*.java"]
    
    for ext in allowed_extensions:
        # DirectoryLoader specifically targets the glob pattern and forces TextLoader
        loader = DirectoryLoader(
            repo_path, 
            glob=f"**/{ext}", 
            loader_cls=TextLoader, 
            loader_kwargs={'autodetect_encoding': True} # Prevents Windows/UTF-8 crashes!
        )
        # Add the found documents to our master list
        documents.extend(loader.load())
        
    print(f"✅ Successfully loaded {len(documents)} distinct code files.")

    # 3. Universal Splitting
    print("🔪 Splitting code into chunks...")
    universal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = universal_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks.")

    # 4. Embed and Store
    print("🧠 Building Vector Database (this might take a minute)...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key, 
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Delete old DB so we don't mix projects
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("💾 Success! Database saved locally.")
    
    # 5. Clean up the temp folder if we cloned a GitHub repo
    if is_temp:
        print("🧹 Cleaning up temporary GitHub files...")
        def remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(repo_path, onerror=remove_readonly)
        
    return True
