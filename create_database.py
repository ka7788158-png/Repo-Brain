import os
import tempfile
import shutil
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Setup
load_dotenv()
db_path = "./repo_brain_db"    # Where the database will be saved

def build_database(source_input):
    """
    Takes either a local folder path or a GitHub URL and builds the Chroma DB.
    """
    repo_path = ""
    is_temp = False

    # 1. Detect if it's a GitHub Link or a Local Folder
    if source_input.startswith("http://") or source_input.startswith("https://"):
        print("🌐 GitHub URL detected! Cloning repository in the background...")
        # Create a hidden temporary folder on your computer
        repo_path = tempfile.mkdtemp()
        Repo.clone_from(source_input, repo_path)
        is_temp = True
    else:
        print("📁 Local folder detected!")
        repo_path = source_input

    print(f"📂 Scanning codebase at: {repo_path}...")
    
    # 2. Load and Parse Code
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md", ".cpp", ".java", ".c", ".s", ".rst",
                  ".lua"],
        exclude=["**/node_modules/**", "**/.git/**", "**/venv/**", "**/.env"]
    )

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} distinct code files.")

    # syntax aware splitting 
    print("splitting code into chunks ...")
    universal_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = universal_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks.")

    # embedding and storing
    print("🧠 Building Vector Database (this might take a few seconds)...")
    embeddings = HuggingFaceEmbeddings ()

    
    # Create and persist the database locally
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"💾 Success! Database saved locally at '{db_path}'.")

    # 5. Clean up the temp folder if we cloned a GitHub repo
    if is_temp:
        print("🧹 Cleaning up temporary GitHub files...")
        # Fix permissions issue on Windows before deleting
        def remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(repo_path, onerror=remove_readonly)
        
    return True