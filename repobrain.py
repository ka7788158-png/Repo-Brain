import os
from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

# we will use LangChain's GenericLoader paired with a LanguageParser. This tells LangChain: 
# "Hey, read this like Python code, and try to keep functions and classes together."

# load your API key
load_dotenv()

# 1. Point this to the folder you want the AI to read
repo_path = "./my_code_folder" 

print(f"Loading code from: {repo_path}...\n")

# the code loader and parser
loader = GenericLoader.from_filesystem(
    repo_path,
    glob = "**/*",
    suffixes = [".py"], # We are telling it to only look for Python files right now
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)

# execute the load
documents = loader.load()

print(f"✅ Success! Loaded {len(documents)} distinct code blocks/files.\n")

# 4. Let's peek at the very first document to see what the AI will see
if documents:
    print("--- 🔍 SNEAK PEEK AT DOCUMENT 0 ---")
    print(documents[0].page_content[:300]) # Print first 300 characters
    print("\n--- 🏷️ METADATA ---")
    print(documents[0].metadata)