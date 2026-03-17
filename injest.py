import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# SETTINGS
# Update this to your local Google Drive path
DRIVE_PATH = "G:/My Drive/Proposals" 
DB_PATH = "./proposal_vector_db"
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def build_knowledge_base():
    print(f"Loading documents from {DRIVE_PATH}...")
    
    # Load .docx files
    loader = DirectoryLoader(
        DRIVE_PATH, 
        glob="**/*.docx", 
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)

    # Create and save the Vector DB locally
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_PATH
    )
    print(f"Success! Saved {len(chunks)} chunks to {DB_PATH}")

if __name__ == "__main__":
    build_knowledge_base()