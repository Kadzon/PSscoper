import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chromadb import ChromaVectorStore
#from llama_index.readers.google import GoogleDriveReader
import chromadb
# load from my google drive (optional, if you want to pull docs directly instead of using the local sync folder)



# 1. Initialize Local Vector Database (Free)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("proposals")

# 2. Setup Free Local Embeddings (No API calls)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 3. Create Vector Store Context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def ingest_data(folder_path):
    """Reads files from your Google Drive sync folder and indexes them."""
    print(f"Indexing documents from {folder_path}...")
    documents = SimpleDirectoryReader(folder_path).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    return index

def query_rag(query_text):
    """Simulates the 'Professional Scoper' looking for past SOW examples."""
    # Re-load index from the persistent storage
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response

# --- TEST EXECUTION ---
if __name__ == "__main__":
    # Path to where you keep your historical SOWs/Transcripts
    PROPOSAL_FOLDER = "./my_proposals" 
    
    if not os.path.exists(PROPOSAL_FOLDER):
        os.makedirs(PROPOSAL_FOLDER)
        print(f"Please drop some .docx or .txt files into {PROPOSAL_FOLDER}")
    else:
        # Step 1: Ingest
        ingest_data(PROPOSAL_FOLDER)
        
        # Step 2: Test Retrieval
        # We test if it understands the 'Block-of-Hours' model [cite: 113]
        test_query = "What is the standard hour allocation for a Senior Engineer in Pillar I?"
        result = query_rag(test_query)
        
        print("\n--- RAG SEARCH RESULT ---")
        print(result)