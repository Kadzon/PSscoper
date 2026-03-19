import os.path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.google import GoogleDriveReader
#from llama_index.readers.google.docs import GoogleDocsReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Setup Local Storage (The folder you asked about)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("proposals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 2. Setup Free Embeddings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def sync_drive_to_local_rag(folder_id):
    """
    Connects to Drive, reads SOWs, and stores them in your local vector folder.
    
    """
    #account for docs, sheets, slides (optional, but good for future-proofing)
    export_config = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    }
    # This will open a browser window for the first-time handshake
    loader = GoogleDriveReader(
        credentials_path="credentials.json", 
        #token_path="token.json",
        
        supports_all_drives=True,
        key_path=None
        )
    
    print(f"Syncing folder: {folder_id}...")
    
    try:
        # load_data with folder_id will now use the export settings for any native docs found
        documents = loader.load_data(folder_id=folder_id, export_mime_types=export_config,)
        
        if not documents:
            print("Warning: No documents were retrieved. Check your folder permissions.")
            return None

        # Index and Save to your local chroma_db folder
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model
        )
        print(f"Success! {len(documents)} documents indexed locally.")
        return index

    except Exception as e:
        print(f"An error occurred during sync: {e}")
        return None

    # print("Fetching documents from Google Drive...")
    # documents = loader.load_data(folder_id=folder_id)
    
    # # Index and Save to your local chroma_db folder
    # index = VectorStoreIndex.from_documents(
    #     documents, 
    #     storage_context=storage_context, 
    #     embed_model=embed_model
    # )
    # print(f"Success! {len(documents)} documents indexed locally.")
    # return index

# Replace with your actual Google Drive Folder ID
# (The string of letters/numbers at the end of the folder URL)
MY_DRIVE_FOLDER_ID = "1K9Qw9q0l4okI5TxnORSM3l5SigVm5MdA"

if __name__ == "__main__":
    sync_drive_to_local_rag(MY_DRIVE_FOLDER_ID)