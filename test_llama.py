import os
from llama_index.core import VectorStoreIndex
from llama_index.readers.google import GoogleDriveReader

# 1. Initialize the Google Drive Reader
# Point to your service account JSON file
loader = GoogleDriveReader(
    service_account_key_path="service_account.json"
)

# 2. Load data from a specific folder ID
# (Find the ID in the folder's URL: ://drive.google.com)
documents = loader.load_data(folder_id="1K9Qw9q0l4okI5TxnORSM3l5SigVm5MdA")

# 3. Create the index (This is where the RAG magic happens)
index = VectorStoreIndex.from_documents(documents)

# 4. Create a query engine and ask a question
query_engine = index.as_query_engine()
response = query_engine.query("What are the key takeaways from the project notes in this folder?")

print(response) 