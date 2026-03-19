import io
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import docx2txt

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_MIME_EXPORT_MAP = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx"
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx"
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx"
    ),
}

# --- Setup ChromaDB and Embeddings (unchanged from your original) ---
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("proposals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_drive_service():
    """Handles OAuth and returns an authenticated Drive service."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)

def resolve_file(service, f: dict) -> dict:
    """
    If a file is a Drive shortcut, follow it to the real target file.
    Returns the real file metadata dict, or the original if not a shortcut.
    """
    if f["mimeType"] != "application/vnd.google-apps.shortcut":
        return f  # Not a shortcut, nothing to do

    shortcut_details = service.files().get(
        fileId=f["id"],
        fields="shortcutDetails(targetId, targetMimeType)",
        supportsAllDrives=True
    ).execute()

    target_id = shortcut_details["shortcutDetails"]["targetId"]
    target_mime = shortcut_details["shortcutDetails"]["targetMimeType"]

    print(f"  🔗 '{f['name']}' is a shortcut → resolving to target {target_id}")

    # Return a corrected metadata dict with the real ID and MIME type
    return {
        "id": target_id,
        "name": f["name"],      # Keep the shortcut's display name
        "mimeType": target_mime
    }


def download_file(service, file_id, file_name, mime_type) -> tuple[bytes, str]:
    """Exports Google Docs or directly downloads binary files."""
    if mime_type in GOOGLE_MIME_EXPORT_MAP:
        export_mime, extension = GOOGLE_MIME_EXPORT_MAP[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        if not file_name.endswith(extension):
            file_name += extension
    else:
        request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue(), file_name


def bytes_to_document(file_bytes: bytes, file_name: str) -> Document | None:
    """Converts downloaded file bytes into a LlamaIndex Document."""
    try:
        if file_name.endswith(".docx"):
            text = docx2txt.process(io.BytesIO(file_bytes))
        elif file_name.endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

        return Document(text=text, metadata={"source": file_name})
    except Exception as e:
        print(f"  Could not parse {file_name}: {e}")
        return None


def sync_drive_to_local_rag(folder_id: str):
    service = get_drive_service()

    print(f"Listing files in folder: {folder_id}...")
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = results.get("files", [])
    if not files:
        print("No files found. Check folder ID and permissions.")
        return None


    documents = []
    for f in files:
         # ✅ Add this line — resolves shortcuts to their real target
        f = resolve_file(service, f)
        mime = f["mimeType"]
        name = f["name"]

        # Skip unsupported Google Workspace types (Forms, Maps, etc.)
        if mime.startswith("application/vnd.google-apps") and mime not in GOOGLE_MIME_EXPORT_MAP:
            print(f"  Skipping unsupported type: {name} ({mime})")
            continue

        print(f"  Downloading: {name}...")
        try:
            file_bytes, final_name = download_file(service, f["id"], name, mime)
            doc = bytes_to_document(file_bytes, final_name)
            if doc:
                documents.append(doc)
        except Exception as e:
            print(f"  Failed on {name}: {e}")

    if not documents:
        print("No documents could be parsed.")
        return None

    print(f"\nIndexing {len(documents)} documents into ChromaDB...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print("Done!")
    return index

### File diagnostic test execution
# def diagnose_folder(folder_id: str):
#     """
#     Prints a full diagnostic report of everything in a Drive folder
#     so you can see exactly what types of files are there and why
#     they might be getting skipped.
#     """
#     service = get_drive_service()

#     print(f"\n{'='*60}")
#     print(f"DIAGNOSING FOLDER: {folder_id}")
#     print(f"{'='*60}\n")

#     # Step 1: Check the folder itself
#     try:
#         folder = service.files().get(
#             fileId=folder_id,
#             fields="id, name, mimeType, capabilities",
#             supportsAllDrives=True
#         ).execute()
#         print(f"✅ Folder found: '{folder['name']}'")
#         print(f"   Can read children: {folder.get('capabilities', {}).get('canListChildren', 'unknown')}\n")
#     except Exception as e:
#         print(f"❌ Could not access folder: {e}")
#         print("   → Check the folder ID and that your account has at least Viewer access.\n")
#         return

#     # Step 2: List ALL files with no filters
#     try:
#         results = service.files().list(
#             q=f"'{folder_id}' in parents and trashed=false",
#             fields="files(id, name, mimeType, size, capabilities)",
#             supportsAllDrives=True,
#             includeItemsFromAllDrives=True,
#             pageSize=100
#         ).execute()
#     except Exception as e:
#         print(f"❌ Failed to list files: {e}")
#         return

#     files = results.get("files", [])

#     if not files:
#         print("❌ No files returned at all.")
#         print("   Possible reasons:")
#         print("   → Files are in a Shared Drive but supportsAllDrives isn't working")
#         print("   → Your OAuth token lacks drive.readonly scope — delete token.json and re-auth")
#         print("   → Folder is empty or files are in subfolders (this only checks 1 level deep)\n")
#         return

#     print(f"Found {len(files)} file(s). Here's what each one is:\n")

#     supported_types = set(GOOGLE_MIME_EXPORT_MAP.keys()) | {
#         "application/pdf",
#         "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         "text/plain",
#     }

#     for f in files:
#         mime = f["mimeType"]
#         name = f["name"]
#         size = f.get("size", "N/A (Google-native file)")
#         can_download = f.get("capabilities", {}).get("canDownload", "unknown")

#         print(f"  📄 {name}")
#         print(f"     MIME type   : {mime}")
#         print(f"     Size        : {size}")
#         print(f"     canDownload : {can_download}")

#         if mime in GOOGLE_MIME_EXPORT_MAP:
#             print(f"     Status      : ✅ Will be EXPORTED as {GOOGLE_MIME_EXPORT_MAP[mime][1]}")
#         elif mime in supported_types:
#             print(f"     Status      : ✅ Will be DOWNLOADED directly")
#         elif mime.startswith("application/vnd.google-apps"):
#             print(f"     Status      : ⚠️  SKIPPED — unsupported Google Workspace type")
#             print(f"     Fix         : Add '{mime}' to GOOGLE_MIME_EXPORT_MAP if you need it")
#         elif mime == "application/vnd.google-apps.folder":
#             print(f"     Status      : ⚠️  SKIPPED — this is a subfolder (not recursed into)")
#         else:
#             print(f"     Status      : ⚠️  SKIPPED — mime type not in supported list")
#             print(f"     Fix         : Add handling for '{mime}' in bytes_to_document()")
#         print()


MY_DRIVE_FOLDER_ID = "1K9Qw9q0l4okI5TxnORSM3l5SigVm5MdA"

if __name__ == "__main__":
   sync_drive_to_local_rag(MY_DRIVE_FOLDER_ID)
   #diagnose_folder(MY_DRIVE_FOLDER_ID)