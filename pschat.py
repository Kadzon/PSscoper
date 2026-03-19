import chainlit as cl
from langchain_openai import ChatOpenAI
from docling.document_converter import DocumentConverter
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import io
# from docx import Document # For .docx handling
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseUpload
# from google.oauth2.credentials import Credentials 



llm = ChatOpenAI(
    model="gemini-2.0-flash", 
    base_url="https://devs.ai/api/v1", 
    api_key=os.environ["DEV_AI_KEY"],
    streaming=True
)
converter = DocumentConverter()


# RAG Setup
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./chroma_db")          # same path as connect_drive.py
chroma_collection = db.get_or_create_collection("proposals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
rag_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)
rag_retriever = rag_index.as_retriever(similarity_top_k=4)  # pull top 4 relevant SOW chunks

def query_rag(question: str) -> str:
    """
    Retrieves the most relevant SOW chunks from ChromaDB
    and returns them as a formatted context string.
    """
    try:
        nodes = rag_retriever.retrieve(question)
        if not nodes:
            return ""
        chunks = []
        for i, node in enumerate(nodes, 1):
            source = node.metadata.get("source", "Unknown SOW")
            chunks.append(f"[SOW Reference {i} — {source}]\n{node.text}")
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""

# 2. System Prompt derived from the SOW Structure
SYSTEM_PROMPT = """
You are the Professional Scoper, an expert assistant for Solutions Engineers writing Statements of Work (SOWs).

You have two sources of knowledge:
1. A RAG knowledge base of previous SOWs — use these as style and structure references.
2. Meeting transcripts or documents uploaded by the user in this session.

Your goal is to help the Solutions Engineer:
- Draft or improve a Scope of Work based on a meeting transcript they upload.
- Answer questions about how previous SOWs were structured or priced.
- Suggest roles, workstreams, timelines, and hour allocations.

When writing or reviewing an SOW, always follow this structure:
1. Executive Summary: Define global objectives.
2. Engagement Roles: Distinguish between the Senior Engineer (technical) and Project Manager (coordination).
3. Pillars/Workstreams: Define the main focus areas, e.g.:
   - Pillar I: Contact Center Optimization (CX)
   - Pillar II: Enterprise Phone System Deployment (EX)
4. Commercials: Use a block-of-hours model with a Not-to-Exceed cap.
5. Timeline: Week-by-week projections.
6. Risks/Assumptions: Include IVR misalignment risks and client dependencies.

When you receive RAG context from previous SOWs, reference them naturally 
(e.g. "Based on a similar project we scoped before...") but do not paste them verbatim.
If no relevant past SOWs are found, rely on best practices and the uploaded transcript.
"""

@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    await cl.Message(
        content=(
            "👋 **Professional Scoper is active.**\n\n"
            "I can help you:\n"
            "- 📝 Draft a new SOW from a meeting transcript — just upload the file\n"
            "- 🔍 Look up how previous SOWs were structured or priced\n"
            "- ✏️  Review and improve a draft SOW\n\n"
            "How can I help you today?"
        )
    ).send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    # Ensure we have a message string to append extracted content to (safe if content is None)
    user_input = message.content if message.content is not None else ""
# Step1: Parse uploaded documents and append their content to the user input for the LLM to process. We also send intermediate messages to the UI to keep the user informed about the parsing status.
    if message.elements:
        for element in message.elements:
            if element.name.lower().endswith((".docx", ".pdf", ".txt", ".md")):
                # Use Docling to convert binary to clean Markdown
                msg_processing = cl.Message(content=f"Parsing {element.name} with Docling...")
                await msg_processing.send()
                
                try:
                    result = converter.convert(element.path)
                    markdown_content = result.document.export_to_markdown()
                    user_input += (
                        f"\n\n--- UPLOADED DOCUMENT: {element.name} ---\n"
                        f"{markdown_content}"
                    )
                    msg_processing.content = f"✅ **{element.name}** parsed successfully."
                except Exception as e:
                    msg_processing.content = f"❌ Could not parse **{element.name}**: {e}"
                await msg_processing.update()
# Step 2: Perform RAG retrieval based on the user's question or the content of the uploaded document. We append the retrieved context to the user input so that the LLM can use it to generate a more informed response.
    rag_context = query_rag(user_input)
    system_message = {"role": "system", "content": SYSTEM_PROMPT}

# history here should NOT include the system message — store only user/assistant turns
    conversation = history  # already excludes system prompt (see on_chat_start fix below)

    if rag_context:
        messages_to_send = (
            [system_message]
            + conversation
            + [{"role": "system", "content": (
                    "Relevant excerpts from previous SOWs in our knowledge base:\n\n"
                    + rag_context
            )}]
            + [{"role": "user", "content": user_input}]
        )
    else:
        messages_to_send = (
            [system_message]
            + conversation
            + [{"role": "user", "content": user_input}]
        )
        
    

#Step 3: Send the combined message history (system prompt + user input + RAG context) to the LLM and stream the response back to the user in real-time. We also update the message history with the new user input and assistant response for future interactions.
    history.append({"role": "user", "content":user_input})
    
    msg = cl.Message(content="")
    
    # Use astream for real-time UI updates
    async for chunk in llm.astream(messages_to_send):
        if chunk.content:
            await msg.stream_token(chunk.content)

    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", history)
    await msg.send()