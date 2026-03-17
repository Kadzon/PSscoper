import chainlit as cl
from langchain_openai import ChatOpenAI
from docling.document_converter import DocumentConverter
import os
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
# 2. System Prompt derived from the SOW Structure
SYSTEM_PROMPT = """
You are the Professional Scoper. Your goal is to generate a structured Scope of Work (SOW).
Structure:
1. Executive Summary: Define global objectives (e.g., 110 agents, 70 sites)[cite: 5, 6].
2. Engagement Roles: Distinguish between the Senior Engineer (technical) and Project Manager (coordination)[cite: 13, 23].
3. Pillars: 
   - Pillar I: Contact Center Optimization (CX)[cite: 33].
   - Pillar II: Enterprise Phone System Deployment (EX)[cite: 67].
4. Commercials: Use a block-of-hours model with a Not-to-Exceed cap[cite: 112, 131].
5. Timeline: Week-by-week projections[cite: 135].
6. Risks/Assumptions: Include IVR misalignment risks and client dependencies[cite: 198, 210].
"""
# def extract_text_from_docx(file_path):
#     """Properly extracts text from a binary .docx file."""
#     try:
#         doc = Document(file_path)
#         return "\n".join([para.text for para in doc.paragraphs])
#     except Exception as e:
#         return f"Error parsing .docx: {str(e)}"

@cl.on_chat_start
async def start():
    # Initialize history with the specific system persona
    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    await cl.Message(content="Professional Scoper is active. I'll help you define roles, workstreams, and hour allocations.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    # Ensure we have a message string to append extracted content to (safe if content is None)
    user_input = message.content if message.content is not None else ""

    if message.elements:
        for element in message.elements:
            if element.name.lower().endswith(".docx"):
                # Use Docling to convert binary to clean Markdown
                msg_processing = cl.Message(content=f"Parsing {element.name} with Docling...")
                await msg_processing.send()
                
                result = converter.convert(element.path)
                markdown_content = result.document.export_to_markdown()
                
                user_input += f"\n\n--- EXTRACTED CONTENT FROM {element.name} ---\n{markdown_content}"
                
                # Update message content then update the message in the UI
                msg_processing.content = f"✅ {element.name} parsed successfully."
                await msg_processing.update()

    history.append({"role": "user", "content":user_input})
    
    msg = cl.Message(content="")
    
    # Use astream for real-time UI updates
    async for chunk in llm.astream(history):
        if chunk.content:
            await msg.stream_token(chunk.content)

    history.append({"role": "assistant", "content": msg.content})
    await msg.send()