import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import aiosqlite
import chromadb
import tiktoken
import fitz

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SUPPORTED_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
DB_PATH = "chat.db"
UPLOAD_DIR = "uploads"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
tokenizer = tiktoken.get_encoding("cl100k_base")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama-3.3-70b-versatile"
    session_id: Optional[str] = None


def chunk_text(text: str) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def parse_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                title TEXT DEFAULT '',
                sources TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


@app.on_event("startup")
async def startup():
    await init_db()


async def save_message(session_id: str, role: str, content: str, model: str, title: str = "", sources: str = "[]"):
    async with aiosqlite.connect(DB_PATH) as db:
        if title:
            await db.execute(
                "INSERT INTO messages (session_id, role, content, model, title, sources) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, model, title, sources)
            )
        else:
            await db.execute(
                "INSERT INTO messages (session_id, role, content, model, sources) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, model, sources)
            )
        await db.commit()


def retrieve_chunks(session_id: str, query: str, top_k: int = 3):
    try:
        collection = chroma_client.get_or_create_collection(name=f"docs_{session_id}")
        print(f"DEBUG RETRIEVE: collection={collection.name}, count={collection.count()}")
        if collection.count() == 0:
            return None, []
        results = collection.query(query_texts=[query], n_results=top_k)
        print(f"DEBUG RETRIEVE: results={results}")
        if results and results["documents"] and results["documents"][0]:
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(chunks)
            sources = []
            for chunk, meta in zip(chunks, metadatas):
                sources.append({
                    "filename": meta.get("filename", "unknown"),
                    "excerpt": chunk[:200]
                })
            context = "\n\n---\n\n".join(chunks)
            return context, sources
    except Exception as e:
        print(f"DEBUG RETRIEVE ERROR: {e}")
    return None, []


@app.post("/api/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, MD files supported")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO documents (id, session_id, filename) VALUES (?, ?, ?)",
            (file_id, session_id, file.filename)
        )
        await db.commit()

    text = parse_file(file_path)
    chunks = chunk_text(text)
    print(f"DEBUG UPLOAD: text_len={len(text)}, chunks={len(chunks)}")

    collection = chroma_client.get_or_create_collection(name=f"docs_{session_id}")
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": file.filename, "doc_id": file_id}] * len(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"DEBUG UPLOAD: collection count after add={collection.count()}")

    return {"ok": True, "file_id": file_id, "filename": file.filename, "chunks": len(chunks)}


@app.get("/api/documents")
async def get_documents(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, filename, created_at FROM documents WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [{"id": row["id"], "filename": row["filename"], "created_at": row["created_at"]} for row in rows]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT session_id, filename FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        session_id = row["session_id"]
        filename = row["filename"]

        try:
            collection = chroma_client.get_collection(name=f"docs_{session_id}")
            collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass

        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()

    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"ok": True}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model not supported. Choose from: {SUPPORTED_MODELS}")

    session_id = request.session_id or str(datetime.now().timestamp())

    context, sources = retrieve_chunks(session_id, request.messages[-1].content) if request.messages else (None, [])
    print(f"DEBUG CHAT: session_id={session_id}, context_len={len(context) if context else 0}, sources={len(sources)}")

    system_msg = {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context if relevant."}
    if context:
        system_msg["content"] = f"Answer based on the following context:\n\n{context}\n\n---\n\nIf the context doesn't contain relevant information, answer from your general knowledge."

    messages_for_api = [system_msg] + [{"role": m.role, "content": m.content} for m in request.messages]

    if request.messages:
        last_user = request.messages[-1]
        if last_user.role == "user":
            title = ""
            if not request.session_id:
                title = last_user.content[:80]
            await save_message(session_id, "user", last_user.content, request.model, title)

    sources_json = json.dumps(sources)

    async def generate():
        full_response = ""
        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=messages_for_api,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        await save_message(session_id, "assistant", full_response, request.model, "", sources_json)

        yield f"data: {json.dumps({'sources': sources, 'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/sessions")
async def get_sessions():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT session_id, MIN(created_at) as started_at,
                   MIN(title) as title, COUNT(*) as message_count
            FROM messages
            GROUP BY session_id
            ORDER BY started_at DESC
        """)
        rows = await cursor.fetchall()
        return [
            {
                "id": row["session_id"],
                "title": row["title"] or "Новый чат",
                "started_at": row["started_at"],
                "message_count": row["message_count"]
            }
            for row in rows
        ]


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT role, content, model, sources, created_at FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "model": row["model"],
                "sources": json.loads(row["sources"]) if row["sources"] else [],
                "created_at": row["created_at"]
            }
            for row in rows
        ]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        chroma_client.delete_collection(name=f"docs_{session_id}")
    except Exception:
        pass

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        await db.commit()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)