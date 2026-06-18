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
from duckduckgo_search import DDGS
import chromadb.utils.embedding_functions as ef

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

SUPPORTED_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
DB_PATH = "chat.db"
UPLOAD_DIR = "uploads"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
tokenizer = tiktoken.get_encoding("cl100k_base")

multilingual_ef = ef.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
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
                grounding REAL DEFAULT NULL,
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

        cursor = await db.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "grounding" not in columns:
            await db.execute("ALTER TABLE messages ADD COLUMN grounding REAL DEFAULT NULL")
            await db.commit()


@app.on_event("startup")
async def startup():
    await init_db()


async def save_message(session_id: str, role: str, content: str, model: str, title: str = "", sources: str = "[]", grounding: float = None):
    async with aiosqlite.connect(DB_PATH) as db:
        if title:
            await db.execute(
                "INSERT INTO messages (session_id, role, content, model, title, sources, grounding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, model, title, sources, grounding)
            )
        else:
            await db.execute(
                "INSERT INTO messages (session_id, role, content, model, sources, grounding) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, model, sources, grounding)
            )
        await db.commit()


def retrieve_chunks(session_id: str, query: str, top_k: int = 3):
    try:
        collection = chroma_client.get_or_create_collection(
            name=f"docs_{session_id}",
            embedding_function=multilingual_ef
        )
        if collection.count() == 0:
            return None, []

        initial_k = min(10, collection.count())
        results = collection.query(query_texts=[query], n_results=initial_k)
        if not results or not results["documents"] or not results["documents"][0]:
            return None, []

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(chunks)

        if len(chunks) > top_k:
            cross_encoder = get_cross_encoder()
            pairs = [(query, chunk) for chunk in chunks]
            scores = cross_encoder.predict(pairs)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = ranked_indices[:top_k]
            chunks = [chunks[i] for i in top_indices]
            metadatas = [metadatas[i] for i in top_indices]

        sources = []
        for chunk, meta in zip(chunks, metadatas):
            sources.append({
                "filename": meta.get("filename", "unknown"),
                "excerpt": chunk[:200]
            })
        context = "\n\n---\n\n".join(chunks)
        return context, sources
    except Exception as e:
        print(f"RETRIEVE ERROR: {e}")
    return None, []


async def check_grounding(context: str, answer: str) -> dict:
    try:
        prompt = (
            f"Given context: {context}\nAnswer: {answer}\n"
            "Rate how much the answer is grounded in the context: "
            "0.0 (pure hallucination) to 1.0 (fully grounded). "
            "Respond with JSON: {\"score\": float, \"reason\": string}"
        )
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"GROUNDING CHECK ERROR: {e}")
        return {"score": 0.0, "reason": f"Grounding check failed: {e}"}


def needs_doc_context(query: str) -> bool:
    keywords = [
        "документ", "файл", "текст", "содержание", "содержимое",
        "прочитай", "прочти", "найди в документе", "что в файле",
        "what does the document", "what's in the file", "summarize",
        "проанализируй", "про файл", "в файле", "в документе",
        "по документу", "из документа", "из файла"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def needs_web_search(query: str) -> bool:
    keywords = [
        "погод", "weather", "новост", "news", "курс", "price", "цена",
        "сегодня", "today", "сейчас", "now", "актуальн", "latest",
        "что происходит", "what's happening", "спорт", "football", "soccer",
        "футбол", "хоккей", "hockey", "олимпиад", "election", "выбор",
        "stock", "bitcoin", "криптовалют", "нефть", "oil",
        "доллар", "dollar", "евро", "euro", "рубль", "ruble",
        "завтра", "tomorrow", "вчера", "yesterday"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def web_search(query: str, max_results: int = 3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            print(f"DEBUG WEB SEARCH: query='{query}', results_count={len(results)}")
            for r in results:
                print(f"  - {r.get('title', '')} | {r.get('href', '')}")
            if results:
                snippets = []
                sources = []
                for r in results:
                    snippet = f"{r.get('title', '')}\n{r.get('body', '')}"
                    snippets.append(snippet)
                    sources.append({
                        "filename": "web",
                        "excerpt": r.get('title', ''),
                        "url": r.get('href', '')
                    })
                context = "\n\n---\n\n".join(snippets)
                return context, sources
    except Exception as e:
        print(f"WEB SEARCH ERROR: {e}")
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

    collection = chroma_client.get_or_create_collection(
        name=f"docs_{session_id}",
        embedding_function=multilingual_ef
    )
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
            collection = chroma_client.get_collection(
                name=f"docs_{session_id}",
                embedding_function=multilingual_ef
            )
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

    doc_context, doc_sources = (None, [])
    if request.messages and needs_doc_context(request.messages[-1].content):
        doc_context, doc_sources = retrieve_chunks(session_id, request.messages[-1].content)

    web_context, web_sources = (None, [])
    if request.messages and needs_web_search(request.messages[-1].content):
        search_query = request.messages[-1].content + " актуальный 2025"
        web_context, web_sources = web_search(search_query)

    all_sources = (doc_sources or []) + (web_sources or [])

    system_prompt = (
        "You are a helpful assistant with access to two sources of information:\n"
        "1. Documents uploaded by the user\n"
        "2. Real-time web search results\n\n"
        "RULES:\n"
        "- If web search results are provided, ALWAYS use them to answer questions about current events, "
        "weather, news, prices, sports scores, or any time-sensitive information.\n"
        "- If document context is provided, use it to answer questions about the user's files.\n"
        "- Combine both sources when relevant.\n"
        "- Never say you don't have access to the internet if web search results are shown below.\n"
        "- Answer in the same language the user writes in."
    )
    if doc_context:
        system_prompt += f"\n\n--- Documents ---\n{doc_context}\n---"
    if web_context:
        system_prompt += f"\n\n--- Web Search Results ---\n{web_context}\n---"

    system_msg = {"role": "system", "content": system_prompt}
    messages_for_api = [system_msg] + [{"role": m.role, "content": m.content} for m in request.messages]

    if request.messages:
        last_user = request.messages[-1]
        if last_user.role == "user":
            title = ""
            if not request.session_id:
                title = last_user.content[:80]
            await save_message(session_id, "user", last_user.content, request.model, title)

    sources_json = json.dumps(all_sources)

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

        grounding = None
        if doc_context and full_response:
            grounding_result = await check_grounding(doc_context, full_response)
            grounding = grounding_result.get("score")
            yield f"data: {json.dumps({'grounding': grounding_result})}\n\n"

        await save_message(session_id, "assistant", full_response, request.model, "", sources_json, grounding)

        yield f"data: {json.dumps({'sources': all_sources, 'done': True, 'session_id': session_id})}\n\n"

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
            "SELECT role, content, model, sources, grounding, created_at FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "model": row["model"],
                "sources": json.loads(row["sources"]) if row["sources"] else [],
                "grounding": row["grounding"],
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
