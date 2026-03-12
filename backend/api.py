import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import UploadFile, File, Form
from pydantic import BaseModel

# Load .env
load_dotenv()

from page_index import LLMClient, PageIndexOptions, page_index_async, extract_pages
from tree_search import TreeSearcher
import uuid

logger = logging.getLogger("pageindex.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="PageIndex API")

# Allow frontend access (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory conversation store: conversation_id -> list of {role, content, timestamp}
CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Return structured validation errors for easier debugging on frontend
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
    model: Optional[str] = Form("gpt-4o"),
):
    """
    Upload a PDF, build a PageIndex (like run_pageindex.py) and store the index
    in the document_store folder. Returns metadata about the saved index.
    """
    # Ensure directories exist
    docs_dir = os.path.join(os.getcwd(), "documents")
    store_dir = os.path.join(os.getcwd(), "document_store")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(store_dir, exist_ok=True)

    filename = os.path.basename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    dest_path = os.path.join(docs_dir, filename)

    # Save uploaded file
    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Build index using LLM
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="No API key provided (form or OPENAI_API_KEY env).")

    try:
        llm_client = LLMClient(api_key=resolved_api_key, model=model)
        opt = PageIndexOptions(model=model)
        index = await page_index_async(dest_path, llm_client, opt, logger=logger)
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    # Save index to document_store
    base = os.path.splitext(filename)[0]
    index_filename = f"{base}_pageindex.json"
    index_path = os.path.join(store_dir, index_filename)
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to save index")
        raise HTTPException(status_code=500, detail=f"Failed to save index: {e}")

    # Prepare top-level summary for response
    top_level = [_node_minimal(n) for n in index.get("structure", [])]

    return {
        "pdf_path": dest_path,
        "index_path": index_path,
        "doc_name": index.get("doc_name"),
        "doc_description": index.get("doc_description"),
        "top_level_structure": top_level,
    }


@app.get("/status/{doc_id}")
async def get_status(doc_id: str):
    """
    Simple status check for a document id (basename without extension).
    Returns whether the index exists in document_store.
    """
    store_dir = os.path.join(os.getcwd(), "document_store")
    # Accept either raw id or full filename
    if doc_id.endswith("_pageindex.json"):
        index_path = os.path.join(store_dir, doc_id)
    else:
        index_path = os.path.join(store_dir, f"{doc_id}_pageindex.json")

    exists = os.path.exists(index_path)
    return {"doc_id": doc_id, "indexed": exists, "index_path": index_path if exists else None}


@app.get("/status/stream/{doc_id}")
async def status_stream(doc_id: str):
    """
    Simple Server-Sent Events (SSE) endpoint that streams a single status update.
    Frontend can poll this as a stream; if you need continuous updates convert indexing
    to background tasks and send progressive events.
    """
    async def event_generator():
        store_dir = os.path.join(os.getcwd(), "document_store")
        index_path = os.path.join(store_dir, f"{doc_id}_pageindex.json")
        if os.path.exists(index_path):
            payload = {"status": "indexed", "index_path": index_path}
        else:
            payload = {"status": "missing"}
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _node_minimal(n: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": n.get("node_id"),
        "title": n.get("title"),
        "start_index": n.get("start_index"),
        "end_index": n.get("end_index"),
        "summary": n.get("summary") or n.get("prefix_summary") or "",
    }


@app.post("/query")
async def query_endpoint(request: Request):
    """
    Flexible query endpoint. Accepts JSON bodies with keys:
      - query OR question (string)
      - pdf_path (optional)
      - api_key, model, base_url, force_reindex, max_depth (optional)

    If pdf_path is omitted, the server will try to pick the most recently
    indexed document from ./document_store and use the corresponding PDF in ./documents/.
    """
    body = await request.json()

    # Extract query text (support multiple keys)
    query_text = body.get("query") or body.get("question") or body.get("q")
    if not query_text:
        raise HTTPException(status_code=400, detail="No query provided (use 'query' or 'question').")

    # Resolve API key and model
    api_key = body.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    model = body.get("model") or "gpt-4o"
    base_url = body.get("base_url")
    force_reindex = bool(body.get("force_reindex", False))
    max_depth = int(body.get("max_depth", 4))

    if not api_key:
        raise HTTPException(status_code=400, detail="No API key provided (body or OPENAI_API_KEY env).")

    # Conversation handling
    conv_id = body.get("conversation_id")
    if conv_id:
        conv = CONVERSATIONS.get(conv_id, [])
    else:
        conv_id = str(uuid.uuid4())
        conv = []
        CONVERSATIONS[conv_id] = conv

    # Append user's query to conversation
    conv.append({"role": "user", "content": query_text, "timestamp": asyncio.get_event_loop().time()})

    # Resolve PDF path
    pdf_path = body.get("pdf_path")
    if not pdf_path:
        # Attempt to pick the most recent index in document_store
        store_dir = os.path.join(os.getcwd(), "document_store")
        if not os.path.exists(store_dir):
            raise HTTPException(status_code=400, detail="No pdf_path provided and document_store not found.")
        candidates = [os.path.join(store_dir, f) for f in os.listdir(store_dir) if f.endswith("_pageindex.json")]
        if not candidates:
            raise HTTPException(status_code=400, detail="No indexed documents found. Please upload a document first.")
        # pick latest modified
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        latest_index = candidates[0]
        base = os.path.basename(latest_index).replace("_pageindex.json", "")
        # corresponding PDF expected in ./documents/<base>.pdf
        possible_pdf = os.path.join(os.getcwd(), "documents", f"{base}.pdf")
        if os.path.exists(possible_pdf):
            pdf_path = possible_pdf
        else:
            # try project root
            possible_pdf = os.path.join(os.getcwd(), f"{base}.pdf")
            if os.path.exists(possible_pdf):
                pdf_path = possible_pdf
            else:
                raise HTTPException(status_code=400, detail=f"No PDF found for indexed document '{base}'. Expected {os.path.join('documents', base+'.pdf')}.")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    base = os.path.splitext(pdf_path)[0]
    index_path = f"{base}_pageindex.json"

    # Load or build index
    index: Dict[str, Any]
    if os.path.exists(index_path) and not force_reindex:
        logger.info(f"Loading index from {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        logger.info("Index not found or force_reindex requested. Building index (this may take time)...")
        llm_client = LLMClient(api_key=api_key, model=model, base_url=base_url or None)
        opt = PageIndexOptions(model=model)
        index = await page_index_async(pdf_path, llm_client, opt, logger=logger)
        # Save index for future use
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        logger.info(f"Index saved to {index_path}")

    tree = index.get("structure", [])

    # Extract pages (required for final answer)
    page_list = extract_pages(pdf_path)

    # Create LLM client and searcher
    llm = LLMClient(api_key=api_key, model=model, base_url=base_url or None)
    searcher = TreeSearcher(llm, max_depth=max_depth)

    try:
        result = await searcher.search(query_text, tree, page_list, conversation=conv)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Prepare minimal node info
    minimal_nodes = [_node_minimal(n) for n in result.relevant_nodes]

    # Append assistant response to conversation
    conv.append({"role": "assistant", "content": result.answer, "timestamp": asyncio.get_event_loop().time()})

    return {
        "conversation_id": conv_id,
        "query": result.query,
        "answer": result.answer,
        "relevant_pages": result.relevant_pages,
        "relevant_nodes": minimal_nodes,
        "reasoning_trace": result.reasoning_trace,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


@app.post("/query/stream")
async def query_stream(request: Request):
    """
    Streaming query endpoint. Returns Server-Sent Events (SSE) with incremental
    reasoning trace and answer chunks. Accepts same JSON body as /query.
    """
    body = await request.json()
    query_text = body.get("query") or body.get("question") or body.get("q")
    if not query_text:
        raise HTTPException(status_code=400, detail="No query provided (use 'query' or 'question').")

    api_key = body.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    model = body.get("model") or "gpt-4o"
    base_url = body.get("base_url")
    force_reindex = bool(body.get("force_reindex", False))
    max_depth = int(body.get("max_depth", 4))

    if not api_key:
        raise HTTPException(status_code=400, detail="No API key provided (body or OPENAI_API_KEY env).")

    # Conversation handling (similar to /query)
    conv_id = body.get("conversation_id")
    if conv_id:
        conv = CONVERSATIONS.get(conv_id, [])
    else:
        conv_id = str(uuid.uuid4())
        conv = []
        CONVERSATIONS[conv_id] = conv

    # Append user's query to conversation
    try:
        conv.append({"role": "user", "content": query_text, "timestamp": asyncio.get_event_loop().time()})
    except Exception:
        logger.exception("Failed to append user message to conversation")

    # Resolve PDF path same logic as /query
    pdf_path = body.get("pdf_path")
    if not pdf_path:
        store_dir = os.path.join(os.getcwd(), "document_store")
        if not os.path.exists(store_dir):
            raise HTTPException(status_code=400, detail="No pdf_path provided and document_store not found.")
        candidates = [os.path.join(store_dir, f) for f in os.listdir(store_dir) if f.endswith("_pageindex.json")]
        if not candidates:
            raise HTTPException(status_code=400, detail="No indexed documents found. Please upload a document first.")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        latest_index = candidates[0]
        base = os.path.basename(latest_index).replace("_pageindex.json", "")
        possible_pdf = os.path.join(os.getcwd(), "documents", f"{base}.pdf")
        if os.path.exists(possible_pdf):
            pdf_path = possible_pdf
        else:
            possible_pdf = os.path.join(os.getcwd(), f"{base}.pdf")
            if os.path.exists(possible_pdf):
                pdf_path = possible_pdf
            else:
                raise HTTPException(status_code=400, detail=f"No PDF found for indexed document '{base}'.")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    base = os.path.splitext(pdf_path)[0]
    index_path = f"{base}_pageindex.json"

    # Load or build index
    if os.path.exists(index_path) and not force_reindex:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        llm_client = LLMClient(api_key=api_key, model=model, base_url=base_url or None)
        opt = PageIndexOptions(model=model)
        index = await page_index_async(pdf_path, llm_client, opt, logger=logger)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    tree = index.get("structure", [])
    page_list = extract_pages(pdf_path)

    # Prepare a queue for incremental events
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    async def on_trace(msg: str):
        await queue.put({"type": "trace", "payload": msg})

    llm = LLMClient(api_key=api_key, model=model, base_url=base_url or None)
    searcher = TreeSearcher(llm, max_depth=max_depth, on_trace=on_trace)

    # Run search in background
    task = asyncio.create_task(searcher.search(query_text, tree, page_list))

    async def event_generator():
        # Stream trace events as they arrive. When search completes, stream answer in chunks.
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue

            # Ensure search completed
            result = await task

            # Send final reasoning trace as metadata
            await queue.put({"type": "final_trace", "payload": result.reasoning_trace})
            # Stream any remaining queued events
            while not queue.empty():
                ev = await queue.get()
                yield f"data: {json.dumps(ev)}\n\n"

            # Stream answer in word chunks
            answer = result.answer or ""
            words = answer.split(" ")
            chunk_size = 20
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                yield f"data: {json.dumps({'type':'answer_chunk','payload':chunk})}\n\n"
                await asyncio.sleep(0.02)

            # Final metadata event
            final_payload = {
                "type": "done",
                "payload": {
                    "relevant_pages": result.relevant_pages,
                    "relevant_nodes": [_node_minimal(n) for n in result.relevant_nodes],
                    "conversation_id": conv_id,
                },
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            # Append assistant response to conversation store
            try:
                conv.append({"role": "assistant", "content": result.answer, "timestamp": asyncio.get_event_loop().time()})
            except Exception:
                logger.exception("Failed to append to conversation store")
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")

