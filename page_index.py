"""
PageIndex Reimplementation
--------------------------
A vectorless, reasoning-based RAG system that builds a hierarchical tree index
from long documents. Inspired by the PageIndex framework (VectifyAI/PageIndex).

This implementation supports any OpenAI-compatible API (OpenAI, Anthropic via
openai-compat, local Ollama, etc.) — no proprietary API keys from VectifyAI needed.

Two document types are supported:
  - PDF  → page_index(pdf_path, ...)
  - Markdown → md_to_tree(md_path, ...)

Core pipeline (PDF):
  1. Extract pages + count tokens
  2. Detect / extract Table of Contents
  3. Generate hierarchical TOC tree via LLM (with fallback strategies)
  4. Recursively subdivide large nodes
  5. Assemble final tree (post_processing / list_to_tree)
  6. Enrichment: node IDs, optional summaries, optional doc description
"""

import re
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import tiktoken
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Config / Options
# ---------------------------------------------------------------------------

@dataclass
class PageIndexOptions:
    model: str = "gpt-4o"
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 50
    max_token_num_each_node: int = 50_000
    if_add_node_id: bool = True
    if_add_node_summary: bool = True
    if_add_doc_description: bool = True
    if_add_node_text: bool = False
    summary_token_threshold: int = 200
    min_node_token: int = 100        # for markdown thinning


# ---------------------------------------------------------------------------
# LLM Client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around any OpenAI-compatible chat endpoint.

    Usage:
        client = LLMClient(api_key="sk-...", model="gpt-4o")
        # or for local Ollama:
        client = LLMClient(base_url="http://localhost:11434/v1", api_key="ollama", model="llama3")
    """

    def __init__(
        self,
        api_key: str = "YOUR_API_KEY",
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.model = model

    async def chat(self, prompt: str, system: str = "", max_tokens: int = 4096) -> tuple[str, str]:
        """Returns (response_text, finish_reason)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        choice = resp.choices[0]
        return choice.message.content or "", choice.finish_reason or "stop"

    async def chat_json(self, prompt: str, system: str = "", max_tokens: int = 4096) -> Any:
        """Returns parsed JSON; strips markdown fences."""
        text, _ = await self.chat(prompt, system=system, max_tokens=max_tokens)
        text = text.strip()
        # Strip ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

_ENC = None

def _get_encoder():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[tuple[str, int]]:
    """
    Returns list of (page_text, token_count) for every page in the PDF.
    Pages are 1-indexed in descriptions but 0-indexed in the list.
    """
    doc = fitz.open(pdf_path)
    result = []
    for page in doc:
        text = page.get_text("text")
        result.append((text, count_tokens(text)))
    doc.close()
    return result


def pages_to_tagged_text(page_list: list[tuple[str, int]], start: int, end: int) -> str:
    """
    Wrap each page with <physical_index_N> tags (1-based page numbers).
    start / end are 1-based, inclusive.
    """
    parts = []
    for i in range(start - 1, end):
        text, _ = page_list[i]
        parts.append(f"<physical_index_{i+1}>\n{text}\n</physical_index_{i+1}>")
    return "\n".join(parts)


def group_pages_into_chunks(
    page_list: list[tuple[str, int]],
    start: int,
    max_tokens: int = 6000,
) -> list[tuple[str, int, int]]:
    """
    Groups consecutive pages into chunks that stay within max_tokens.
    Returns list of (chunk_text, chunk_start_page, chunk_end_page).
    Pages are 1-based.
    """
    chunks = []
    cur_tokens = 0
    cur_start = start
    cur_parts: list[str] = []

    for i in range(start - 1, len(page_list)):
        text, tok = page_list[i]
        page_num = i + 1
        tagged = f"<physical_index_{page_num}>\n{text}\n</physical_index_{page_num}>"
        if cur_parts and cur_tokens + tok > max_tokens:
            chunks.append(("\n".join(cur_parts), cur_start, page_num - 1))
            cur_parts = [tagged]
            cur_tokens = tok
            cur_start = page_num
        else:
            cur_parts.append(tagged)
            cur_tokens += tok

    if cur_parts:
        chunks.append(("\n".join(cur_parts), cur_start, len(page_list)))

    return chunks


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_TOC_CHECK = "You are an expert document analyst."

PROMPT_CHECK_TOC = """You are given the first pages of a document. Determine:
1. Does a Table of Contents exist? (yes/no)
2. If yes, extract its full raw text as-is.
3. Does the TOC include page numbers? (yes/no)

Respond ONLY in JSON with keys:
  "toc_exists": "yes" or "no"
  "toc_content": "<raw TOC text or empty string>"
  "page_index_given_in_toc": "yes" or "no"

Document pages:
{text}
"""

PROMPT_TOC_TO_JSON = """Convert the following Table of Contents into a JSON array.
Each entry should have:
  "structure": hierarchical number like "1", "1.1", "2.3.1" etc.
  "title": section title (fix spacing only, do NOT change wording)
  "page": page number as integer (if available, else null)

Return ONLY the JSON array, no explanation.

TOC text:
{toc_text}
"""

PROMPT_FIND_PAGES_FOR_TOC = """You are given document pages with <physical_index_N> tags.
Below is a list of section titles. For each title, find the physical page index where
the section STARTS (i.e., the title first appears). Return ONLY a JSON array of objects
with keys "title" and "physical_index" (integer). If a title cannot be found, set
physical_index to null.

Section titles:
{titles}

Document pages:
{text}
"""

PROMPT_GENERATE_TOC_INIT = """You are an expert in extracting hierarchical structure.
Your task: generate a JSON array representing the Table of Contents of the document below.

Rules:
- Each entry has:
    "structure": numeric hierarchy string like "1", "1.1", "2", "2.1.3"
    "title": original title (fix spacing only)
    "physical_index": integer page number where the section starts (use <physical_index_N> tags)
- The document pages contain tags like <physical_index_N> marking the start of page N.
- Only include top-level sections and one level of subsections unless finer detail is evident.
- Return ONLY the JSON array.

Document pages:
{text}
"""

PROMPT_GENERATE_TOC_CONTINUE = """You are continuing to extract the hierarchical Table of Contents.
Previous structure extracted so far:
{prev_structure}

Now process the following additional pages and EXTEND the JSON array with new sections.
Apply the same rules (structure, title, physical_index). Return the FULL updated JSON array.

Additional pages:
{text}
"""

PROMPT_GENERATE_SUMMARY = """Write a concise 2-4 sentence summary of the following document section.
Focus on the main topics, key findings, or decisions described.

Section title: {title}
Section text:
{text}

Summary:"""

PROMPT_DOC_DESCRIPTION = """Given the following document tree structure, write ONE sentence that
uniquely describes this document (suitable for distinguishing it from other documents in a collection).

Tree structure:
{tree_json}

One-sentence description:"""


# ---------------------------------------------------------------------------
# Phase 2: TOC Detection
# ---------------------------------------------------------------------------

async def detect_and_extract_toc(
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    toc_check_pages: int = 20,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Checks the first `toc_check_pages` pages for a Table of Contents.
    Returns dict with keys: toc_exists, toc_content, page_index_given_in_toc.
    """
    end = min(toc_check_pages, len(page_list))
    text = pages_to_tagged_text(page_list, 1, end)
    prompt = PROMPT_CHECK_TOC.format(text=text)
    try:
        result = await llm.chat_json(prompt, system=SYSTEM_TOC_CHECK, max_tokens=2048)
        if logger:
            logger.debug(f"TOC check result: {result}")
        return result
    except Exception as e:
        if logger:
            logger.warning(f"TOC detection failed: {e}")
        return {"toc_exists": "no", "toc_content": "", "page_index_given_in_toc": "no"}


async def toc_text_to_json(toc_text: str, llm: LLMClient) -> list[dict]:
    """Converts raw TOC text into structured JSON list."""
    prompt = PROMPT_TOC_TO_JSON.format(toc_text=toc_text)
    try:
        return await llm.chat_json(prompt, max_tokens=3000)
    except Exception:
        return []


async def find_physical_indices_for_toc(
    toc_items: list[dict],
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    chunk_size: int = 6000,
) -> list[dict]:
    """
    For each TOC item, find its physical page in the PDF using LLM.
    Processes pages in chunks.
    """
    titles = [item["title"] for item in toc_items]
    titles_json = json.dumps(titles, indent=2)

    # Try to find in chunks
    all_mappings: dict[str, Optional[int]] = {t: None for t in titles}

    chunks = group_pages_into_chunks(page_list, 1, max_tokens=chunk_size)
    for chunk_text, chunk_start, chunk_end in chunks:
        # Only ask about titles not yet found
        remaining = [t for t, v in all_mappings.items() if v is None]
        if not remaining:
            break
        prompt = PROMPT_FIND_PAGES_FOR_TOC.format(
            titles=json.dumps(remaining, indent=2),
            text=chunk_text,
        )
        try:
            found = await llm.chat_json(prompt, max_tokens=2048)
            for item in found:
                title = item.get("title")
                idx = item.get("physical_index")
                if title in all_mappings and idx is not None:
                    all_mappings[title] = int(idx)
        except Exception:
            pass

    # Merge mappings back into toc_items
    result = []
    for item in toc_items:
        item = dict(item)
        item["physical_index"] = all_mappings.get(item["title"])
        if item["physical_index"] is not None:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Phase 3: Generate TOC from raw pages (no TOC in document)
# ---------------------------------------------------------------------------

async def generate_toc_from_pages(
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    start_page: int = 1,
    chunk_tokens: int = 6000,
) -> list[dict]:
    """
    When the document has no TOC, generate a hierarchical structure
    by reading through the pages chunk by chunk.
    """
    chunks = group_pages_into_chunks(page_list, start_page, max_tokens=chunk_tokens)
    toc_items: list[dict] = []

    for i, (chunk_text, chunk_start, chunk_end) in enumerate(chunks):
        try:
            if i == 0:
                prompt = PROMPT_GENERATE_TOC_INIT.format(text=chunk_text)
            else:
                prev = json.dumps(toc_items, indent=2)
                prompt = PROMPT_GENERATE_TOC_CONTINUE.format(
                    prev_structure=prev, text=chunk_text
                )
            toc_items = await llm.chat_json(prompt, max_tokens=3000)
        except Exception:
            pass  # Keep whatever we have so far

    return toc_items


# ---------------------------------------------------------------------------
# Phase 4 & 5: Post-processing → hierarchical tree
# ---------------------------------------------------------------------------

def _parse_structure_key(s: str) -> tuple[int, ...]:
    """'1.2.3' → (1, 2, 3)"""
    try:
        return tuple(int(x) for x in s.split(".") if x.strip())
    except ValueError:
        return (0,)


def list_to_tree(flat: list[dict], doc_end_page: int) -> list[dict]:
    """
    Converts a flat list of sections (with 'structure' field like '1', '1.1', '2')
    into a hierarchical tree. Each node gets start_index / end_index page numbers.
    """
    if not flat:
        return []

    # Sort by structure key
    flat = sorted(flat, key=lambda x: _parse_structure_key(x.get("structure", "0")))

    # Assign end_index = next sibling's start_index - 1, or doc_end_page
    result_flat = []
    for i, item in enumerate(flat):
        node = {
            "title": item.get("title", ""),
            "structure": item.get("structure", ""),
            "start_index": item.get("physical_index") or item.get("start_index", 1),
            "end_index": doc_end_page,
            "nodes": [],
        }
        if i + 1 < len(flat):
            next_start = flat[i + 1].get("physical_index") or flat[i + 1].get("start_index", doc_end_page)
            node["end_index"] = max(node["start_index"], next_start - 1)
        result_flat.append(node)

    # Build hierarchy using structure keys
    roots: list[dict] = []
    stack: list[tuple[tuple, dict]] = []  # (structure_key, node)

    for node in result_flat:
        key = _parse_structure_key(node["structure"])
        # Pop stack entries that are at same or deeper level
        while stack and len(stack[-1][0]) >= len(key):
            stack.pop()
        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            roots.append(node)
        stack.append((key, node))

    return roots


def post_processing(toc_items: list[dict], doc_end_page: int) -> list[dict]:
    """Full post-processing: validate page numbers then build tree."""
    valid = [item for item in toc_items if item.get("physical_index") is not None]
    return list_to_tree(valid, doc_end_page)


# ---------------------------------------------------------------------------
# Enrichment: node IDs
# ---------------------------------------------------------------------------

def write_node_id(nodes: list[dict], counter: list[int] | None = None) -> list[dict]:
    """
    Assigns sequential zero-padded 4-digit IDs in depth-first order.
    Mutates nodes in place; returns the list.
    """
    if counter is None:
        counter = [0]
    for node in nodes:
        node["node_id"] = str(counter[0]).zfill(4)
        counter[0] += 1
        if node.get("nodes"):
            write_node_id(node["nodes"], counter)
    return nodes


# ---------------------------------------------------------------------------
# Enrichment: node text
# ---------------------------------------------------------------------------

def add_node_text(nodes: list[dict], page_list: list[tuple[str, int]]):
    """Attach full text of page range to each node."""
    for node in nodes:
        start = node.get("start_index", 1)
        end = node.get("end_index", len(page_list))
        texts = [page_list[i][0] for i in range(start - 1, min(end, len(page_list)))]
        node["text"] = "\n".join(texts)
        if node.get("nodes"):
            add_node_text(node["nodes"], page_list)


# ---------------------------------------------------------------------------
# Enrichment: summaries
# ---------------------------------------------------------------------------

async def _summarize_node(node: dict, llm: LLMClient, threshold: int) -> None:
    text = node.get("text", "")
    if not text:
        return
    tok = count_tokens(text)
    if tok <= threshold:
        node["summary"] = text
        return
    is_leaf = not node.get("nodes")
    prompt = PROMPT_GENERATE_SUMMARY.format(title=node.get("title", ""), text=text[:8000])
    try:
        summary, _ = await llm.chat(prompt, max_tokens=300)
        key = "summary" if is_leaf else "prefix_summary"
        node[key] = summary.strip()
    except Exception:
        pass


async def generate_summaries(
    nodes: list[dict],
    llm: LLMClient,
    threshold: int = 200,
):
    """Concurrently generate summaries for all nodes (depth-first)."""
    tasks = []
    for node in nodes:
        tasks.append(_summarize_node(node, llm, threshold))
        if node.get("nodes"):
            tasks.append(generate_summaries(node["nodes"], llm, threshold))
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Enrichment: document description
# ---------------------------------------------------------------------------

def _clean_tree_for_description(nodes: list[dict]) -> list[dict]:
    """Strip text fields to keep prompt small."""
    cleaned = []
    for node in nodes:
        c = {k: v for k, v in node.items() if k not in ("text",)}
        if c.get("nodes"):
            c["nodes"] = _clean_tree_for_description(c["nodes"])
        cleaned.append(c)
    return cleaned


async def generate_doc_description(tree: list[dict], llm: LLMClient) -> str:
    clean = _clean_tree_for_description(tree)
    prompt = PROMPT_DOC_DESCRIPTION.format(tree_json=json.dumps(clean, indent=2)[:6000])
    try:
        desc, _ = await llm.chat(prompt, max_tokens=200)
        return desc.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Recursive node subdivision
# ---------------------------------------------------------------------------

async def process_large_node_recursively(
    node: dict,
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    opt: PageIndexOptions,
    logger: Optional[logging.Logger] = None,
    depth: int = 0,
):
    """
    If a node spans too many pages / tokens, subdivide it by running
    TOC generation on just that node's pages.
    """
    if depth > 5:
        return

    start = node.get("start_index", 1)
    end = node.get("end_index", len(page_list))
    num_pages = end - start + 1
    total_tokens = sum(page_list[i][1] for i in range(start - 1, min(end, len(page_list))))

    if num_pages <= opt.max_page_num_each_node and total_tokens <= opt.max_token_num_each_node:
        return  # Already small enough

    if logger:
        logger.debug(f"Subdividing node '{node.get('title')}' ({num_pages} pages, {total_tokens} tokens)")

    # Generate sub-TOC for this node's pages only
    sub_page_list = page_list[start - 1: end]
    sub_toc = await generate_toc_from_pages(sub_page_list, llm, start_page=1, chunk_tokens=6000)

    # Re-offset physical_index back to global
    for item in sub_toc:
        if item.get("physical_index") is not None:
            item["physical_index"] = item["physical_index"] + start - 1

    sub_tree = post_processing(sub_toc, end)

    if sub_tree and len(sub_tree) > 1:
        node["nodes"] = sub_tree
        # Recursively process large children
        tasks = [
            process_large_node_recursively(child, page_list, llm, opt, logger, depth + 1)
            for child in sub_tree
        ]
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main PDF entry point
# ---------------------------------------------------------------------------

async def page_index_async(
    pdf_path: str,
    llm: LLMClient,
    opt: Optional[PageIndexOptions] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Full PageIndex pipeline for a PDF document.
    Returns a dict with keys: doc_name, doc_description, structure.
    """
    if opt is None:
        opt = PageIndexOptions()
    if logger is None:
        logger = logging.getLogger("pageindex")

    logger.info(f"Loading PDF: {pdf_path}")
    page_list = extract_pages(pdf_path)
    total_pages = len(page_list)
    logger.info(f"Total pages: {total_pages}")

    # ---- Phase 2: TOC detection ----
    toc_result = await detect_and_extract_toc(
        page_list, llm, toc_check_pages=opt.toc_check_page_num, logger=logger
    )
    toc_exists = toc_result.get("toc_exists", "no") == "yes"
    toc_has_pages = toc_result.get("page_index_given_in_toc", "no") == "yes"
    toc_text = toc_result.get("toc_content", "")

    toc_items: list[dict] = []

    if toc_exists and toc_text.strip():
        logger.info("TOC found. Parsing...")
        toc_items = await toc_text_to_json(toc_text, llm)

        if toc_has_pages:
            logger.info("TOC has page numbers — mapping to physical indices.")
            # The page numbers in TOC may be logical (printed), not physical (PDF index).
            # We use the page numbers as hints but verify them.
            toc_items = await find_physical_indices_for_toc(toc_items, page_list, llm)
        else:
            logger.info("TOC lacks page numbers — searching for section starts in pages.")
            toc_items = await find_physical_indices_for_toc(toc_items, page_list, llm)
    else:
        logger.info("No TOC found — generating structure from document content.")
        toc_items = await generate_toc_from_pages(page_list, llm, chunk_tokens=6000)

    # ---- Phase 5: Build tree ----
    tree = post_processing(toc_items, total_pages)
    if not tree:
        logger.warning("Tree generation produced no nodes. Creating single root node.")
        tree = [{"title": "Document", "start_index": 1, "end_index": total_pages, "nodes": []}]

    # ---- Phase 4: Subdivide large nodes ----
    tasks = [
        process_large_node_recursively(node, page_list, llm, opt, logger)
        for node in tree
    ]
    await asyncio.gather(*tasks)

    # ---- Enrichment ----
    if opt.if_add_node_id:
        write_node_id(tree)

    if opt.if_add_node_text or opt.if_add_node_summary:
        add_node_text(tree, page_list)

    if opt.if_add_node_summary:
        logger.info("Generating node summaries...")
        await generate_summaries(tree, llm, opt.summary_token_threshold)
        if not opt.if_add_node_text:
            # Remove raw text unless user wants it
            _strip_text(tree)

    doc_description = ""
    if opt.if_add_doc_description:
        logger.info("Generating document description...")
        doc_description = await generate_doc_description(tree, llm)

    import os
    return {
        "doc_name": os.path.basename(pdf_path),
        "doc_description": doc_description,
        "structure": tree,
    }


def _strip_text(nodes: list[dict]):
    for node in nodes:
        node.pop("text", None)
        if node.get("nodes"):
            _strip_text(node["nodes"])


def page_index(
    pdf_path: str,
    llm: LLMClient,
    opt: Optional[PageIndexOptions] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Synchronous wrapper around page_index_async."""
    return asyncio.run(page_index_async(pdf_path, llm, opt, logger))


# ---------------------------------------------------------------------------
# Markdown entry point
# ---------------------------------------------------------------------------

def extract_nodes_from_markdown(md_text: str) -> list[dict]:
    """
    Parses markdown headers (#, ##, ###, ...) into a flat list of nodes.
    Skips headers inside code blocks.
    """
    lines = md_text.splitlines()
    nodes: list[dict] = []
    in_code_block = False
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    for line_num, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = header_pattern.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            nodes.append({"title": title, "level": level, "line_num": line_num})

    return nodes


def extract_node_text_content(
    nodes: list[dict], md_lines: list[str]
) -> list[dict]:
    """Attaches text content (from header to next header) to each node."""
    enriched = []
    for i, node in enumerate(nodes):
        start_line = node["line_num"] + 1
        end_line = nodes[i + 1]["line_num"] if i + 1 < len(nodes) else len(md_lines)
        text = "\n".join(md_lines[start_line:end_line]).strip()
        enriched.append({**node, "text": text, "token_count": count_tokens(text)})
    return enriched


def thin_nodes(nodes: list[dict], min_tokens: int) -> list[dict]:
    """
    Merge nodes with very low token counts into their parent's text
    rather than keeping them as separate children. This prevents
    overly granular trees for documentation with tiny sections.
    """
    result: list[dict] = []
    for node in nodes:
        if node.get("token_count", 0) < min_tokens and result:
            # Append this node's text to the previous sibling
            prev = result[-1]
            prev["text"] = prev.get("text", "") + "\n" + node.get("text", "")
            prev["token_count"] = prev.get("token_count", 0) + node.get("token_count", 0)
        else:
            result.append(dict(node))
    return result


def build_tree_from_nodes(nodes: list[dict]) -> list[dict]:
    """
    Converts a flat list of nodes (with 'level' field) into a hierarchy
    using a stack-based algorithm.
    """
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (level, node)

    for node in nodes:
        node = {**node, "nodes": []}
        level = node["level"]

        # Pop all entries at same or deeper level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            roots.append(node)

        stack.append((level, node))

    return roots


async def _enrich_md_tree(
    tree: list[dict],
    llm: LLMClient,
    opt: PageIndexOptions,
    logger: Optional[logging.Logger],
):
    if opt.if_add_node_id:
        write_node_id(tree)
    if opt.if_add_node_summary:
        logger.info("Generating node summaries for markdown tree...")
        await generate_summaries(tree, llm, opt.summary_token_threshold)
        if not opt.if_add_node_text:
            _strip_text(tree)
    doc_description = ""
    if opt.if_add_doc_description:
        doc_description = await generate_doc_description(tree, llm)
    return doc_description


async def md_to_tree_async(
    md_path: str,
    llm: LLMClient,
    opt: Optional[PageIndexOptions] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    if opt is None:
        opt = PageIndexOptions()
    if logger is None:
        logger = logging.getLogger("pageindex")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    md_lines = md_text.splitlines()

    logger.info(f"Parsing markdown: {md_path}")
    nodes = extract_nodes_from_markdown(md_text)
    nodes = extract_node_text_content(nodes, md_lines)
    nodes = thin_nodes(nodes, opt.min_node_token)
    tree = build_tree_from_nodes(nodes)
    doc_description = await _enrich_md_tree(tree, llm, opt, logger)

    import os
    return {
        "doc_name": os.path.basename(md_path),
        "doc_description": doc_description,
        "structure": tree,
    }


def md_to_tree(
    md_path: str,
    llm: LLMClient,
    opt: Optional[PageIndexOptions] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Synchronous wrapper for md_to_tree_async."""
    return asyncio.run(md_to_tree_async(md_path, llm, opt, logger))
