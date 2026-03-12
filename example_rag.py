"""
example_rag.py
--------------
Minimal, end-to-end example of vectorless RAG using the PageIndex reimplementation.

This mirrors the logic in VectifyAI's cookbook/pageindex_RAG_simple.ipynb but
uses YOUR own API key (OpenAI, Anthropic-compat, Ollama, etc.).

Steps:
  1. Index a PDF  → produces a hierarchical tree JSON
  2. Ask a question  → tree search returns relevant pages + LLM-generated answer

Run:
    python example_rag.py --pdf your_doc.pdf --query "What is the main conclusion?"
"""

import asyncio
import json
import os
import sys
import argparse

# Add parent dir so imports work when run directly
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

from page_index import (
    LLMClient,
    PageIndexOptions,
    page_index_async,
    extract_pages,
)
from tree_search import TreeSearcher


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--force_reindex", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Set --api_key or OPENAI_API_KEY env var.")
        sys.exit(1)

    llm = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url or None,
    )

    # -------------------------------------------------------------------------
    # Step 1: Build or load index
    # -------------------------------------------------------------------------
    index_path = args.pdf.replace(".pdf", "_pageindex.json")

    if os.path.exists(index_path) and not args.force_reindex:
        print(f"✅ Loading cached index: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
    else:
        print(f"📄 Building PageIndex for: {args.pdf}")
        opt = PageIndexOptions(
            model=args.model,
            if_add_node_summary=True,
            if_add_doc_description=True,
        )
        index = await page_index_async(args.pdf, llm, opt)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"✅ Index saved: {index_path}")

    # Print tree summary
    print("\n📑 Document structure (top level):")
    for node in index.get("structure", [])[:8]:
        nid = node.get("node_id", "?")
        title = node.get("title", "Untitled")
        pages = f"pp. {node.get('start_index')}-{node.get('end_index')}"
        summary = node.get("summary", "")[:80]
        print(f"  [{nid}] {title}  ({pages})")
        if summary:
            print(f"       → {summary}")

    # -------------------------------------------------------------------------
    # Step 2: Tree search
    # -------------------------------------------------------------------------
    print(f"\n🔍 Query: {args.query}\n")
    tree = index.get("structure", [])
    page_list = extract_pages(args.pdf)

    searcher = TreeSearcher(llm, max_depth=4)
    result = await searcher.search(args.query, tree, page_list)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()
    print(f"📌 Relevant pages: {result.relevant_pages}")
    print("\n🧠 Reasoning trace:")
    for step in result.reasoning_trace:
        print(f"  • {step}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
