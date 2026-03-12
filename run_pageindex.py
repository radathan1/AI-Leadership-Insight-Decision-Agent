#!/usr/bin/env python3
"""
run_pageindex.py
----------------
Command-line interface for the PageIndex reimplementation.

Usage (PDF):
    python run_pageindex.py --pdf_path document.pdf --api_key sk-... --model gpt-4o

Usage (Markdown):
    python run_pageindex.py --md_path document.md --api_key sk-...

Usage (local Ollama):
    python run_pageindex.py --pdf_path doc.pdf --api_key ollama \
        --base_url http://localhost:11434/v1 --model llama3.1

Usage (query / search):
    python run_pageindex.py --pdf_path doc.pdf --api_key sk-... \
        --query "What is the revenue for Q3?"

The generated index is saved as <filename>_pageindex.json.
If --query is provided, the tool loads the saved index and runs a tree search.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from page_index import (
    LLMClient,
    PageIndexOptions,
    page_index_async,
    md_to_tree_async,
    extract_pages,
)
from tree_search import TreeSearcher
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()


def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("pageindex")


async def run_index(args, logger):
    llm = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url or None,
    )

    opt = PageIndexOptions(
        model=args.model,
        toc_check_page_num=args.toc_check_pages,
        max_page_num_each_node=args.max_pages_per_node,
        max_token_num_each_node=args.max_tokens_per_node,
        if_add_node_id=args.add_node_id,
        if_add_node_summary=args.add_node_summary,
        if_add_doc_description=args.add_doc_description,
        if_add_node_text=args.add_node_text,
    )

    if args.pdf_path:
        result = await page_index_async(args.pdf_path, llm, opt, logger)
        base = os.path.splitext(args.pdf_path)[0]
    elif args.md_path:
        result = await md_to_tree_async(args.md_path, llm, opt, logger)
        base = os.path.splitext(args.md_path)[0]
    else:
        logger.error("Provide --pdf_path or --md_path")
        sys.exit(1)

    out_path = f"{base}_pageindex.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"PageIndex saved to: {out_path}")
    return result, out_path


async def run_query(args, logger):
    # Load existing index or build it first
    if args.pdf_path:
        base = os.path.splitext(args.pdf_path)[0]
    elif args.md_path:
        base = os.path.splitext(args.md_path)[0]
    else:
        logger.error("Provide --pdf_path or --md_path to load/build an index.")
        sys.exit(1)

    index_path = f"{base}_pageindex.json"

    if not os.path.exists(index_path):
        logger.info("No existing index found. Building index first...")
        result, _ = await run_index(args, logger)
    else:
        logger.info(f"Loading existing index from {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    tree = result.get("structure", [])

    # Load page list for text retrieval (PDF only)
    page_list = []
    if args.pdf_path and os.path.exists(args.pdf_path):
        logger.info("Loading PDF pages for text retrieval...")
        page_list = extract_pages(args.pdf_path)

    llm = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url or None,
    )

    searcher = TreeSearcher(llm, logger=logger)
    logger.info(f"Searching: {args.query}")
    search_result = await searcher.search(args.query, tree, page_list)

    print("\n" + "=" * 60)
    print(f"QUERY: {search_result.query}")
    print("=" * 60)
    print(f"\nANSWER:\n{search_result.answer}")
    print(f"\nRELEVANT PAGES: {search_result.relevant_pages}")
    print("\nREASONING TRACE:")
    for step in search_result.reasoning_trace:
        print(f"  • {step}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="PageIndex: vectorless, reasoning-based RAG"
    )

    # Input
    parser.add_argument("--pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--md_path", type=str, help="Path to Markdown file")

    # LLM settings
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (default: gpt-4o)")
    parser.add_argument("--base_url", type=str, default="",
                        help="Custom API base URL (e.g. http://localhost:11434/v1 for Ollama)")

    # Index options
    parser.add_argument("--toc-check-pages", type=int, default=20)
    parser.add_argument("--max-pages-per-node", type=int, default=50)
    parser.add_argument("--max-tokens-per-node", type=int, default=50000)
    parser.add_argument("--add-node-id", action="store_true", default=True)
    parser.add_argument("--add-node-summary", action="store_true", default=True)
    parser.add_argument("--add-doc-description", action="store_true", default=True)
    parser.add_argument("--add-node-text", action="store_true", default=False)

    # Query mode
    parser.add_argument("--query", type=str, default="",
                        help="If provided, run a tree search with this query")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    if not args.api_key:
        logger.error("No API key provided. Use --api_key or set OPENAI_API_KEY env var.")
        sys.exit(1)

    if args.query:
        asyncio.run(run_query(args, logger))
    else:
        asyncio.run(run_index(args, logger))


if __name__ == "__main__":
    main()
