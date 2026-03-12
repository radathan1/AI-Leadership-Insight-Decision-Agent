"""
tree_search.py
--------------
Reasoning-based retrieval over a PageIndex tree structure.

This module implements the "tree search" phase of PageIndex:
given a query and a tree-indexed document, use an LLM to reason
over the tree hierarchy and identify the most relevant pages/sections,
without any vector similarity search.

Usage:
    from tree_search import TreeSearcher
    searcher = TreeSearcher(llm_client)
    result = await searcher.search(query, tree_structure, page_list)
    # result.relevant_pages, result.relevant_text, result.reasoning_trace
"""

import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from page_index import LLMClient, count_tokens


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_TREE_SEARCH_ROOT = """You are an expert at navigating hierarchical document structures to find relevant information.

Given a user QUERY and a document tree structure, identify which top-level sections are MOST LIKELY to contain the answer.

Query: {query}

Document tree (top-level sections with summaries):
{tree_summary}

Return ONLY a JSON object with:
  "reasoning": "brief explanation of which sections are relevant and why"
  "relevant_node_ids": ["0001", "0002", ...]  (list of node_ids to explore further)
  "confidence": "high" | "medium" | "low"
"""

PROMPT_TREE_SEARCH_DRILL = """You are drilling into a document section to find the most relevant subsections for a query.

Query: {query}

Current section: "{section_title}"
Subsections available:
{subsection_summary}

Return ONLY a JSON object with:
  "reasoning": "brief explanation"
  "relevant_node_ids": ["0003", "0004", ...]  (node_ids of subsections to explore)
  "answer_likely_here": true | false  (true if this section itself likely contains the answer)
"""

PROMPT_EXTRACT_ANSWER = """You are answering a question using relevant document sections.

Query: {query}

Relevant document sections:
{context}

Provide a clear, accurate answer based solely on the provided sections.
If the answer cannot be determined from the text, say so explicitly.
Include page references (e.g., "Page 5") where relevant.
"""

PROMPT_NODE_RELEVANCE = """Rate the relevance of the following document section to the query.

Query: {query}

Section title: {title}
Section summary: {summary}

Return ONLY a JSON object:
  "relevance_score": 0-10 (10 = highly relevant)
  "reasoning": "brief explanation"
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    query: str
    answer: str
    relevant_nodes: list[dict] = field(default_factory=list)
    relevant_pages: list[int] = field(default_factory=list)
    relevant_text: str = ""
    reasoning_trace: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def _get_node_by_id(nodes: list[dict], node_id: str) -> Optional[dict]:
    for node in nodes:
        if node.get("node_id") == node_id:
            return node
        found = _get_node_by_id(node.get("nodes", []), node_id)
        if found:
            return found
    return None


def _node_summary_text(node: dict) -> str:
    """Returns a compact summary string for a node."""
    title = node.get("title", "Untitled")
    summary = node.get("summary") or node.get("prefix_summary") or ""
    start = node.get("start_index", "?")
    end = node.get("end_index", "?")
    node_id = node.get("node_id", "?")
    child_count = len(node.get("nodes", []))
    s = f"[{node_id}] {title} (pages {start}-{end})"
    if summary:
        s += f"\n    Summary: {summary[:200]}"
    if child_count:
        s += f"\n    Has {child_count} subsection(s)"
    return s


def _flatten_to_nodes(nodes: list[dict]) -> list[dict]:
    """Depth-first flattening of all nodes."""
    result = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_to_nodes(node.get("nodes", [])))
    return result


def _get_page_range_text(
    page_list: list[tuple[str, int]], start: int, end: int, max_tokens: int = 6000
) -> str:
    """Extract and concatenate page texts within token budget."""
    parts = []
    used = 0
    for i in range(start - 1, min(end, len(page_list))):
        text, tok = page_list[i]
        if used + tok > max_tokens:
            remaining = max_tokens - used
            # Add partial text
            enc_text = text[: int(len(text) * remaining / max(tok, 1))]
            parts.append(f"[Page {i+1}]\n{enc_text}")
            break
        parts.append(f"[Page {i+1}]\n{text}")
        used += tok
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# TreeSearcher
# ---------------------------------------------------------------------------

class TreeSearcher:
    """
    Performs reasoning-based retrieval over a PageIndex tree.

    The algorithm:
    1. Show top-level node summaries to the LLM → select relevant branches
    2. For each selected branch, recurse into its children
    3. Collect leaf pages when no further children OR node is directly relevant
    4. Synthesize final answer from collected page texts
    """

    def __init__(
        self,
        llm: LLMClient,
        max_depth: int = 4,
        max_pages_per_answer: int = 10,
        logger: Optional[logging.Logger] = None,
        on_trace: Optional[callable] = None,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_pages_per_answer = max_pages_per_answer
        self.logger = logger or logging.getLogger("pageindex.search")
        # Optional callback to receive incremental reasoning trace strings.
        # If provided it may be a coroutine function or a normal callable.
        self.on_trace = on_trace

    async def search(
        self,
        query: str,
        tree: list[dict],
        page_list: list[tuple[str, int]],
        conversation: Optional[list[dict]] = None,
    ) -> SearchResult:
        """
        Main entry point. Returns a SearchResult with the answer and
        the nodes/pages that were used.
        """
        result = SearchResult(query=query, answer="", reasoning_trace=[])

        # Phase 1: Tree traversal to find relevant nodes
        relevant_nodes = await self._traverse(query, tree, depth=0, trace=result.reasoning_trace)
        result.relevant_nodes = relevant_nodes

        # Phase 2: Collect page ranges
        all_pages: list[int] = []
        for node in relevant_nodes:
            start = node.get("start_index", 1)
            end = node.get("end_index", len(page_list))
            for p in range(start, min(end + 1, len(page_list) + 1)):
                if p not in all_pages:
                    all_pages.append(p)
        all_pages = sorted(all_pages)[: self.max_pages_per_answer]
        result.relevant_pages = all_pages

        # Phase 3: Build context from pages
        if page_list and all_pages:
            context_parts = []
            for p in all_pages:
                if 1 <= p <= len(page_list):
                    context_parts.append(f"[Page {p}]\n{page_list[p-1][0]}")
            result.relevant_text = "\n\n---\n\n".join(context_parts)
        else:
            result.relevant_text = ""

        # Phase 4: Generate answer (include conversation history if provided)
        if result.relevant_text:
            result.answer = await self._generate_answer(query, result.relevant_text, conversation)
        else:
            result.answer = "No relevant sections found for this query."

        return result

    async def _traverse(
        self,
        query: str,
        nodes: list[dict],
        depth: int,
        trace: list[str],
    ) -> list[dict]:
        """
        Recursively traverse the tree, selecting relevant nodes.
        Returns list of relevant leaf-level nodes.
        """
        if not nodes or depth > self.max_depth:
            return []

        # Build summary of current level
        summary_lines = [_node_summary_text(n) for n in nodes]
        tree_summary = "\n\n".join(summary_lines)

        if depth == 0:
            prompt = PROMPT_TREE_SEARCH_ROOT.format(
                query=query, tree_summary=tree_summary
            )
        else:
            parent_title = nodes[0].get("title", "section") if nodes else "section"
            sub_summary = "\n\n".join(
                _node_summary_text(n) for n in nodes
            )
            prompt = PROMPT_TREE_SEARCH_DRILL.format(
                query=query,
                section_title=parent_title,
                subsection_summary=sub_summary,
            )

        try:
            llm_result = await self.llm.chat_json(prompt, max_tokens=800)
            reasoning = llm_result.get("reasoning", "")
            selected_ids = llm_result.get("relevant_node_ids", [])
            answer_here = llm_result.get("answer_likely_here", False)

            if reasoning:
                trace_msg = f"[depth={depth}] {reasoning}"
                trace.append(trace_msg)
                # Emit incremental trace if callback provided
                if self.on_trace:
                    try:
                        if asyncio.iscoroutinefunction(self.on_trace):
                            await self.on_trace(trace_msg)
                        else:
                            # run non-async callback in thread pool to avoid blocking loop
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(None, self.on_trace, trace_msg)
                    except Exception:
                        self.logger.exception("on_trace callback failed")

            # Map selected IDs to nodes
            id_to_node = {n.get("node_id"): n for n in nodes}
            selected = [id_to_node[nid] for nid in selected_ids if nid in id_to_node]

            if not selected:
                # Fallback: take all nodes at this level if LLM returned nothing
                selected = nodes

        except Exception as e:
            self.logger.warning(f"Tree search LLM call failed at depth {depth}: {e}")
            selected = nodes  # Fallback: use all nodes
            answer_here = True
            # Emit failure trace
            fail_msg = f"[depth={depth}] LLM call failed: {e}"
            trace.append(fail_msg)
            if self.on_trace:
                try:
                    if asyncio.iscoroutinefunction(self.on_trace):
                        await self.on_trace(fail_msg)
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self.on_trace, fail_msg)
                except Exception:
                    self.logger.exception("on_trace callback failed")

        # Recurse into children
        relevant: list[dict] = []
        for node in selected:
            children = node.get("nodes", [])
            if children and depth < self.max_depth:
                child_results = await self._traverse(query, children, depth + 1, trace)
                relevant.extend(child_results)
            else:
                relevant.append(node)

        return relevant

    async def _generate_answer(self, query: str, context: str, conversation: Optional[list[dict]] = None) -> str:
        """Generate a final answer from the retrieved context and optional conversation history."""
        # Truncate context if too long
        max_ctx_tokens = 12000
        if count_tokens(context) > max_ctx_tokens:
            context = context[:max_ctx_tokens * 4]  # rough char estimate
        # Build conversation string if provided
        conv_text = ""
        if conversation:
            # Use last 10 messages
            last_msgs = conversation[-10:]
            parts = []
            for m in last_msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role.capitalize()}: {content}")
            conv_text = "\n".join(parts)

        if conv_text:
            prompt = PROMPT_EXTRACT_ANSWER.format(query=query, context=f"Conversation history:\n{conv_text}\n\nRelevant document sections:\n{context}")
        else:
            prompt = PROMPT_EXTRACT_ANSWER.format(query=query, context=context)
        try:
            answer, _ = await self.llm.chat(prompt, max_tokens=1000)
            return answer.strip()
        except Exception as e:
            return f"Answer generation failed: {e}"


# ---------------------------------------------------------------------------
# Convenience: quick search without async boilerplate
# ---------------------------------------------------------------------------

def search(
    query: str,
    tree: list[dict],
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    max_depth: int = 4,
) -> SearchResult:
    """Synchronous convenience wrapper."""
    searcher = TreeSearcher(llm, max_depth=max_depth)
    return asyncio.run(searcher.search(query, tree, page_list))
