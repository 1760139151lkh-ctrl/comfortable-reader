"""Tiny real MCP 2026-07-28 stdio server over two bounded book tasks.

Launched only by c41_mcp_evidence_client.py. The SDK handles protocol framing.
This server does not authenticate a human user or grant OS permissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
RETRIEVAL_REPORT = WORK / "runs/c40_book_retrieval_stable_a/report.json"
RETRIEVAL_SHA = "47a31f6459b8fc0339727210c51026a955c4cccc8b82b2bd0fbfbf881ffff161"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(note_path: Path, expected_sha: str) -> MCPServer:
    note = note_path.resolve()
    if not note.is_relative_to(RUNS.resolve()) or len(expected_sha) != 64:
        raise ValueError("note must be inside work/runs and bound to a SHA-256")
    mcp = MCPServer("C41 local evidence desk")

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two integers from 0 through 1000; no external side effect."""
        if not (0 <= a <= 1000 and 0 <= b <= 1000):
            raise ToolError("both integers must be between 0 and 1000")
        return a * b

    @mcp.tool()
    def read_note() -> str:
        """Read only the preselected teaching note if its bytes still match."""
        if not note.is_file() or sha(note) != expected_sha:
            raise ToolError("note source changed; recheck it before use")
        return note.read_text(encoding="utf-8")

    @mcp.resource("evidence://c40/retrieval-summary")
    def retrieval_summary() -> str:
        """Return a compact, version-bound summary of the local C40 run."""
        if sha(RETRIEVAL_REPORT) != RETRIEVAL_SHA:
            raise ValueError("C40 report changed")
        r = json.loads(RETRIEVAL_REPORT.read_text(encoding="utf-8"))
        return json.dumps({"paragraphs": r["paragraphs"], "gold_questions": r["gold_questions"],
                           "recall_at_5": r["recall_at_5"], "source_sha256": RETRIEVAL_SHA},
                          ensure_ascii=False)

    @mcp.prompt()
    def check_note(question: str) -> str:
        """Ask for an answer grounded in the named note, without granting access."""
        return ("Read the permitted note source, cite its current SHA-256, and answer only what "
                f"it supports. If the source changed, stop. Question: {question}")

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    build(args.note, args.expected_sha).run(transport="stdio")
