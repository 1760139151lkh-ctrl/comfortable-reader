"""Launch the C41 server as a real stdio MCP subprocess and inspect results."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
ORIGINAL = WORK / "runs/c39_host_lab_inspectable/final_note.txt"
ORIGINAL_SHA = "e18ebeed0a9d7a3855c5b6511011c355bc045d11402751144c0c3a5f8b6cd056"
SERVER = WORK / "code/c41_mcp_evidence_server.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def demo(out: Path) -> None:
    out = out.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs output directory")
    if sha(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("the C39 original note changed")
    out.mkdir(parents=True)
    copy = out / "note_copy.txt"
    shutil.copyfile(ORIGINAL, copy)
    assert sha(copy) == ORIGINAL_SHA
    spec = StdioServerParameters(command=sys.executable,
                                 args=[str(SERVER), "--note", str(copy), "--expected-sha", ORIGINAL_SHA],
                                 cwd=str(WORK.parent))
    async with Client(stdio_client(spec), mode="auto") as client:
        version = client.protocol_version
        capabilities = client.server_capabilities.model_dump(exclude_none=True)
        listed = await client.list_tools()
        tool_names = [tool.name for tool in listed.tools]
        assert "multiply" in tool_names and "read_note" in tool_names
        multiply = await client.call_tool("multiply", {"a": 12, "b": 13})
        bad_args = await client.call_tool("multiply", {"a": "twelve", "b": 13})
        read_before = await client.call_tool("read_note")
        resources = await client.list_resources()
        resource = await client.read_resource("evidence://c40/retrieval-summary")
        prompts = await client.list_prompts()
        prompt = await client.get_prompt("check_note", {"question": "What is the final line?"})
        # Only the teaching copy changes; the original C39 file remains untouched.
        copy.write_bytes(copy.read_bytes() + b"altered demonstration copy\n")
        read_after = await client.call_tool("read_note")
    data = {
        "sdk_mcp_version": "2.0.0",
        "protocol_version": version,
        "discovered_capabilities": capabilities,
        "tool_names": tool_names,
        "multiply_12_13": multiply.structured_content,
        "wrong_type_is_error": bad_args.is_error,
        "note_before": read_before.content[0].text,
        "resource_names": [str(item.uri) for item in resources.resources],
        "retrieval_resource": resource.contents[0].text,
        "prompt_names": [item.name for item in prompts.prompts],
        "prompt_text": prompt.messages[0].content.text,
        "note_after_copy_change_is_error": read_after.is_error,
        "note_after_copy_change_text": read_after.content[0].text,
        "original_c39_sha256_unchanged": sha(ORIGINAL) == ORIGINAL_SHA,
        "copy_sha256_after_change": sha(copy),
        "scope": "real MCP Python SDK v2 stdio subprocess and current protocol; client calls chosen by author, no model trained/used to choose tools, no remote OAuth or OS sandbox",
    }
    assert version == "2026-07-28"
    assert not multiply.is_error and str(multiply.structured_content.get("result")) == "156"
    assert data["wrong_type_is_error"] and data["note_after_copy_change_is_error"]
    assert data["original_c39_sha256_unchanged"]
    (out / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: data[k] for k in ("sdk_mcp_version", "protocol_version", "tool_names",
                                           "multiply_12_13", "wrong_type_is_error",
                                           "note_after_copy_change_is_error", "original_c39_sha256_unchanged")},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(demo(args.out_dir))
