"""Two real local A2A v1.0 endpoints; the author client verifies artifacts.

The two servers are deterministic role examples, not spawned LLM subagents.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
SERVER = WORK / "code/c41_a2a_evidence_server.py"
REPORT = WORK / "runs/c40_book_retrieval_stable_a/report.json"
REPORT_SHA = "47a31f6459b8fc0339727210c51026a955c4cccc8b82b2bd0fbfbf881ffff161"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_card(http: httpx.AsyncClient, base: str, proc: subprocess.Popen):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("A2A server exited before card became available")
        try:
            response = await http.get(base + "/.well-known/agent-card.json", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("A2A card did not become available")


async def ask_one(http: httpx.AsyncClient, base: str):
    resolver = A2ACardResolver(httpx_client=http, base_url=base)
    card = await resolver.get_agent_card()
    client = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=http))
    request = SendMessageRequest(message=new_text_message("What is C40 R@5?", role=Role.ROLE_USER))
    chunks = [item async for item in client.send_message(request)]
    if len(chunks) != 1 or not chunks[0].HasField("task"):
        raise RuntimeError("expected one final A2A Task")
    task = chunks[0].task
    if task.status.state != TaskState.TASK_STATE_COMPLETED or len(task.artifacts) != 1:
        raise RuntimeError("task did not finish with one artifact")
    payload = json.loads(task.artifacts[0].parts[0].text)
    return {"card_name": card.name,
            "card_skill_ids": [skill.id for skill in card.skills],
            "interface_version": card.supported_interfaces[0].protocol_version,
            "task_id": task.id, "task_state": "TASK_STATE_COMPLETED",
            "artifact_name": task.artifacts[0].name, "artifact": payload}


async def demo(out_dir: Path):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs output directory")
    if sha(REPORT) != REPORT_SHA:
        raise RuntimeError("C40 report changed")
    out.mkdir(parents=True)
    ports = [free_port(), free_port()]
    if ports[0] == ports[1]:
        raise RuntimeError("port collision")
    modes = ("source_reader", "unverified")
    procs = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for mode, port in zip(modes, ports):
            procs.append(subprocess.Popen([sys.executable, str(SERVER), "--mode", mode, "--port", str(port)],
                                          cwd=str(WORK.parent), stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                          creationflags=flags))
        async with httpx.AsyncClient(timeout=5) as http:
            for port, proc in zip(ports, procs):
                await wait_card(http, f"http://127.0.0.1:{port}", proc)
            replies = [await ask_one(http, f"http://127.0.0.1:{port}") for port in ports]
        truth = json.loads(REPORT.read_text(encoding="utf-8"))
        accepted = [r for r in replies if r["artifact"].get("source_sha256") == REPORT_SHA
                    and r["artifact"].get("r_at_5_count") == truth["recall_at_5"]]
        rejected = [r for r in replies if r not in accepted]
        assert len(accepted) == len(rejected) == 1
        report = {"sdk_version": "1.0.2", "protocol_version": "1.0",
                  "two_card_declared_same_skill": replies[0]["card_skill_ids"] == replies[1]["card_skill_ids"],
                  "request_budget": 2, "replies": replies,
                  "accepted_task_id": accepted[0]["task_id"], "rejected_task_id": rejected[0]["task_id"],
                  "source_count": truth["recall_at_5"], "source_report_sha256": REPORT_SHA,
                  "reason_rejected": "protocol completed and card advertised skill, but artifact count and source SHA did not match the real report",
                  "scope": "two local deterministic A2A SDK servers over loopback; no LLM worker, no external delegation, no auth or durable task store"}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"version": report["protocol_version"],
                          "same_card_skill": report["two_card_declared_same_skill"],
                          "completed_tasks": len(replies), "accepted_artifacts": len(accepted),
                          "rejected_artifacts": len(rejected), "request_budget": 2}, ensure_ascii=False))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            if proc.stderr is not None:
                proc.stderr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(demo(args.out_dir))
