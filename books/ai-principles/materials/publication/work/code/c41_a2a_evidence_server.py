"""A2A v1.0 loopback server with a deterministic teaching executor.

One mode reads an actual C40 report; the other intentionally gives a plausible
unsupported count. Neither process runs a language model or handles user data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import uvicorn
from a2a.helpers import new_task_from_user_message, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (AgentCapabilities, AgentCard, AgentInterface, AgentSkill,
                       TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent)
from starlette.applications import Starlette

WORK = Path(__file__).resolve().parents[1]
REPORT = WORK / "runs/c40_book_retrieval_stable_a/report.json"
REPORT_SHA = "47a31f6459b8fc0339727210c51026a955c4cccc8b82b2bd0fbfbf881ffff161"


class EvidenceExecutor(AgentExecutor):
    def __init__(self, mode: str):
        self.mode = mode

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=context.task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING,
                              message=new_text_message("Checking the assigned source."))))
        if self.mode == "source_reader":
            actual_sha = hashlib.sha256(REPORT.read_bytes()).hexdigest()
            if actual_sha != REPORT_SHA:
                raise RuntimeError("C40 source report changed")
            r = json.loads(REPORT.read_text(encoding="utf-8"))
            result = {"r_at_5_count": r["recall_at_5"], "question_count": r["gold_questions"],
                      "source_sha256": actual_sha, "source": "C40 published-book retrieval report"}
        else:
            result = {"r_at_5_count": 12, "question_count": 12,
                      "source_sha256": "unverified", "source": "plausible assertion without reading"}
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=context.task_id, context_id=context.context_id,
            artifact=new_text_artifact(name="retrieval_result", text=json.dumps(result, ensure_ascii=False))))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=context.task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise RuntimeError("these short deterministic tasks do not support mid-flight cancel")


def application(mode: str, port: int) -> Starlette:
    skill = AgentSkill(id="c40_retrieval_count", name="Report a C40 retrieval count",
                       description="Return a count about the fixed C40 teaching report.",
                       tags=["textbook", "retrieval"], examples=["What is C40 R@5?"])
    card = AgentCard(name=f"C41 {mode}", description="Loopback teaching executor; no LLM or external account",
                     version="1.0.0", default_input_modes=["text/plain"],
                     default_output_modes=["application/json"],
                     capabilities=AgentCapabilities(streaming=False),
                     supported_interfaces=[AgentInterface(protocol_binding="JSONRPC",
                                                          protocol_version="1.0",
                                                          url=f"http://127.0.0.1:{port}/")],
                     skills=[skill])
    handler = DefaultRequestHandler(agent_executor=EvidenceExecutor(mode),
                                    task_store=InMemoryTaskStore(), agent_card=card)
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
    return Starlette(routes=routes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_reader", "unverified"), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(application(args.mode, args.port), host="127.0.0.1", port=args.port,
                log_level="warning", access_log=False)
