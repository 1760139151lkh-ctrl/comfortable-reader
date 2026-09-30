"""C39: bounded local host execution with receipts and two failure worlds.

This lab never edits user documents or calls remote services. The only
side-effect target is a fresh SQLite teaching ledger under work/runs. Its
application checks are not an operating-system sandbox or an authorization
system for untrusted arbitrary Python.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
SCHEMA = """
CREATE TABLE notes (doc_id TEXT PRIMARY KEY, content TEXT NOT NULL, version INTEGER NOT NULL);
CREATE TABLE grants (subject TEXT PRIMARY KEY, tool TEXT NOT NULL, doc_id TEXT NOT NULL,
                     remaining INTEGER NOT NULL, max_text_bytes INTEGER NOT NULL);
CREATE TABLE operations (op_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL,
                         receipt_json TEXT NOT NULL);
CREATE TABLE cancellations (op_id TEXT PRIMARY KEY, reason TEXT NOT NULL);
CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, op_id TEXT NOT NULL,
                     event TEXT NOT NULL, detail_json TEXT NOT NULL);
"""


class Denied(Exception):
    pass


class KeyConflict(Exception):
    pass


class StaleVersion(Exception):
    pass


class Cancelled(Exception):
    pass


def safe_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(RUNS.resolve()):
        raise ValueError("C39 lab paths must stay inside work/runs")
    return resolved


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def connect(db: Path):
    db = safe_path(db)
    c = sqlite3.connect(db, timeout=5, isolation_level=None)
    c.execute("PRAGMA busy_timeout=5000")
    c.execute("PRAGMA synchronous=FULL")
    return c


def init(db: Path, budget=3):
    db = safe_path(db)
    if db.exists():
        raise RuntimeError("database exists")
    db.parent.mkdir(parents=True, exist_ok=True)
    with connect(db) as c:
        c.executescript(SCHEMA)
        c.execute("INSERT INTO notes VALUES (?,?,?)", ("lesson_note", "Start\n", 0))
        c.execute("INSERT INTO grants VALUES (?,?,?,?,?)", ("author_demo", "append_note", "lesson_note", budget, 80))


def validate(req: dict):
    if not isinstance(req, dict) or set(req) != {"subject", "tool", "doc_id", "text", "expected_version", "op_id"}:
        raise Denied("request must contain exactly the six defined fields")
    if not isinstance(req["subject"], str) or not isinstance(req["tool"], str) or not isinstance(req["doc_id"], str):
        raise Denied("subject/tool/document must be strings")
    if not isinstance(req["text"], str) or not 1 <= len(req["text"].encode("utf-8")) <= 80 or "\x00" in req["text"]:
        raise Denied("text must be 1..80 UTF-8 bytes and contain no NUL")
    if isinstance(req["expected_version"], bool) or not isinstance(req["expected_version"], int) or req["expected_version"] < 0:
        raise Denied("expected_version must be a nonnegative integer")
    if not isinstance(req["op_id"], str) or re.fullmatch(r"op-[a-z0-9-]{1,40}", req["op_id"]) is None:
        raise Denied("op_id must be an application-issued logical operation ID")


def event(c, op_id, name, detail):
    c.execute("INSERT INTO events(op_id,event,detail_json) VALUES (?,?,?)", (op_id, name, canonical(detail)))


def apply(db: Path, req: dict, fault=None, marker=None):
    """The subject and operation ID are supplied by the *host*, never model text."""
    validate(req)
    payload = {key: req[key] for key in ("subject", "tool", "doc_id", "text", "expected_version")}
    payload_sha = sha_bytes(canonical(payload).encode("utf-8"))
    c = connect(db)
    try:
        c.execute("BEGIN IMMEDIATE")
        old = c.execute("SELECT payload_sha256,receipt_json FROM operations WHERE op_id=?", (req["op_id"],)).fetchone()
        if old:
            if old[0] != payload_sha:
                raise KeyConflict("same operation ID has different parameters")
            receipt = json.loads(old[1])
            c.execute("COMMIT")
            return {"receipt": receipt, "replayed": True}
        if c.execute("SELECT 1 FROM cancellations WHERE op_id=?", (req["op_id"],)).fetchone():
            raise Cancelled("operation was cancelled before dispatch")
        grant = c.execute("SELECT tool,doc_id,remaining,max_text_bytes FROM grants WHERE subject=?", (req["subject"],)).fetchone()
        if not grant or req["tool"] != grant[0] or req["doc_id"] != grant[1]:
            raise Denied("tool or document not in this subject's host grant")
        if len(req["text"].encode("utf-8")) > grant[3]:
            raise Denied("text exceeds host grant")
        row = c.execute("SELECT content,version FROM notes WHERE doc_id=?", (req["doc_id"],)).fetchone()
        if not row:
            raise Denied("unknown note")
        if req["expected_version"] != row[1]:
            raise StaleVersion(f"expected {req['expected_version']}, current {row[1]}")
        if grant[2] <= 0:
            raise Denied("write budget exhausted")
        content = row[0] + req["text"]
        next_version = row[1] + 1
        receipt = {"op_id": req["op_id"], "subject": req["subject"], "tool": req["tool"],
                   "doc_id": req["doc_id"], "before_version": row[1], "after_version": next_version,
                   "content_sha256": sha_bytes(content.encode("utf-8")), "status": "committed"}
        c.execute("UPDATE notes SET content=?,version=? WHERE doc_id=?", (content, next_version, req["doc_id"]))
        c.execute("UPDATE grants SET remaining=remaining-1 WHERE subject=?", (req["subject"],))
        c.execute("INSERT INTO operations VALUES (?,?,?)", (req["op_id"], payload_sha, canonical(receipt)))
        event(c, req["op_id"], "committed", {"version": next_version, "content_sha256": receipt["content_sha256"]})
        if fault == "pause_before_commit":
            Path(marker).write_text("inside transaction, before COMMIT\n", encoding="utf-8")
            time.sleep(60)
        c.execute("COMMIT")
        if fault == "pause_after_commit":
            Path(marker).write_text("COMMIT finished, before reply\n", encoding="utf-8")
            time.sleep(60)
        return {"receipt": receipt, "replayed": False}
    except BaseException:
        if c.in_transaction:
            c.execute("ROLLBACK")
        raise
    finally:
        c.close()


def cancel(db: Path, op_id: str):
    c = connect(db)
    try:
        c.execute("BEGIN IMMEDIATE")
        existing = c.execute("SELECT receipt_json FROM operations WHERE op_id=?", (op_id,)).fetchone()
        if existing:
            c.execute("COMMIT")
            return {"status": "already_committed", "receipt": json.loads(existing[0])}
        c.execute("INSERT OR IGNORE INTO cancellations VALUES (?,?)", (op_id, "caller_requested_before_dispatch"))
        event(c, op_id, "cancel_before_dispatch", {})
        c.execute("COMMIT")
        return {"status": "cancelled_before_dispatch"}
    finally:
        c.close()


def snapshot(db: Path):
    with connect(db) as c:
        note = c.execute("SELECT content,version FROM notes WHERE doc_id='lesson_note'").fetchone()
        grant = c.execute("SELECT remaining FROM grants WHERE subject='author_demo'").fetchone()
        ops = c.execute("SELECT op_id,receipt_json FROM operations ORDER BY rowid").fetchall()
        events = c.execute("SELECT seq,op_id,event,detail_json FROM events ORDER BY seq").fetchall()
    return {"content": note[0], "version": note[1], "content_sha256": sha_bytes(note[0].encode("utf-8")),
            "remaining_write_budget": grant[0], "operations": [{"op_id": k, "receipt": json.loads(v)} for k, v in ops],
            "events": [{"seq": seq, "op_id": op, "event": name, "detail": json.loads(detail)} for seq, op, name, detail in events]}


def request(op_id, text, expected_version, *, subject="author_demo", tool="append_note", doc_id="lesson_note"):
    return {"subject": subject, "tool": tool, "doc_id": doc_id, "text": text,
            "expected_version": expected_version, "op_id": op_id}


def expect_failure(callback, error_type):
    try:
        callback()
    except error_type as error:
        return {"rejected": type(error).__name__, "reason": str(error)}
    raise AssertionError(f"expected {error_type.__name__}")


def kill_at_marker(db: Path, req: dict, out: Path, fault: str):
    req_path = out / (req["op_id"] + ".request.json")
    marker = out / (req["op_id"] + "." + fault + ".marker")
    req_path.write_text(canonical(req) + "\n", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "execute-child",
                                "--db", str(db), "--request", str(req_path), "--fault", fault,
                                "--marker", str(marker)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=flags)
    deadline = time.monotonic() + 10
    try:
        while not marker.exists():
            if process.poll() is not None:
                raise RuntimeError(f"fault child exited before marker: {process.returncode} {process.stderr.read().decode(errors='replace')}")
            if time.monotonic() > deadline:
                raise TimeoutError("fault child did not reach injection point")
            time.sleep(0.03)
        seen = marker.read_text(encoding="utf-8").strip()
        process.kill()
        process.wait(timeout=5)
        return {"fault": fault, "child_marker": seen, "caller_received_reply": False,
                "child_exit_code": process.returncode}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def init_remote(db):
    db = safe_path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        raise RuntimeError("remote simulation exists")
    with connect(db) as c:
        c.execute("CREATE TABLE effects (row_id INTEGER PRIMARY KEY AUTOINCREMENT, op_id TEXT NOT NULL, content TEXT NOT NULL, payload_sha256 TEXT NOT NULL)")


def remote_apply(db, op_id, content, dedupe):
    with connect(db) as c:
        c.execute("BEGIN IMMEDIATE")
        content_sha = sha_bytes(content.encode("utf-8"))
        old = c.execute("SELECT row_id,payload_sha256 FROM effects WHERE op_id=?", (op_id,)).fetchone() if dedupe else None
        if old:
            if old[1] != content_sha:
                raise KeyConflict("remote mock: same ID with different content")
            c.execute("COMMIT")
            return {"effect_row_id": old[0], "replayed": True}
        c.execute("INSERT INTO effects(op_id,content,payload_sha256) VALUES (?,?,?)", (op_id, content, content_sha))
        row_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute("COMMIT")
        return {"effect_row_id": row_id, "replayed": False}


def remote_count(db):
    with connect(db) as c:
        return c.execute("SELECT COUNT(*) FROM effects").fetchone()[0]


def remote_worlds(out: Path):
    results = {}
    common_client = {"op_id": "op-remote-1", "state": "sent_no_receipt", "content": "reserve seat"}
    for world, initially_applied in (("none_yet", False), ("applied_reply_lost", True)):
        base = out / world
        base.mkdir(parents=True)
        client_path = base / "client_visible.json"
        client_path.write_text(canonical(common_client) + "\n", encoding="utf-8")
        remote = base / "remote.sqlite"
        init_remote(remote)
        if initially_applied:
            remote_apply(remote, common_client["op_id"], common_client["content"], dedupe=True)
        before = remote_count(remote)
        retry = remote_apply(remote, common_client["op_id"], common_client["content"], dedupe=True)
        mismatch = expect_failure(lambda: remote_apply(remote, common_client["op_id"], "different intent", dedupe=True), KeyConflict)
        results[world] = {"caller_visible_sha256": sha_file(client_path), "remote_effects_before_retry": before,
                          "same_id_retry": retry, "same_id_changed_payload": mismatch,
                          "remote_effects_after_retry": remote_count(remote)}
    no_contract = out / "applied_no_remote_dedupe"
    no_contract.mkdir()
    (no_contract / "client_visible.json").write_text(canonical(common_client) + "\n", encoding="utf-8")
    remote = no_contract / "remote.sqlite"
    init_remote(remote)
    remote_apply(remote, common_client["op_id"], common_client["content"], dedupe=False)
    before = remote_count(remote)
    retry = remote_apply(remote, common_client["op_id"], common_client["content"], dedupe=False)
    results["applied_no_remote_dedupe"] = {"caller_visible_sha256": sha_file(no_contract / "client_visible.json"),
                                           "remote_effects_before_retry": before, "same_id_retry": retry,
                                           "remote_effects_after_retry": remote_count(remote)}
    assert len({v["caller_visible_sha256"] for v in results.values()}) == 1
    assert [results[k]["remote_effects_before_retry"] for k in ("none_yet", "applied_reply_lost")] == [0, 1]
    assert results["none_yet"]["remote_effects_after_retry"] == results["applied_reply_lost"]["remote_effects_after_retry"] == 1
    assert results["applied_no_remote_dedupe"]["remote_effects_after_retry"] == 2
    return results


def demo(out: Path):
    out = safe_path(out)
    if out.exists():
        raise RuntimeError("run directory exists")
    out.mkdir(parents=True)
    db = out / "local_ledger.sqlite"
    init(db, budget=3)
    report = {"scope": "author-driven local host experiment; no model trained to write; all effects confined to fresh work/runs directory",
              "initial": snapshot(db)}
    normal = request("op-normal", "alpha\n", 0)
    report["normal"] = apply(db, normal)
    assert report["normal"]["receipt"]["after_version"] == 1
    report["same_id_replay"] = apply(db, normal)
    assert report["same_id_replay"]["replayed"] and snapshot(db)["version"] == 1
    report["same_id_different_text"] = expect_failure(lambda: apply(db, request("op-normal", "different\n", 0)), KeyConflict)
    report["unapproved_tool"] = expect_failure(lambda: apply(db, request("op-unsafe", "delete me\n", 1, tool="delete_file")), Denied)
    report["unapproved_document"] = expect_failure(lambda: apply(db, request("op-outside", "x\n", 1, doc_id="user_real_document")), Denied)
    assert snapshot(db)["version"] == 1

    before = request("op-before", "beta\n", 1)
    report["killed_before_commit"] = kill_at_marker(db, before, out, "pause_before_commit")
    report["after_before_commit_kill"] = snapshot(db)
    assert report["after_before_commit_kill"]["version"] == 1
    report["before_commit_same_id_retry"] = apply(db, before)
    assert snapshot(db)["version"] == 2

    after = request("op-after", "gamma\n", 2)
    report["killed_after_commit_before_reply"] = kill_at_marker(db, after, out, "pause_after_commit")
    report["after_after_commit_kill"] = snapshot(db)
    assert report["after_after_commit_kill"]["version"] == 3
    report["after_commit_same_id_retry"] = apply(db, after)
    assert report["after_commit_same_id_retry"]["replayed"] and snapshot(db)["version"] == 3
    report["stale_new_operation"] = expect_failure(lambda: apply(db, request("op-stale", "late\n", 1)), StaleVersion)
    report["exhausted_budget"] = expect_failure(lambda: apply(db, request("op-extra", "delta\n", 3)), Denied)
    final = snapshot(db)
    assert final["content"] == "Start\nalpha\nbeta\ngamma\n" and final["version"] == 3 and final["remaining_write_budget"] == 0
    assert len(final["operations"]) == 3
    note = out / "final_note.txt"
    note.write_text(final["content"], encoding="utf-8")
    report["final_local"] = {"content": final["content"], "version": final["version"], "remaining_budget": final["remaining_write_budget"],
                             "committed_operations": [item["op_id"] for item in final["operations"]],
                             "final_note_sha256": sha_file(note), "sqlite_sha256_after_close": sha_file(db)}

    race_db = out / "same_id_concurrent.sqlite"
    init(race_db, budget=1)
    race_request = request("op-race", "once\n", 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(apply, race_db, race_request) for _ in range(2)]
        replies = [future.result() for future in futures]
    race = snapshot(race_db)
    assert race["version"] == 1 and race["content"] == "Start\nonce\n" and sorted(reply["replayed"] for reply in replies) == [False, True]
    report["concurrent_same_id"] = {"replayed_flags": [reply["replayed"] for reply in replies], "effect_count": len(race["operations"]), "content": race["content"]}

    two_intents_db = out / "two_distinct_intents.sqlite"
    init(two_intents_db, budget=2)
    first_intent = apply(two_intents_db, request("op-intent-a", "same\n", 0))
    second_intent = apply(two_intents_db, request("op-intent-b", "same\n", 1))
    two_intents = snapshot(two_intents_db)
    assert two_intents["version"] == 2 and two_intents["content"] == "Start\nsame\nsame\n"
    report["same_text_distinct_logical_ids"] = {"first": first_intent, "second": second_intent,
                                                "final_version": two_intents["version"], "content": two_intents["content"],
                                                "lesson": "different operation IDs represent two authorized append intents even with identical text"}

    cancel_db = out / "cancel.sqlite"
    init(cancel_db, budget=1)
    report["cancel_before_dispatch"] = cancel(cancel_db, "op-cancel-early")
    report["cancelled_execution_rejected"] = expect_failure(lambda: apply(cancel_db, request("op-cancel-early", "never\n", 0)), Cancelled)
    committed = apply(cancel_db, request("op-cancel-late", "already\n", 0))
    report["cancel_after_commit"] = cancel(cancel_db, "op-cancel-late")
    assert report["cancel_after_commit"]["status"] == "already_committed" and snapshot(cancel_db)["content"] == "Start\nalready\n"
    report["cancel_note"] = "cancellation request after commit did not roll back the effect"

    report["separate_remote_worlds"] = remote_worlds(out / "two_worlds")
    report["source"] = {"program_sha256": sha_file(Path(__file__).resolve()),
                        "receipt_boundary": "same local SQLite transaction for note change, op_id record, budget decrement and receipt; remote simulation uses separate physical databases",
                        "limits": "application allowlist is not OS sandbox; remote worlds are a controlled simulation and no external API was called"}
    path = out / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_version": final["version"], "effects": len(final["operations"]),
                      "before_kill_version": report["after_before_commit_kill"]["version"],
                      "after_kill_version": report["after_after_commit_kill"]["version"],
                      "same_id_replay": report["after_commit_same_id_retry"]["replayed"],
                      "remote_worlds": {k: (v["remote_effects_before_retry"], v["remote_effects_after_retry"]) for k, v in report["separate_remote_worlds"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out-dir", type=Path, required=True)
    child = sub.add_parser("execute-child")
    child.add_argument("--db", type=Path, required=True)
    child.add_argument("--request", type=Path, required=True)
    child.add_argument("--fault", choices=("pause_before_commit", "pause_after_commit"), required=True)
    child.add_argument("--marker", type=Path, required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "demo":
        demo(args.out_dir)
    elif args.action == "inspect":
        print(json.dumps(snapshot(args.db), ensure_ascii=False, indent=2))
    else:
        request_path = safe_path(args.request)
        marker_path = safe_path(args.marker)
        apply(args.db, json.loads(request_path.read_text(encoding="utf-8")), args.fault, marker_path)
