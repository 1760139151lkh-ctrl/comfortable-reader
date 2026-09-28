"""C40: provenance-aware synthetic user facts across real process restarts.

Only author-invented preferences/location examples are changed or deleted.
The C39 final note is COPIED into this demo; its original is never modified.
Deletion is scoped to this active SQLite store and two synthetic source files,
not prior output receipts, backups, other apps, or model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
C39_NOTE = WORK / "runs/c39_host_lab_inspectable/final_note.txt"
C39_NOTE_SHA = "e18ebeed0a9d7a3855c5b6511011c355bc045d11402751144c0c3a5f8b6cd056"
AS_OF_FIRST = "2026-09-25"
AS_OF_LATER = "2026-09-28"
SCHEMA = """
CREATE TABLE facts (key TEXT PRIMARY KEY, value TEXT, version INTEGER NOT NULL,
                    source_path TEXT, source_sha256 TEXT, source_kind TEXT NOT NULL,
                    valid_from TEXT NOT NULL, expires_on TEXT, status TEXT NOT NULL);
CREATE TABLE candidate_observations (key TEXT, value TEXT, source_kind TEXT,
                                     source_path TEXT, source_sha256 TEXT);
CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,
                     action TEXT NOT NULL, source_sha256 TEXT, at_date TEXT NOT NULL);
"""


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(out: Path, child: Path):
    target = child.resolve()
    if not target.is_relative_to(out.resolve()):
        raise ValueError("source outside this synthetic memory lab")
    return target


def check_out(out: Path):
    path = out.resolve()
    if not path.is_relative_to(RUNS.resolve()):
        raise ValueError("memory lab must stay inside work/runs")
    return path


def connect(out):
    c = sqlite3.connect(check_out(out) / "active_memory.sqlite")
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA secure_delete=ON")
    return c


def source_file(out, name, statement):
    path = within(out, out / "sources" / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("source file exists")
    path.write_text(json.dumps(statement, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def insert_fact(c, key, value, path, kind, valid_from, expires_on=None):
    c.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
              (key, value, 1, str(path), sha(path), kind, valid_from, expires_on, "active"))
    c.execute("INSERT INTO events(key,action,source_sha256,at_date) VALUES (?,?,?,?)",
              (key, "created", sha(path), valid_from))


def query(out, key, as_of):
    out = check_out(out)
    with connect(out) as c:
        row = c.execute("SELECT value,version,source_path,source_sha256,source_kind,valid_from,expires_on,status FROM facts WHERE key=?", (key,)).fetchone()
        conflict_count = c.execute("SELECT COUNT(*) FROM candidate_observations WHERE key=?", (key,)).fetchone()[0]
    if row is None:
        return {"key": key, "status": "unknown", "value": None, "untrusted_candidate_count": conflict_count}
    value, version, source_path, source_sha, kind, valid_from, expires_on, status = row
    base = {"key": key, "version": version, "source_kind": kind,
            "source_sha256": source_sha, "untrusted_candidate_count": conflict_count}
    if status == "deleted":
        return {**base, "status": "deleted", "value": None}
    if as_of < valid_from:
        return {**base, "status": "not_yet_valid", "value": None}
    if expires_on is not None and as_of >= expires_on:
        return {**base, "status": "expired_hold", "value": None, "expires_on": expires_on}
    source = Path(source_path)
    if not source.exists() or sha(source) != source_sha:
        return {**base, "status": "source_changed_hold", "value": None}
    return {**base, "status": "current", "value": value,
            "source_path": str(source.relative_to(out)), "valid_from": valid_from, "expires_on": expires_on}


def phase1(out):
    out = check_out(out)
    if out.exists():
        raise RuntimeError("output exists")
    out.mkdir(parents=True)
    if sha(C39_NOTE) != C39_NOTE_SHA:
        raise RuntimeError("C39 source artifact changed")
    copy = within(out, out / "sources" / "c39_note_copy.txt")
    copy.parent.mkdir(parents=True)
    copy.write_bytes(C39_NOTE.read_bytes())
    assert sha(copy) == C39_NOTE_SHA
    theme = source_file(out, "synthetic_user_theme_first.json", {"made_up_for_textbook": True, "key": "reading_theme", "value": "dark", "speaker": "fictional user"})
    room = source_file(out, "synthetic_user_room.json", {"made_up_for_textbook": True, "key": "study_room", "value": "B204", "expires_on": "2026-09-27", "speaker": "fictional user"})
    tool = source_file(out, "synthetic_untrusted_tool.json", {"made_up_for_textbook": True, "key": "reading_theme", "value": "light", "speaker": "tool output, not user"})
    marker_value = secrets.token_hex(8)
    marker = source_file(out, "synthetic_recovery_marker.json", {"made_up_for_textbook": True, "key": "recovery_marker", "value": marker_value, "speaker": "author-generated only in first process"})
    with connect(out) as c:
        c.executescript(SCHEMA)
        insert_fact(c, "reading_theme", "dark", theme, "synthetic_user_explicit", AS_OF_FIRST)
        insert_fact(c, "study_room", "B204", room, "synthetic_user_explicit", AS_OF_FIRST, "2026-09-27")
        insert_fact(c, "c39_note_text", copy.read_text(encoding="utf-8"), copy,
                    "verified_local_artifact_copy", AS_OF_FIRST)
        insert_fact(c, "recovery_marker", marker_value, marker, "synthetic_first_process_random", AS_OF_FIRST)
        c.execute("INSERT INTO candidate_observations VALUES (?,?,?,?,?)", ("reading_theme", "light", "untrusted_tool", str(tool), sha(tool)))
        c.commit()
    results = {key: query(out, key, AS_OF_FIRST) for key in ("reading_theme", "study_room", "c39_note_text", "recovery_marker", "not_recorded")}
    assert results["reading_theme"]["value"] == "dark" and results["reading_theme"]["untrusted_candidate_count"] == 1
    assert results["c39_note_text"]["value"] == C39_NOTE.read_text(encoding="utf-8") and results["not_recorded"]["status"] == "unknown"
    return {"phase": "first_process_created_sources_and_memory", "queries": results, "database_sha256": sha(out / "active_memory.sqlite")}


def phase2(out):
    out = check_out(out)
    before = query(out, "reading_theme", AS_OF_FIRST)
    recovered_marker = query(out, "recovery_marker", AS_OF_FIRST)
    if before["value"] != "dark" or before["untrusted_candidate_count"] != 1:
        raise RuntimeError("fresh process did not recover the old source")
    if recovered_marker["status"] != "current" or len(recovered_marker["value"]) != 16:
        raise RuntimeError("fresh process did not recover first-process random value")
    corrected = source_file(out, "synthetic_user_theme_correction.json", {"made_up_for_textbook": True, "key": "reading_theme", "value": "warm", "speaker": "fictional user correction"})
    with connect(out) as c:
        old = c.execute("SELECT version,source_sha256 FROM facts WHERE key='reading_theme'").fetchone()
        c.execute("UPDATE facts SET value=?,version=?,source_path=?,source_sha256=?,source_kind=?,valid_from=?,status='active' WHERE key='reading_theme'",
                  ("warm", old[0] + 1, str(corrected), sha(corrected), "synthetic_user_correction", "2026-09-26"))
        c.execute("INSERT INTO events(key,action,source_sha256,at_date) VALUES (?,?,?,?)", ("reading_theme", "corrected_old_source_" + old[1][:8], sha(corrected), "2026-09-26"))
        c.commit()
    after = query(out, "reading_theme", "2026-09-26")
    assert after["value"] == "warm" and after["version"] == 2 and after["untrusted_candidate_count"] == 1
    return {"phase": "fresh_process_recovered_then_user_corrected", "before": before,
            "recovered_first_process_random_marker": recovered_marker, "after": after}


def phase3(out):
    out = check_out(out)
    theme = query(out, "reading_theme", AS_OF_LATER)
    room = query(out, "study_room", AS_OF_LATER)
    assert theme["value"] == "warm" and room["status"] == "expired_hold" and room["value"] is None
    copied = within(out, out / "sources" / "c39_note_copy.txt")
    copied.write_bytes(copied.read_bytes() + b"altered teaching copy\n")
    changed = query(out, "c39_note_text", AS_OF_LATER)
    if changed["status"] != "source_changed_hold" or sha(C39_NOTE) != C39_NOTE_SHA:
        raise RuntimeError("source drift check or original C39 protection failed")
    return {"phase": "fresh_process_checked_expiry_and_source_drift", "theme": theme, "room": room,
            "c39_fact_after_copy_changed": changed, "original_c39_artifact_sha256_unchanged": sha(C39_NOTE)}


def phase4(out):
    out = check_out(out)
    theme_paths = [within(out, out / "sources" / name) for name in ("synthetic_user_theme_first.json", "synthetic_user_theme_correction.json")]
    with connect(out) as c:
        old = c.execute("SELECT version FROM facts WHERE key='reading_theme'").fetchone()
        if old is None:
            raise RuntimeError("theme fact missing")
        c.execute("UPDATE facts SET value=NULL,version=?,source_path=NULL,source_sha256=NULL,source_kind='user_deleted',status='deleted' WHERE key='reading_theme'", (old[0] + 1,))
        c.execute("DELETE FROM candidate_observations WHERE key='reading_theme'")
        c.execute("INSERT INTO events(key,action,source_sha256,at_date) VALUES (?,?,?,?)", ("reading_theme", "user_deleted_active_value", None, AS_OF_LATER))
        c.commit()
        c.execute("VACUUM")
    for path in theme_paths:
        if path.exists():
            path.unlink()  # only these two verified synthetic files, never recursive
    later = query(out, "reading_theme", AS_OF_LATER)
    with connect(out) as c:
        active = c.execute("SELECT value,status FROM facts WHERE key='reading_theme'").fetchone()
        candidates = c.execute("SELECT COUNT(*) FROM candidate_observations WHERE key='reading_theme'").fetchone()[0]
    assert later["status"] == "deleted" and later["value"] is None
    assert active == (None, "deleted") and candidates == 0 and all(not path.exists() for path in theme_paths)
    return {"phase": "fresh_process_user_deleted_active_memory", "query": later,
            "active_db_row": {"value": active[0], "status": active[1]},
            "untrusted_candidates_remaining": candidates, "two_synthetic_source_files_removed": True,
            "boundary": "prior session output receipts and other backups/transcripts are outside this deletion scope; no claim of model-weight unlearning or forensic erasure"}


def demo(out):
    out = check_out(out)
    if out.exists():
        raise RuntimeError("output exists")
    phases = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for phase in ("phase1", "phase2", "phase3", "phase4"):
        command = [sys.executable, str(Path(__file__).resolve()), phase, "--out-dir", str(out)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=20, creationflags=flags)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} failed: {process.stderr}")
        receipt = json.loads(process.stdout)
        phases.append(receipt)
        (out / f"{phase}_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_theme = query(out, "reading_theme", AS_OF_LATER)
    final_room = query(out, "study_room", AS_OF_LATER)
    final_c39 = query(out, "c39_note_text", AS_OF_LATER)
    report = {"scope": "four actual separate Python processes on author-synthetic user statements and copied C39 artifact; not a human memory study",
              "phase_receipts": [f"{phase}_receipt.json" for phase in ("phase1", "phase2", "phase3", "phase4")],
              "results": {"session2_recovered_old_theme": phases[1]["before"]["value"] == "dark",
                          "fresh_process_recovered_unpredictable_marker": phases[1]["recovered_first_process_random_marker"]["value"] == phases[0]["queries"]["recovery_marker"]["value"],
                          "correction_changed_theme": phases[1]["after"]["value"] == "warm",
                          "later_room_expired": phases[2]["room"]["status"] == "expired_hold",
                          "copied_c39_source_drift_held": phases[2]["c39_fact_after_copy_changed"]["status"] == "source_changed_hold",
                          "user_deleted_active_theme": final_theme["status"] == "deleted" and final_theme["value"] is None,
                          "original_c39_note_unchanged": sha(C39_NOTE) == C39_NOTE_SHA},
              "final_status": {"theme": final_theme["status"], "room": final_room["status"], "c39_fact": final_c39["status"]},
              "active_db_sha256": sha(out / "active_memory.sqlite"), "original_c39_note_sha256": sha(C39_NOTE),
              "limitations": "source changes/expiry stop current use; synthetic correction/deletion affects this active DB and two own source files only. Prior phase receipts intentionally remain and may contain old fictional values; no parameter unlearning or global erase."}
    assert all(report["results"].values())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"four_separate_processes": len(phases), **report["results"], "final_status": report["final_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("demo", "phase1", "phase2", "phase3", "phase4"):
        choice = sub.add_parser(action)
        choice.add_argument("--out-dir", type=Path, required=True)
    query_parser = sub.add_parser("query")
    query_parser.add_argument("--out-dir", type=Path, required=True)
    query_parser.add_argument("--key", required=True)
    query_parser.add_argument("--as-of", required=True)
    args = p.parse_args()
    if args.action == "demo":
        demo(args.out_dir)
    elif args.action == "query":
        print(json.dumps(query(args.out_dir, args.key, args.as_of), ensure_ascii=False))
    else:
        print(json.dumps(globals()[args.action](args.out_dir), ensure_ascii=False))
