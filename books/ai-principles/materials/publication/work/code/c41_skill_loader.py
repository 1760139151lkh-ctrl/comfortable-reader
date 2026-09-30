"""Teach metadata-first skill discovery and manifest-bound, on-demand reads.

This is a small local loader for a teaching copy, not the Codex or MCP host.
It never executes the skill's instructions or gives a model extra permissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
SKILL = WORK / "data/c41_skill_demo/check-book-source"
FILES = ("SKILL.md", "references/source-rules.md")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontmatter(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().strip() != "---":
            raise ValueError("SKILL.md frontmatter missing")
        lines = []
        for line in stream:
            if line.strip() == "---":
                break
            lines.append(line.rstrip("\r\n"))
        else:
            raise ValueError("SKILL.md frontmatter not closed")
    metadata = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if not sep or key not in ("name", "description") or not value.strip():
            raise ValueError("this teaching loader accepts only name/description")
        metadata[key] = value.strip()
    if set(metadata) != {"name", "description"} or metadata["name"] != path.parent.name:
        raise ValueError("name and skill directory disagree")
    return metadata


def manifest(skill_dir: Path):
    meta = frontmatter(skill_dir / "SKILL.md")
    return {"origin": "local teaching copy, not installed in Codex",
            "frontmatter": meta,
            "resources": [{"path": name, "sha256": sha(skill_dir / name),
                           "bytes": (skill_dir / name).stat().st_size} for name in FILES]}


def load(skill_dir: Path, registered):
    contents = {}
    for item in registered["resources"]:
        file = (skill_dir / item["path"]).resolve()
        if not file.is_relative_to(skill_dir.resolve()) or not file.is_file():
            raise ValueError("resource outside skill or missing")
        if file.stat().st_size != item["bytes"] or sha(file) != item["sha256"]:
            raise ValueError("resource changed since manifest")
        contents[item["path"]] = file.read_text(encoding="utf-8")
    if frontmatter(skill_dir / "SKILL.md") != registered["frontmatter"]:
        raise ValueError("frontmatter changed since discovery")
    return contents


def demo(out_dir: Path):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new work/runs output directory")
    out.mkdir(parents=True)
    copy = out / "check-book-source"
    for name in FILES:
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SKILL / name, target)
    registered = manifest(copy)
    # Discovery reads metadata and file hashes only; instructions stay unloaded.
    listed = {"frontmatter": registered["frontmatter"],
              "file_count": len(registered["resources"]),
              "instruction_content_loaded": False}
    loaded = load(copy, registered)  # author explicitly selects the skill here
    assert "SKILL.md" in loaded and "references/source-rules.md" in loaded
    support = copy / "references/source-rules.md"
    support.write_bytes(support.read_bytes() + b"changed teaching copy\n")
    try:
        load(copy, registered)
    except ValueError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("changed supporting file should be rejected")
    report = {"discovered": listed, "selected_by": "author demonstration, not model selection",
              "after_selection_content_loaded": sorted(loaded),
              "old_manifest_rejects_changed_support_file": rejected,
              "registered_manifest": registered,
              "original_example_support_sha256_unchanged": sha(SKILL / FILES[1]) == registered["resources"][1]["sha256"],
              "scope": "local copy of an uninstalled teaching skill; no Codex host activation, MCP Skills extension call, model training, tool grant or real user approval"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_only_before_selection": not listed["instruction_content_loaded"],
                      "loaded_files_after_selection": len(loaded),
                      "changed_copy_rejected": bool(rejected),
                      "source_unchanged": report["original_example_support_sha256_unchanged"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    demo(args.out_dir)
