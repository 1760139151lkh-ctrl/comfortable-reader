"""Fetch official LibriSpeech dev-clean, verify OpenSLR MD5, safely extract."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/data/librispeech_dev_clean"
URL = "https://openslr.trmal.net/resources/12/dev-clean.tar.gz"
EXPECTED_MD5 = "42e2234ba48799c1f50f24a7926300a1"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "dev-clean.tar.gz"
    extracted_root = OUT / "extracted"
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"verified source already exists: {OUT / 'manifest.json'}")
    if extracted_root.exists() and any(extracted_root.iterdir()):
        raise FileExistsError(f"preserve partial extraction for inspection: {extracted_root}")
    md5 = hashlib.md5()
    if archive.exists():
        with archive.open("rb") as source:
            while chunk := source.read(1 << 20):
                md5.update(chunk)
    else:
        with urllib.request.urlopen(URL, timeout=180) as source, archive.open("wb") as target:
            while chunk := source.read(1 << 20):
                target.write(chunk)
                md5.update(chunk)
    if md5.hexdigest() != EXPECTED_MD5:
        raise RuntimeError(
            f"OpenSLR dev-clean archive MD5 mismatch: {md5.hexdigest()}"
        )
    extracted_root.mkdir(exist_ok=True)
    extracted_files = 0
    flac_count = 0
    transcript_count = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported tar member: {member.name}")
            if not (member.name.startswith("LibriSpeech/dev-clean/") or
                    member.name in {"LibriSpeech/LICENSE.TXT", "LibriSpeech/README.TXT",
                                    "LibriSpeech/CHAPTERS.TXT", "LibriSpeech/SPEAKERS.TXT",
                                    "LibriSpeech/BOOKS.TXT"}):
                raise RuntimeError(f"unexpected tar path: {member.name}")
            target = (extracted_root / member.name).resolve()
            if extracted_root.resolve() not in target.parents:
                raise RuntimeError(f"tar path escapes source directory: {member.name}")
            item = source.extractfile(member)
            if item is None:
                raise RuntimeError(f"cannot open tar member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with item, target.open("wb") as output:
                shutil.copyfileobj(item, output, length=1 << 20)
            if target.stat().st_size != member.size:
                raise RuntimeError(f"extracted member size mismatch: {member.name}")
            extracted_files += 1
            flac_count += target.suffix.lower() == ".flac"
            transcript_count += target.name.endswith(".trans.txt")
    manifest = {
        "official_page": "https://www.openslr.org/12/",
        "archive_url": URL,
        "official_md5_list": "https://openslr.trmal.net/resources/12/md5sum.txt",
        "official_expected_archive_md5": EXPECTED_MD5,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "license": "CC BY 4.0, per OpenSLR SLR12 page",
        "official_identity": "LibriSpeech dev-clean development set; approximately 16kHz clean read English audiobook speech, not an official training split",
        "extraction": "only regular files under LibriSpeech/dev-clean/ and the five LibriSpeech top-level metadata files, path containment checked",
        "extracted_regular_file_count": extracted_files,
        "flac_count": flac_count,
        "transcript_file_count": transcript_count,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("LibriSpeech dev-clean", archive.stat().st_size,
          "MD5", md5.hexdigest(), "FLAC", flac_count,
          "transcript files", transcript_count)


if __name__ == "__main__":
    main()
