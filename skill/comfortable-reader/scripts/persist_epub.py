#!/usr/bin/env python3
"""Audit and import the exact reviewed EPUB, without rebuilding or flattening it."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import zipfile
from contextlib import closing
from pathlib import Path
from xml.etree import ElementTree as ET

from build_reader import BuildError, add_to_calibre_library
from epub_tools import EpubAuditError, audit_epub


def check_pdf_coverage(path: Path, epub: Path, expected_sha256: str) -> None:
    from build_from_ir import load_packet
    from collections import Counter
    report=json.loads(path.read_text(encoding='utf-8'))
    if report.get('epub_sha256')!=expected_sha256 or report.get('scope')!='full' or report.get('source_review_passed') is not True:
        raise EpubAuditError('PDF coverage must be complete and belong to these final EPUB bytes.')
    for field in ('missing_source_ids','duplicate_source_ids','out_of_scope_source_ids','unreviewed_block_ids','issues'):
        if report.get(field)!=[]:
            raise EpubAuditError('PDF coverage has unresolved entries: '+field)
    packet_path=Path(report['source_packet'])
    packet,units=load_packet(packet_path)
    source=Path(packet['source']['path']).read_bytes()
    if packet['errors'] or hashlib.sha256(source).hexdigest()!=report.get('source_sha256') or report.get('source_sha256')!=packet['source']['sha256'] or report.get('inventory_sha256')!=packet.get('inventory_sha256'):
        raise EpubAuditError('PDF source packet or source bytes differ from the coverage report.')
    expected=set(units)
    if set(report.get('expected_source_ids',[]))!=expected:
        raise EpubAuditError('Coverage omits source units from the source packet.')
    with zipfile.ZipFile(epub) as archive:
        container=ET.fromstring(archive.read('META-INF/container.xml'))
        package_name=next(n.get('full-path') for n in container.iter() if n.tag.endswith('rootfile'))
        package=ET.fromstring(archive.read(package_name))
        from epub_tools import OPF,resolve_resource
        items={n.get('id'):resolve_resource(package_name,n.get('href',''))[0] for n in package.find('{'+OPF+'}manifest')}
        documents=[items[n.get('idref')] for n in package.find('{'+OPF+'}spine') if n.get('linear','yes')!='no']
        targets={name+'#'+n.get('id') for name in documents for n in ET.fromstring(archive.read(name)).iter() if n.get('id')}
    counts=Counter()
    for row in report.get('mapped_units',[]):
        if row.get('target') not in targets:
            raise EpubAuditError('Coverage target is not an actual readable spine anchor.')
        counts[row['source_id']]+=1
    mapped=set(counts)
    for row in report.get('excluded_units',[]):
        if row.get('reason') not in {'page_background','running_header','page_number','verified_duplicate','explicit_user_exclusion'} or not row.get('evidence'):
            raise EpubAuditError('Unexplained PDF source exclusion.')
        if row['reason']=='verified_duplicate' and row.get('same_as') not in mapped:
            raise EpubAuditError('A duplicate exclusion has no retained source twin.')
        counts[row['source_id']]+=1
    if set(counts)!=expected or any(n!=1 for n in counts.values()):
        raise EpubAuditError('PDF source units are missing, duplicated, or invented in coverage.')


def check_review(path: Path, expected_sha256: str, epub: Path | None = None, source_sha256: str | None = None) -> dict:
    review = json.loads(path.read_text(encoding="utf-8-sig"))
    if review.get("epub_sha256") != expected_sha256:
        raise EpubAuditError("Review belongs to different EPUB bytes. Recheck the changed artifact.")
    for field in ("source_review", "render_review"):
        evidence = review.get(field)
        if not isinstance(evidence, dict) or evidence.get("status") != "passed" or not evidence.get("evidence"):
            raise EpubAuditError(f"{field} needs a completed check and its evidence; an untested preview is not passed.")
        if any(not isinstance(item, dict) or not item.get("kind") or not item.get("location") or not item.get("result") for item in evidence["evidence"]):
            raise EpubAuditError(f"{field}.evidence entries need kind, location, and result; a vague sentence is not evidence.")
    if review.get("unresolved_issues") != []:
        raise EpubAuditError("Resolve the recorded issues before import; do not import a draft.")
    method=review['source_review'].get('method')
    if method not in {'verbatim_source','unchanged_epub','pdf_reconstruction'}:
        raise EpubAuditError('Declare source_review.method: verbatim_source, unchanged_epub, or pdf_reconstruction.')
    if method=='verbatim_source' and review['source_review'].get('byte_equal') is not True:
        raise EpubAuditError('Verbatim source review must record byte_equal=true after comparing the exact embedded entry.')
    if method=='verbatim_source' and not review['source_review'].get('source_sha256'):
        raise EpubAuditError('Verbatim source review must include the compared source_sha256.')
    if source_sha256 is not None and review['source_review'].get('source_sha256') != source_sha256:
        raise EpubAuditError('Review source_sha256 does not match the supplied source file.')
    if method=='unchanged_epub' and review['source_review'].get('original_epub_sha256')!=expected_sha256:
        raise EpubAuditError('Unchanged EPUB review must match the original EPUB hash.')
    if epub is not None:
        with zipfile.ZipFile(epub) as archive:
            has_original_pdf=any('/original/' in name and name.lower().endswith('.pdf') for name in archive.namelist())
        if has_original_pdf and method=='verbatim_source':
            raise EpubAuditError('Embedding original PDF bytes is not visible-content verification. Supply PDF reconstruction coverage.')
        if method=='pdf_reconstruction':
            name=review['source_review'].get('coverage_report')
            if not name:
                raise EpubAuditError('PDF reconstruction requires a source-unit coverage report.')
            check_pdf_coverage((path.parent/name).resolve(),epub,expected_sha256)
    return review


def library_epub(library: Path, book_id: int) -> Path:
    database = (library / "metadata.db").resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT books.path, data.name FROM books JOIN data ON data.book=books.id "
            "WHERE books.id=? AND upper(data.format)='EPUB'", (book_id,),
        ).fetchall()
    if len(rows) != 1:
        raise EpubAuditError(f"Book {book_id} does not resolve to exactly one EPUB in the verified library.")
    return library / rows[0][0] / (rows[0][1] + ".epub")


def persist(path: Path, library: Path, review_path: Path, source_sha256: str | None = None) -> dict:
    if not (library / "metadata.db").is_file():
        raise EpubAuditError("The target must be an existing verified Calibre library; no new library was created.")
    audit = audit_epub(path)
    if audit["errors"] or audit["raw_math_fallbacks"] or audit["external_render_resources"]:
        raise EpubAuditError("EPUB is not ready: " + "; ".join(audit["errors"] or ["unrendered math or external rendering dependencies"]))
    check_review(review_path, str(audit["epub_sha256"]), path, source_sha256)
    if not audit["title"]:
        raise EpubAuditError("Set a book title in the final EPUB before reviewing and importing it.")
    # Ignore ZIP timestamps/compression when detecting an unchanged repackage.
    key = str(audit["content_sha256"])[:24]
    # Older builder versions used a different identifier recipe. A title is
    # only a candidate filter: deduplicate only after exact EPUB content matches.
    with closing(sqlite3.connect((library / "metadata.db").resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        candidates = connection.execute(
            "SELECT books.id FROM books JOIN data ON data.book=books.id "
            "WHERE books.title=? AND upper(data.format)='EPUB'", (str(audit["title"]),),
        ).fetchall()
    matches = []
    for (ident,) in candidates:
        candidate = library_epub(library, ident)
        if candidate.is_file() and audit_epub(candidate)["content_sha256"] == audit["content_sha256"]:
            matches.append(ident)
    if matches:
        result = {"path": str(library), "status": "already_present", "book_ids": matches,
                  "matched_by": "exact_epub_content"}
    else:
        result = add_to_calibre_library(path, library, str(audit["title"]),
                                      " & ".join(audit["authors"]) or "未知",
                                      str(audit["language"]) or "en", key)
    copies = []
    for ident in result["book_ids"]:
        saved = library_epub(library, ident)
        saved_audit = audit_epub(saved)
        if saved_audit["content_sha256"] != audit["content_sha256"]:
            raise EpubAuditError(f"Calibre record {ident} contains different content; do not claim successful delivery.")
        copies.append({"book_id": ident, "path": str(saved.resolve()), "epub_sha256": saved_audit["epub_sha256"]})
    if len(copies) != 1:
        raise EpubAuditError("Multiple records share the identifier. Resolve the duplicate records without discarding annotations.")
    return {"status": "ok", "audit": audit, "library": result, "stored_copies": copies,
            "delivery_status": "awaiting_desktop_verification"}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--source", type=Path, help="Optional original file to compare against an embedded entry")
    parser.add_argument("--source-entry", help="Exact ZIP member containing the untouched original bytes")
    parser.add_argument("--report", type=Path, help="Save this tool's audit JSON")
    parser.add_argument("--add-to-library", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--review-report", type=Path)
    args = parser.parse_args(argv)
    try:
        path = args.epub.expanduser().resolve()
        result = {"status": "ok", "audit": audit_epub(path), "delivery_status": "draft_not_imported"}
        if bool(args.source) != bool(args.source_entry):
            raise EpubAuditError("Use --source and --source-entry together.")
        if args.source:
            original = args.source.read_bytes()
            with zipfile.ZipFile(path) as archive:
                embedded = archive.read(args.source_entry)
            if embedded != original:
                raise EpubAuditError("The embedded original is not byte-equal to the supplied source.")
            result["source"] = {"path": str(args.source.resolve()), "entry": args.source_entry,
                                "sha256": hashlib.sha256(original).hexdigest(), "byte_equal": True}
        audit = result["audit"]
        if audit["errors"] or audit["raw_math_fallbacks"] or audit["external_render_resources"]:
            result["status"] = "needs_attention"
        if args.add_to_library:
            if not args.library or not args.review_report:
                raise EpubAuditError("Import requires --library and --review-report. See references/delivery-gates.md.")
            saved = persist(path, args.library.expanduser().resolve(), args.review_report,
                            hashlib.sha256(args.source.read_bytes()).hexdigest() if args.source else None)
            result.update(saved)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 1
    except (BuildError, EpubAuditError, OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile, sqlite3.Error) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
