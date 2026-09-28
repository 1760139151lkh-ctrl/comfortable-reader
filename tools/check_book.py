"""Validate one private or public book without adding it to a catalog."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from book_format import BookError, validate_book


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("book",type=Path)
    args=parser.parse_args()
    spec,documents=validate_book(args.book,args.book.name)
    print(json.dumps({"status":"structurally_valid", "id":spec["id"], "revision":spec["revision"],
        "chapters":len(documents), "resources":len(spec.get("resources",[])),
        "activities":len(spec.get("activities",[])), "rights":spec["rights"]["status"],
        "execution_authorized":False},ensure_ascii=False))


if __name__=="__main__":
    try:main()
    except (BookError,OSError,UnicodeError,ValueError,KeyError) as error:
        print(f"书籍结构未通过：{error}",file=sys.stderr);raise SystemExit(2)
