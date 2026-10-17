"""Cuts the first N chapters of a plain-text book into deploy/k8s/data/,
re-headed "Chapter 1", "Chapter 2", ... so chapter detection is exact.

The benchmark book is Pride and Prejudice (public domain), e.g. from
Project Gutenberg (https://www.gutenberg.org/ebooks/1342, "Plain Text
UTF-8"). Generated books are git-ignored -- never committed.

    python deploy/k8s/make_bench_book.py pg1342.txt --chapters 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alebrijes.ingest import parse_book

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="Plain-text book with chapter headings")
    parser.add_argument("--chapters", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "bench.txt")
    args = parser.parse_args()

    book = parse_book(args.source)
    if len(book.chapters) < args.chapters:
        raise SystemExit(f"'{args.source}' only parses into {len(book.chapters)} chapters")

    blocks = []
    for number, chapter in enumerate(book.chapters[: args.chapters], start=1):
        blocks.append(f"Chapter {number}")
        # parse_book keeps each chapter's own heading as its first paragraph.
        blocks.extend(p.text for p in chapter.paragraphs[1:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    words = sum(len(b.split()) for b in blocks)
    print(f"Wrote {args.out}: {args.chapters} chapters, {words:,} words")


if __name__ == "__main__":
    main()
