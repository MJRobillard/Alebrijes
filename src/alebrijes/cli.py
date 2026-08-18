"""`alebrijes` command-line entry point.

Subcommands are wired up so `alebrijes --help` documents the intended surface
(Phase 2's exit criteria — one installable command replacing pocket-tts-agent's
four overlapping scripts). `voices` and `render` are still stubs; `cast` is
implemented: ingest -> segment -> attribute -> a reviewable cast.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_voices(args: argparse.Namespace) -> int:
    raise NotImplementedError("`alebrijes voices` (create/list cloned voice states) not yet implemented.")


def _cmd_cast(args: argparse.Namespace) -> int:
    from alebrijes.cast import save
    from alebrijes.cast.build import LOW_CONFIDENCE_THRESHOLD, build_cast_file

    book_path = Path(args.book)
    if book_path.suffix.lower() != ".epub":
        raise NotImplementedError(
            f"Ingest for '{book_path.suffix}' files is not yet implemented; only .epub is supported so far."
        )

    from alebrijes.ingest.epub import parse

    book = parse(book_path)
    result = build_cast_file(book)

    out_path = Path(args.out) if args.out else book_path.with_suffix(".cast.yaml")
    save(result.cast_file, out_path)

    print(f"Wrote cast file: {out_path}")
    print(f"Characters found: {len(result.cast_file.entries)}")

    low_confidence = result.low_confidence()
    if low_confidence:
        print(f"{len(low_confidence)} line(s) need review (confidence < {LOW_CONFIDENCE_THRESHOLD}):")
        for attribution in low_confidence[:10]:
            preview = attribution.span.text[:60]
            print(f"  [{attribution.confidence:.2f}] {attribution.character}: {preview!r}")

    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    raise NotImplementedError("`alebrijes render` (synthesize + assemble a cast.yaml into audio) not yet implemented.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alebrijes", description="Local, multi-voice book-to-audiobook engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    voices = subparsers.add_parser("voices", help="Create or list cloned voice states.")
    voices.set_defaults(func=_cmd_voices)

    cast = subparsers.add_parser("cast", help="Run attribution and write an editable cast file.")
    cast.add_argument("book", help="Path to an epub/pdf/txt source file.")
    cast.add_argument("--out", help="Where to write the cast file (default: <book>.cast.yaml).")
    cast.set_defaults(func=_cmd_cast)

    render = subparsers.add_parser("render", help="Synthesize and assemble a book using its cast file.")
    render.add_argument("book", help="Path to an epub/pdf/txt source file.")
    render.add_argument("--cast", required=True, help="Path to the cast.yaml produced by `alebrijes cast`.")
    render.set_defaults(func=_cmd_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
