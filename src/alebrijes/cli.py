"""`alebrijes` command-line entry point.

Skeleton only: subcommands are wired up so `alebrijes --help` documents the
intended surface (Phase 2's exit criteria — one installable command replacing
pocket-tts-agent's four overlapping scripts), but each body is a stub until
the ingest/segment/attribute stages are implemented.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_voices(args: argparse.Namespace) -> int:
    raise NotImplementedError("`alebrijes voices` (create/list cloned voice states) not yet implemented.")


def _cmd_cast(args: argparse.Namespace) -> int:
    raise NotImplementedError("`alebrijes cast` (run attribution, write a cast.yaml for review) not yet implemented.")


def _cmd_render(args: argparse.Namespace) -> int:
    raise NotImplementedError("`alebrijes render` (synthesize + assemble a cast.yaml into audio) not yet implemented.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alebrijes", description="Local, multi-voice book-to-audiobook engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    voices = subparsers.add_parser("voices", help="Create or list cloned voice states.")
    voices.set_defaults(func=_cmd_voices)

    cast = subparsers.add_parser("cast", help="Run attribution and write an editable cast file.")
    cast.add_argument("book", help="Path to an epub/pdf/txt source file.")
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
