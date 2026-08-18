"""`alebrijes` command-line entry point.

Subcommands are wired up so `alebrijes --help` documents the intended surface
(Phase 2's exit criteria — one installable command replacing pocket-tts-agent's
four overlapping scripts). `cast` and `render` are implemented: ingest ->
segment -> attribute -> a reviewable cast.yaml -> a rendered, chaptered .m4b.
`voices` creates a cloned voice state from a local audio file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
_DEFAULT_VOICES_DIR = Path.home() / ".local" / "share" / "alebrijes" / "voices"


def _ingest_book(book_path: Path):
    """Parses `book_path` with the ingest module for its extension.

    Kept in one place so `cast` and `render` support exactly the same set of
    formats and fail the same way on an unsupported one.
    """
    suffix = book_path.suffix.lower()
    if suffix == ".epub":
        from alebrijes.ingest.epub import parse
    elif suffix == ".pdf":
        from alebrijes.ingest.pdf import parse
    elif suffix == ".txt":
        from alebrijes.ingest.txt import parse
    else:
        raise NotImplementedError(
            f"Ingest for '{suffix}' files is not supported. Supported formats: .epub, .pdf, .txt."
        )
    return parse(book_path)


def _cmd_gui(args: argparse.Namespace) -> int:
    from alebrijes.gui import main as gui_main

    gui_main()
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    if args.voices_command == "add":
        return _cmd_voices_add(args)
    if args.voices_command == "list":
        return _cmd_voices_list(args)
    raise NotImplementedError(f"`alebrijes voices {args.voices_command}` is not implemented.")


def _cmd_voices_list(args: argparse.Namespace) -> int:
    voices_dir = Path(args.voices_dir)
    voice_files = sorted(voices_dir.glob("*.pt")) if voices_dir.exists() else []
    if not voice_files:
        print(f"No cloned voices found in {voices_dir}")
        return 0
    print(f"Cloned voices in {voices_dir}:")
    for voice_file in voice_files:
        print(f"  {voice_file.stem}")
    return 0


def _cmd_voices_add(args: argparse.Namespace) -> int:
    from alebrijes.synth.agent import create_voice_state

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Voice source file not found: {source_path}")
    if source_path.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise ValueError(
            f"'{source_path}' doesn't look like an audio file (expected one of: "
            f"{', '.join(sorted(_AUDIO_EXTENSIONS))}). Voice cloning only accepts local audio "
            "files — no URLs or YouTube links."
        )

    voices_dir = Path(args.voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    out_path = voices_dir / f"{args.name}.pt"

    import torch

    state = create_voice_state(source_path, clip_start=args.start, clip_end=args.end)
    torch.save(state, out_path)

    print(f"Saved voice state: {out_path}")
    return 0


def _cmd_cast(args: argparse.Namespace) -> int:
    from alebrijes.cast import save
    from alebrijes.cast.build import LOW_CONFIDENCE_THRESHOLD, build_cast_file

    book_path = Path(args.book)
    book = _ingest_book(book_path)
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
    from alebrijes.cast import load
    from alebrijes.render import render_book

    book_path = Path(args.book)
    book = _ingest_book(book_path)
    cast_file = load(Path(args.cast))
    out_path = Path(args.out) if args.out else book_path.with_suffix(".m4b")

    print(f"Rendering '{book.title}' ({len(book.chapters)} chapters) with cast: {args.cast}")
    result_path = render_book(book, cast_file, out_path, pause_ms=args.pause_ms)
    print(f"Wrote audiobook: {result_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alebrijes", description="Local, multi-voice book-to-audiobook engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gui = subparsers.add_parser("gui", help="Launch the desktop app window.")
    gui.set_defaults(func=_cmd_gui)

    voices = subparsers.add_parser("voices", help="Create or list cloned voice states.")
    voices.set_defaults(func=_cmd_voices)
    voices_subparsers = voices.add_subparsers(dest="voices_command", required=True)

    voices_add = voices_subparsers.add_parser("add", help="Clone a voice from a local audio file.")
    voices_add.add_argument("--source", required=True, help="Local audio file path (no URLs or YouTube links).")
    voices_add.add_argument("--name", required=True, help="Name for the saved voice state (without extension).")
    voices_add.add_argument("--start", type=float, default=None, help="Clip start offset in seconds.")
    voices_add.add_argument("--end", type=float, default=None, help="Clip end offset in seconds.")
    voices_add.add_argument(
        "--voices-dir", default=str(_DEFAULT_VOICES_DIR), help="Directory to save the voice state in."
    )

    voices_list = voices_subparsers.add_parser("list", help="List saved cloned voices.")
    voices_list.add_argument(
        "--voices-dir", default=str(_DEFAULT_VOICES_DIR), help="Directory to list voice states from."
    )

    cast = subparsers.add_parser("cast", help="Run attribution and write an editable cast file.")
    cast.add_argument("book", help="Path to an epub/pdf/txt source file.")
    cast.add_argument("--out", help="Where to write the cast file (default: <book>.cast.yaml).")
    cast.set_defaults(func=_cmd_cast)

    render = subparsers.add_parser("render", help="Synthesize and assemble a book using its cast file.")
    render.add_argument("book", help="Path to an epub/pdf/txt source file.")
    render.add_argument("--cast", required=True, help="Path to the cast.yaml produced by `alebrijes cast`.")
    render.add_argument("--out", help="Where to write the audiobook (default: <book>.m4b).")
    render.add_argument("--pause-ms", type=int, default=250, help="Silence between lines, in milliseconds.")
    render.set_defaults(func=_cmd_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
