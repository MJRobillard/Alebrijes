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

from alebrijes.config import AUDIO_EXTENSIONS, Config

_DEFAULT_VOICES_DIR = Config().voices_dir


def _ingest_book(book_path: Path):
    from alebrijes.ingest import parse_book

    return parse_book(book_path)


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
    if source_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"'{source_path}' doesn't look like an audio file (expected one of: "
            f"{', '.join(sorted(AUDIO_EXTENSIONS))}). Voice cloning only accepts local audio "
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


def _cmd_chapters(args: argparse.Namespace) -> int:
    print(len(_ingest_book(Path(args.book)).chapters))
    return 0


def _cmd_render_chapter(args: argparse.Namespace) -> int:
    import os

    from alebrijes.cast import load
    from alebrijes.render import build_render_agent, part_paths_for, render_chapter_part

    book = _ingest_book(Path(args.book))
    index = args.chapter if args.chapter is not None else int(os.environ.get("JOB_COMPLETION_INDEX", "-1"))
    if not 0 <= index < len(book.chapters):
        raise SystemExit(
            f"Chapter index {index} out of range: '{book.title}' has {len(book.chapters)} chapters "
            "(pass --chapter, or run as a Kubernetes Indexed Job pod, which sets JOB_COMPLETION_INDEX)."
        )

    # Checked here as well as in render_chapter_part so a re-run pod skips
    # loading the TTS model too, not just the synthesis.
    if all(path.exists() for path in part_paths_for(Path(args.parts_dir), book.chapters[index].index)):
        print(f"Chapter {index + 1}/{len(book.chapters)} already rendered, skipping")
        return 0

    cast_file = load(Path(args.cast))
    agent, voice_states = build_render_agent(cast_file)
    meta_path = render_chapter_part(
        book.chapters[index],
        cast_file,
        agent,
        voice_states,
        Path(args.parts_dir),
        pause_ms=args.pause_ms,
        script_format=book.source_format == "script",
        book_title=book.title,
    )
    print(f"Chapter {index + 1}/{len(book.chapters)} done: {meta_path}")
    return 0


def _cmd_assemble(args: argparse.Namespace) -> int:
    from alebrijes.render import assemble_parts

    book_path = Path(args.book)
    book = _ingest_book(book_path)
    out_path = Path(args.out) if args.out else book_path.with_suffix(".m4b")
    result_path = assemble_parts(book, Path(args.parts_dir), out_path)
    print(f"Wrote audiobook: {result_path}")
    return 0


def _cmd_rag(args: argparse.Namespace) -> int:
    if args.rag_command == "build":
        return _cmd_rag_build(args)
    raise NotImplementedError(f"`alebrijes rag {args.rag_command}` is not implemented.")


def _cmd_rag_build(args: argparse.Namespace) -> int:
    from alebrijes.llm.ollama import OllamaClient
    from alebrijes.rag import save as save_rag
    from alebrijes.rag.build import build_rag_index
    from alebrijes.transcript import load as load_transcript

    transcript_path = Path(args.transcript)
    transcript = load_transcript(transcript_path)

    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        name = transcript_path.name
        stem = name[: -len(".transcript.json")] if name.endswith(".transcript.json") else transcript_path.stem
        out_prefix = str(transcript_path.parent / stem)

    def on_progress(done: int, total: int) -> None:
        print(f"Embedding span {done}/{total}", end="\r")

    client = OllamaClient()
    index = build_rag_index(transcript, client.embed, on_progress=on_progress)
    print()

    npz_path = Path(f"{out_prefix}.rag.npz")
    json_path = Path(f"{out_prefix}.rag.json")
    save_rag(index, npz_path, json_path)

    print(f"Wrote RAG index: {npz_path}, {json_path}")
    print(f"Spans indexed: {len(index.entries)}")
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

    # Distributed render: `render` split into its parallel per-chapter step
    # and its final stitch step, e.g. one Kubernetes Indexed Job pod per
    # chapter followed by one assemble Job (see deploy/k8s/).
    chapters = subparsers.add_parser("chapters", help="Print how many chapters a book parses into.")
    chapters.add_argument("book", help="Path to an epub/pdf/txt source file.")
    chapters.set_defaults(func=_cmd_chapters)

    render_chapter = subparsers.add_parser("render-chapter", help="Render one chapter to a parts directory.")
    render_chapter.add_argument("book", help="Path to an epub/pdf/txt source file.")
    render_chapter.add_argument("--cast", required=True, help="Path to the cast.yaml produced by `alebrijes cast`.")
    render_chapter.add_argument("--parts-dir", required=True, help="Directory shared by every chapter's part.")
    render_chapter.add_argument(
        "--chapter", type=int, help="0-based chapter index (default: $JOB_COMPLETION_INDEX, set by Kubernetes)."
    )
    render_chapter.add_argument("--pause-ms", type=int, default=250, help="Silence between lines, in milliseconds.")
    render_chapter.set_defaults(func=_cmd_render_chapter)

    assemble = subparsers.add_parser("assemble", help="Stitch rendered chapter parts into one .m4b.")
    assemble.add_argument("book", help="Path to the same source file the parts were rendered from.")
    assemble.add_argument("--parts-dir", required=True, help="Directory `render-chapter` wrote parts to.")
    assemble.add_argument("--out", help="Where to write the audiobook (default: <book>.m4b).")
    assemble.set_defaults(func=_cmd_assemble)

    rag = subparsers.add_parser("rag", help="Build a RAG index for interactive narration.")
    rag.set_defaults(func=_cmd_rag)
    rag_subparsers = rag.add_subparsers(dest="rag_command", required=True)

    rag_build = rag_subparsers.add_parser("build", help="Embed a book's transcript into a searchable RAG index.")
    rag_build.add_argument(
        "--transcript", required=True, help="Path to the <book>.transcript.json produced by `alebrijes render`."
    )
    rag_build.add_argument(
        "--out-prefix",
        help="Prefix for the output <prefix>.rag.npz/.rag.json (default: derived from --transcript).",
    )
    rag_build.set_defaults(func=_cmd_rag_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Must happen before anything imports huggingface_hub (transitively,
    # before anything imports pocket_tts) -- see hf_bundle.py's docstring.
    from alebrijes.hf_bundle import configure_bundled_hf_cache

    configure_bundled_hf_cache()

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
