"""EPUB parsing: source file -> `alebrijes.ingest.Book`.

Ports the generically reusable core of pocket-tts-agent's
`examples/ci_tts_from_epub.py`: OPF/spine parsing via container.xml, HTML text
extraction, and chapter-header detection (including the image-alt-text
fallback some EPUBs need). Left behind: that file's `--book` multi-book
omnibus filtering — it was built for one specific omnibus EPUB in the old
repo's `input/` folder (numbered top-level directories per sub-book), not
general EPUB support, and can be ported later if a real omnibus book needs it.
"""

from __future__ import annotations

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from alebrijes.ingest import Book, Chapter, Paragraph

_CHAPTER_HEADER_RE = re.compile(
    r"^\s*(?:chapter|chap\.?)\s*(\d+)\b|^\s*(\d+)\s*[:.]\s+", re.IGNORECASE
)
_IMG_ALT_RE = re.compile(r"<img\b[^>]*\balt\s*=\s*\"([^\"]*)\"", re.IGNORECASE)
_IMG_ALT_CHAPTER_RE = re.compile(r"^\s*(\d+)\s+\S")
_DC_TITLE_RE = re.compile(r"<dc:title[^>]*>(.*?)</dc:title>", re.IGNORECASE | re.DOTALL)


class _BodyTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "br",
        "tr",
        "blockquote",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def _extract_text_from_html(raw_html: bytes) -> str:
    parser = _BodyTextParser()
    parser.feed(raw_html.decode("utf-8", errors="ignore"))
    parser.close()
    return parser.get_text()


def _opf_path_from_container(archive: zipfile.ZipFile) -> str:
    try:
        container_xml = archive.read("META-INF/container.xml")
    except KeyError as exc:
        raise ValueError("EPUB is missing META-INF/container.xml") from exc

    root = ET.fromstring(container_xml)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = root.find(".//c:rootfile", ns)
    if rootfile is None:
        raise ValueError("EPUB container.xml has no rootfile entry")

    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        raise ValueError("EPUB rootfile is missing full-path")
    return opf_path


def _spine_documents(archive: zipfile.ZipFile, opf_path: str) -> list[str]:
    opf_xml = archive.read(opf_path)
    root = ET.fromstring(opf_xml)
    ns = {"opf": "http://www.idpf.org/2007/opf"}

    manifest: dict[str, str] = {}
    for item in root.findall(".//opf:manifest/opf:item", ns):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        media_type = item.attrib.get("media-type", "")
        if item_id and href and ("html" in media_type or href.endswith((".xhtml", ".html"))):
            manifest[item_id] = href

    opf_dir = str(Path(opf_path).parent)
    if opf_dir == ".":
        opf_dir = ""

    spine_docs: list[str] = []
    for itemref in root.findall(".//opf:spine/opf:itemref", ns):
        item_id = itemref.attrib.get("idref")
        if not item_id or item_id not in manifest:
            continue
        spine_docs.append(str((Path(opf_dir) / manifest[item_id]).as_posix()))

    if not spine_docs:
        raise ValueError("No readable HTML/XHTML spine documents found in EPUB")
    return spine_docs


def _book_title(archive: zipfile.ZipFile, opf_path: str, fallback: str) -> str:
    match = _DC_TITLE_RE.search(archive.read(opf_path).decode("utf-8", errors="ignore"))
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return fallback


def _chapter_number_from_img_alt(raw_html: bytes) -> int | None:
    match = _IMG_ALT_RE.search(raw_html.decode("utf-8", errors="ignore"))
    if not match:
        return None
    alt_match = _IMG_ALT_CHAPTER_RE.match(match.group(1))
    if not alt_match:
        return None
    return int(alt_match.group(1))


def _chapter_title(archive: zipfile.ZipFile, doc_path: str) -> str:
    raw_html = archive.read(doc_path)
    alt_match = _IMG_ALT_RE.search(raw_html.decode("utf-8", errors="ignore"))
    if alt_match:
        alt_text = alt_match.group(1).strip()
        num_match = _IMG_ALT_CHAPTER_RE.match(alt_text)
        if num_match:
            return alt_text[num_match.end(1) :].strip()
        if alt_text:
            return alt_text
    text = _extract_text_from_html(raw_html)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            header_match = _CHAPTER_HEADER_RE.match(stripped)
            return stripped[header_match.end() :].strip(" :.-") if header_match else stripped
    return ""


def _chapter_start_indices(archive: zipfile.ZipFile, spine_docs: list[str]) -> dict[int, int]:
    """Maps chapter number -> spine index of that chapter's first document.

    A table-of-contents page will match many chapter-header-shaped lines at
    once (one per chapter listed); only pages with exactly one match are
    treated as a real chapter start, not a listing page.
    """
    chapter_to_spine_index: dict[int, int] = {}
    for idx, doc_path in enumerate(spine_docs):
        try:
            raw_html = archive.read(doc_path)
        except KeyError:
            continue
        text = _extract_text_from_html(raw_html)
        found = False
        if text:
            matches: list[int] = []
            for line in text.splitlines():
                match = _CHAPTER_HEADER_RE.match(line)
                if match:
                    num_str = next((g for g in match.groups() if g), None)
                    if num_str:
                        matches.append(int(num_str))
            if len(matches) == 1:
                chapter_to_spine_index.setdefault(matches[0], idx)
                found = True
        if not found:
            chapter_num = _chapter_number_from_img_alt(raw_html)
            if chapter_num is not None:
                chapter_to_spine_index.setdefault(chapter_num, idx)
    return chapter_to_spine_index


def _paragraphs_for_range(archive: zipfile.ZipFile, spine_docs: list[str], start: int, stop: int) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    for i in range(start, stop):
        try:
            raw_html = archive.read(spine_docs[i])
        except KeyError:
            continue
        text = _extract_text_from_html(raw_html)
        for line in text.splitlines():
            if line:
                paragraphs.append(Paragraph(text=line, index=len(paragraphs)))
    return paragraphs


def parse(path: Path) -> Book:
    if not path.exists():
        raise FileNotFoundError(f"EPUB not found: {path}")

    with zipfile.ZipFile(path, "r") as archive:
        opf_path = _opf_path_from_container(archive)
        spine_docs = _spine_documents(archive, opf_path)
        book_title = _book_title(archive, opf_path, fallback=path.stem)

        chapter_to_index = _chapter_start_indices(archive, spine_docs)
        if not chapter_to_index:
            paragraphs = _paragraphs_for_range(archive, spine_docs, 0, len(spine_docs))
            return Book(title=book_title, chapters=[Chapter(title=book_title, index=0, paragraphs=paragraphs)])

        ordered = sorted(chapter_to_index.items())
        chapters: list[Chapter] = []
        for position, (_chapter_num, start_idx) in enumerate(ordered):
            stop_idx = ordered[position + 1][1] if position + 1 < len(ordered) else len(spine_docs)
            title = _chapter_title(archive, spine_docs[start_idx])
            paragraphs = _paragraphs_for_range(archive, spine_docs, start_idx, stop_idx)
            chapters.append(Chapter(title=title, index=position, paragraphs=paragraphs))

        return Book(title=book_title, chapters=chapters)
