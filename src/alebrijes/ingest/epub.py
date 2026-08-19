"""EPUB parsing: source file -> `alebrijes.ingest.Book`.

Ports the generically reusable core of pocket-tts-agent's
`examples/ci_tts_from_epub.py`: OPF/spine parsing via container.xml, HTML text
extraction, and chapter-header detection (including the image-alt-text
fallback some EPUBs need). Left behind: that file's `--book` multi-book
omnibus filtering — it was built for one specific omnibus EPUB in the old
repo's `input/` folder (numbered top-level directories per sub-book), not
general EPUB support, and can be ported later if a real omnibus book needs it.

Real-world EPUBs vary more than the spec suggests, so this also guards
against the common failure modes documented by ebooklib/epubcheck issue
trackers and DRM tooling:

- **DRM** (Adobe ADEPT / similar): the zip itself still opens fine — only the
  content inside is encrypted, which would otherwise surface as a confusing
  garbled-text or XML-parse failure deep in chapter extraction. Detected up
  front via `META-INF/encryption.xml` (the same signal Calibre's DRM-removal
  tooling keys off) and reported as a clear, actionable error instead.
- **Malformed/missing container.xml or OPF XML**: reported as a clear error
  rather than a raw `xml.etree.ElementTree.ParseError`.
- **Manifest items referenced but missing from the archive** (a documented
  ebooklib issue class): skipped rather than crashing the whole parse.
- **Missing XML namespace declarations** in a malformed OPF (some
  EPUB2-generation tools omit them): manifest/spine/title lookups retry with
  a namespace wildcard before giving up.
"""

from __future__ import annotations

import re
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from alebrijes.ingest import Book, Chapter, Paragraph

_DRM_MARKER = "META-INF/encryption.xml"

_STRONG_CHAPTER_HEADER_RE = re.compile(
    r"^\s*(?:chapter|chap\.?)\s*(\d+)\b|^\s*(\d+)\s*[:.]\s+", re.IGNORECASE
)
# A line that's *just* a number (nothing else) -- some books (e.g. Random
# House's Red Rising/Golden Son EPUBs) render the chapter heading as a bare
# number on its own line, with the actual chapter title as a *separate*
# following line rather than on the same line as the number. Kept as a
# distinct, weaker pattern (only consulted when a page has zero strong
# matches, see `_chapter_start_indices`) rather than folded into
# `_CHAPTER_HEADER_RE`: some chapter bodies contain an unrelated isolated
# numeral line of their own (e.g. a running chapter-number footer), which
# would otherwise push an otherwise-good strong-format page's match count
# above one and wrongly disqualify it as a chapter start.
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,4})\s*$")
_CHAPTER_HEADER_RE = re.compile(_STRONG_CHAPTER_HEADER_RE.pattern + r"|" + _BARE_NUMBER_RE.pattern, re.IGNORECASE)
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
        lines = [
            unicodedata.normalize("NFKC", re.sub(r"\s+", " ", line)).strip() for line in text.splitlines()
        ]
        return "\n".join(line for line in lines if line)


def _extract_text_from_html(raw_html: bytes) -> str:
    parser = _BodyTextParser()
    parser.feed(raw_html.decode("utf-8", errors="ignore"))
    parser.close()
    return parser.get_text()


def _check_not_drm_protected(archive: zipfile.ZipFile) -> None:
    """Raises a clear error for DRM-protected EPUBs instead of a confusing
    garbled-text or XML-parse failure deep in chapter extraction.

    `META-INF/encryption.xml` marks per-file content encryption (Adobe ADEPT
    and similar schemes) — the same signal DRM-removal tooling like Calibre's
    DeDRM plugin keys off. The zip container and its XML metadata still open
    fine; only the chapter content itself is unreadable, which is why this
    needs an explicit check rather than surfacing naturally as an error.
    """
    if _DRM_MARKER in archive.namelist():
        raise ValueError(
            "This EPUB is DRM-protected (found META-INF/encryption.xml) and can't be read. "
            "Remove DRM first (e.g. with Calibre + a DeDRM plugin) before importing."
        )


def _parse_xml(raw_xml: bytes, description: str) -> ET.Element:
    try:
        return ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise ValueError(f"{description} is not valid XML: {exc}") from exc


def _findall_with_namespace_fallback(root: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    """Some EPUB2-generation tools omit namespace declarations in the OPF.

    Retries with a namespace wildcard before giving up, rather than silently
    returning an empty manifest/spine for an otherwise-valid file.
    """
    found = root.findall(path, ns)
    if found:
        return found
    wildcard_path = re.sub(r"\b\w+:", "{*}", path)
    return root.findall(wildcard_path)


def _opf_path_from_container(archive: zipfile.ZipFile) -> str:
    try:
        container_xml = archive.read("META-INF/container.xml")
    except KeyError as exc:
        raise ValueError("EPUB is missing META-INF/container.xml") from exc

    root = _parse_xml(container_xml, "META-INF/container.xml")
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfiles = _findall_with_namespace_fallback(root, ".//c:rootfile", ns)
    if not rootfiles:
        raise ValueError("EPUB container.xml has no rootfile entry")

    opf_path = rootfiles[0].attrib.get("full-path")
    if not opf_path:
        raise ValueError("EPUB rootfile is missing full-path")
    return opf_path


def _read_opf(archive: zipfile.ZipFile, opf_path: str) -> bytes:
    try:
        return archive.read(opf_path)
    except KeyError as exc:
        raise ValueError(f"EPUB manifest file '{opf_path}' referenced by container.xml is missing from the archive") from exc


def _spine_documents(archive: zipfile.ZipFile, opf_path: str) -> list[str]:
    root = _parse_xml(_read_opf(archive, opf_path), opf_path)
    ns = {"opf": "http://www.idpf.org/2007/opf"}

    archive_names = set(archive.namelist())
    opf_dir = str(Path(opf_path).parent)
    if opf_dir == ".":
        opf_dir = ""

    manifest: dict[str, str] = {}
    for item in _findall_with_namespace_fallback(root, ".//opf:manifest/opf:item", ns):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        media_type = item.attrib.get("media-type", "")
        if not (item_id and href and ("html" in media_type or href.endswith((".xhtml", ".html")))):
            continue
        resolved = str((Path(opf_dir) / href).as_posix())
        if resolved not in archive_names:
            # Manifest references a file that isn't actually in the archive
            # (a documented ebooklib issue class) — skip it rather than
            # crashing the whole parse over one bad entry.
            continue
        manifest[item_id] = href

    spine_docs: list[str] = []
    for itemref in _findall_with_namespace_fallback(root, ".//opf:spine/opf:itemref", ns):
        item_id = itemref.attrib.get("idref")
        if not item_id or item_id not in manifest:
            continue
        spine_docs.append(str((Path(opf_dir) / manifest[item_id]).as_posix()))

    if not spine_docs:
        raise ValueError("No readable HTML/XHTML spine documents found in EPUB")
    return spine_docs


def _book_title(archive: zipfile.ZipFile, opf_path: str, fallback: str) -> str:
    match = _DC_TITLE_RE.search(_read_opf(archive, opf_path).decode("utf-8", errors="ignore"))
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, stripped in enumerate(lines):
        header_match = _CHAPTER_HEADER_RE.match(stripped)
        if not header_match:
            # Not the header line yet -- e.g. a running book-title header
            # repeated at the top of every page before the real chapter
            # heading. Keep scanning rather than treating it as the title.
            continue
        remainder = stripped[header_match.end() :].strip(" :.-")
        if remainder:
            return remainder
        # Bare-number heading ("1") with no title text on the same line --
        # the title is the next line instead.
        return lines[i + 1] if i + 1 < len(lines) else stripped
    return lines[0] if lines else ""


def _chapter_start_indices(archive: zipfile.ZipFile, spine_docs: list[str]) -> list[int]:
    """Returns the spine indices that start a chapter, in spine order.

    A table-of-contents page will match many chapter-header-shaped lines at
    once (one per chapter listed); only pages with exactly one match are
    treated as a real chapter start, not a listing page.

    Deliberately *not* keyed/deduped by chapter number: an omnibus of
    several books in one EPUB (e.g. a trilogy bind-up) restarts numbering
    at "Chapter 1" for each sub-book, and a previous version of this
    function used chapter number as a dict key, so every sub-book's repeat
    of "Chapter 1", "Chapter 2", etc. was silently dropped as a duplicate —
    collapsing most of the omnibus into a couple of giant merged chapters.
    Spine position is always unique, so every real chapter start survives
    regardless of how its number was labeled.
    """
    indices: list[int] = []
    for idx, doc_path in enumerate(spine_docs):
        try:
            raw_html = archive.read(doc_path)
        except KeyError:
            continue
        text = _extract_text_from_html(raw_html)
        found = False
        if text:
            lines = text.splitlines()
            strong_matches = sum(1 for line in lines if _STRONG_CHAPTER_HEADER_RE.match(line))
            if strong_matches == 1:
                indices.append(idx)
                found = True
            elif strong_matches == 0:
                bare_matches = sum(1 for line in lines if _BARE_NUMBER_RE.match(line))
                if bare_matches == 1:
                    indices.append(idx)
                    found = True
        if not found:
            chapter_num = _chapter_number_from_img_alt(raw_html)
            if chapter_num is not None:
                indices.append(idx)
    return indices


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

    try:
        archive_handle = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"'{path}' is not a valid EPUB (not a readable zip archive): {exc}") from exc

    with archive_handle as archive:
        _check_not_drm_protected(archive)
        opf_path = _opf_path_from_container(archive)
        spine_docs = _spine_documents(archive, opf_path)
        book_title = _book_title(archive, opf_path, fallback=path.stem)

        start_indices = _chapter_start_indices(archive, spine_docs)
        if not start_indices:
            paragraphs = _paragraphs_for_range(archive, spine_docs, 0, len(spine_docs))
            return Book(title=book_title, chapters=[Chapter(title=book_title, index=0, paragraphs=paragraphs)])

        chapters: list[Chapter] = []
        for position, start_idx in enumerate(start_indices):
            stop_idx = start_indices[position + 1] if position + 1 < len(start_indices) else len(spine_docs)
            title = _chapter_title(archive, spine_docs[start_idx])
            paragraphs = _paragraphs_for_range(archive, spine_docs, start_idx, stop_idx)
            chapters.append(Chapter(title=title, index=position, paragraphs=paragraphs))

        return Book(title=book_title, chapters=chapters)
