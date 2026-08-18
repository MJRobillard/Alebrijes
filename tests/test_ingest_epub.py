from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from alebrijes.ingest.epub import parse
from epub_fixtures import build_epub as _build_epub


def test_parse_extracts_chapter_titles_and_paragraphs(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {
            "ch1.xhtml": "<p>Chapter 1: The Beginning</p><p>Colin said, \"Let's go.\"</p>",
            "ch2.xhtml": "<p>Chapter 2: The Middle</p><p>Audrey nodded.</p>",
        },
    )

    book = parse(epub_path)

    assert book.title == "Fixture Book"
    assert len(book.chapters) == 2
    assert book.chapters[0].title == "The Beginning"
    assert book.chapters[1].title == "The Middle"
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "Chapter 1: The Beginning",
        'Colin said, "Let\'s go."',
    ]


def test_parse_falls_back_to_single_chapter_when_no_headers_found(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Untitled Fixture",
        {"page1.xhtml": "<p>Just some narration with no chapter headings anywhere.</p>"},
    )

    book = parse(epub_path)

    assert len(book.chapters) == 1
    assert book.chapters[0].title == "Untitled Fixture"
    assert book.chapters[0].paragraphs[0].text == "Just some narration with no chapter headings anywhere."


def test_parse_strips_html_tags_and_collapses_whitespace(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": "<p>Chapter 1</p><p>Some   <b>bold</b>   text   with   spaces.</p>"},
    )

    book = parse(epub_path)

    assert book.chapters[0].paragraphs[1].text == "Some bold text with spaces."


def test_parse_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse(tmp_path / "does_not_exist.epub")


def test_parse_uses_image_alt_text_chapter_heading_fallback(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {
            "ch1.xhtml": '<img alt="1 Only the Dark"/><p>The story begins.</p>',
            "ch2.xhtml": '<img alt="2 A New Dawn"/><p>The story continues.</p>',
        },
    )

    book = parse(epub_path)

    assert len(book.chapters) == 2
    assert book.chapters[0].title == "Only the Dark"
    assert book.chapters[1].title == "A New Dawn"


def test_parse_rejects_drm_protected_epub_with_clear_error(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": "<p>Chapter 1</p><p>Some narration text here.</p>"},
        extra_files={"META-INF/encryption.xml": "<encryption/>"},
    )

    with pytest.raises(ValueError, match="DRM"):
        parse(epub_path)


def test_parse_raises_clear_error_for_malformed_opf_xml(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    container_xml = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("OEBPS/content.opf", "<package><manifest not closed properly")

    with pytest.raises(ValueError, match="not valid XML"):
        parse(epub_path)


def test_parse_raises_clear_error_when_zip_is_not_a_valid_archive(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"this is not a zip file at all")

    with pytest.raises(ValueError, match="not a valid EPUB"):
        parse(epub_path)


def test_parse_skips_manifest_entries_missing_from_archive(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    # A manifest/spine entry ("ghost") points at a file that was never
    # actually written into the archive — a documented ebooklib issue class.
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture Book</dc:title></metadata>
  <manifest>
    <item id="ghost" href="missing.xhtml" media-type="application/xhtml+xml"/>
    <item id="doc0" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ghost"/>
    <itemref idref="doc0"/>
  </spine>
</package>
"""
    container_xml = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/ch1.xhtml", "<html><body><p>Chapter 1</p><p>Real content.</p></body></html>")

    book = parse(epub_path)

    assert book.chapters[0].paragraphs[-1].text == "Real content."


def test_parse_normalizes_ligatures_to_ascii(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": "<p>Chapter 1</p><p>A diﬃcult waﬀe recipe.</p>"},
    )

    book = parse(epub_path)

    assert book.chapters[0].paragraphs[1].text == "A difficult waffe recipe."
