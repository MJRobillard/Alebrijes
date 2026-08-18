"""Shared helper for building tiny synthetic EPUBs in tests.

Not a test module itself (no test_ prefix) — imported by tests that need a
real, valid EPUB zip without checking in a binary fixture (Phase 0's "no
committed media" rule).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _opf(book_title: str, doc_names: list[str]) -> str:
    manifest_items = "\n".join(
        f'<item id="doc{i}" href="{name}" media-type="application/xhtml+xml"/>' for i, name in enumerate(doc_names)
    )
    spine_items = "\n".join(f'<itemref idref="doc{i}"/>' for i in range(len(doc_names)))
    return f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{book_title}</dc:title>
  </metadata>
  <manifest>
    {manifest_items}
  </manifest>
  <spine>
    {spine_items}
  </spine>
</package>
"""


def build_epub(path: Path, book_title: str, docs: dict[str, str]) -> None:
    """`docs` maps spine filename -> raw XHTML body content (already includes tags)."""
    doc_names = list(docs)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", _opf(book_title, doc_names))
        for name, body in docs.items():
            archive.writestr(f"OEBPS/{name}", f"<html><body>{body}</body></html>")
