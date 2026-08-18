"""Shared helper for building tiny real PDFs in tests (via reportlab).

Not a test module itself (no test_ prefix). reportlab is a dev-only
dependency — see pyproject.toml — used purely to generate throwaway fixture
PDFs so tests exercise the real pypdf integration path, without checking in
a binary fixture (Phase 0's "no committed media" rule).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def build_pdf(path: Path, pages: list[list[str]], encrypt_user_password: str | None = None) -> None:
    """`pages` is a list of pages, each a list of literal text lines to draw."""
    kwargs = {}
    if encrypt_user_password is not None:
        from reportlab.lib.pdfencrypt import StandardEncryption

        kwargs["encrypt"] = StandardEncryption(encrypt_user_password, canPrint=1)

    c = canvas.Canvas(str(path), pagesize=letter, **kwargs)
    width, height = letter
    for page_lines in pages:
        y = height - 72
        for line in page_lines:
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
