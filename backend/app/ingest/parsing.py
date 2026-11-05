"""Extract text per page from PDF, Markdown and plain-text files."""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".md", ".txt"}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] with 1-based page numbers. Text files count as one page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    if suffix in {".md", ".txt"}:
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED)}")


def title_from(path: Path, pages: list[tuple[int, str]]) -> str:
    """Use the first substantial line of the document, falling back to the file name."""
    for _, text in pages[:2]:
        for line in text.splitlines():
            line = line.strip()
            if 8 <= len(line) <= 90 and any(c.isalpha() for c in line):
                return line
    return path.stem.replace("-", " ").replace("_", " ").title()
