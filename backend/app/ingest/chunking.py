"""Turn extracted page text into overlapping, heading-aware chunks."""

import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
BULLETS = re.compile(r"^[•●▪‣\-\*•]\s*")


@dataclass
class Paragraph:
    page: int
    text: str
    is_heading: bool = False


@dataclass
class Chunk:
    index: int
    page: int
    heading: str | None
    content: str


def clean(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)            # words hyphenated across lines
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


NOT_HEADING = re.compile(r"^(note|example|attn|tip|important|source)\b|\d\s*$", re.I)


def is_index_page(text: str) -> bool:
    """Tables of contents and indexes: most lines end in a page number. They add noise to search."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    return len(lines) >= 8 and sum(bool(re.search(r"\d\s*$", ln)) for ln in lines) / len(lines) >= 0.4


def looks_like_heading(line: str) -> bool:
    words = line.split()
    if NOT_HEADING.search(line):
        return False
    if not 1 <= len(words) <= 12 or len(line) > 90:
        return False
    if line.endswith((".", ",", ";", ":")) or BULLETS.match(line):
        return False
    if re.fullmatch(r"[\d\s.]+", line):                 # page numbers
        return False
    capitalised = sum(w[0].isupper() for w in words if w[0].isalpha())
    alpha_words = sum(1 for w in words if w[0].isalpha())
    return line.endswith("?") or (alpha_words > 0 and capitalised / alpha_words >= 0.6)


def short_heading(line: str, next_line: str) -> bool:
    """Sentence-case headings ("What hospice covers"): a short line with no closing punctuation that
    starts a paragraph and is followed by a line starting with a capital. A line wrapped mid-sentence
    is followed by a lowercase continuation instead."""
    words = line.split()
    return (2 <= len(words) <= 8 and not NOT_HEADING.search(line) and len(line) <= 60 and line[0].isupper()
            and not line.endswith((".", ",", ";", ":", "-")) and not BULLETS.match(line)
            and bool(next_line) and next_line[0].isupper())


def paragraphs(pages: list[tuple[int, str]]) -> list[Paragraph]:
    """Group PDF lines into paragraphs: a blank line, a bullet, or a heading starts a new one."""
    out: list[Paragraph] = []
    for page, raw in pages:
        if is_index_page(raw):
            continue
        buf: list[str] = []

        def flush():
            if buf:
                out.append(Paragraph(page, " ".join(buf)))
                buf.clear()

        lines = [ln.strip() for ln in clean(raw).splitlines()]
        for i, line in enumerate(lines):
            nxt = next((ln for ln in lines[i + 1:] if ln), "")
            # PDF text rarely has blank lines between paragraphs, so a heading may follow the previous
            # paragraph directly; accept it when that paragraph ended a sentence.
            after_sentence = not buf or buf[-1].endswith((".", "?", "!", ")", "”"))
            if not line:
                flush()
            elif after_sentence and (looks_like_heading(line) or short_heading(line, nxt)):
                flush()
                out.append(Paragraph(page, line, is_heading=True))
            elif BULLETS.match(line):
                flush()
                buf.append("- " + BULLETS.sub("", line))
            else:
                buf.append(line)
        flush()
    return [p for p in out if p.text and not re.fullmatch(r"[\d\s]+", p.text)]


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sentence in SENTENCE_END.split(text):
        if cur and len(cur) + len(sentence) + 1 > limit:
            parts.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        parts.append(cur)
    # A single sentence longer than the limit is cut on whitespace as a last resort.
    final = []
    for p in parts:
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit) or limit
            final.append(p[:cut])
            p = p[cut:].strip()
        final.append(p)
    return final


def chunk_pages(pages: list[tuple[int, str]], max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Pack paragraphs into chunks of about max_chars, carrying ~overlap chars into the next chunk.

    Each chunk remembers the page it starts on and the most recent heading, so answers can cite
    "Medicare & You, p. 34 - Part B covers".
    """
    chunks: list[Chunk] = []
    heading: str | None = None
    carry = ""                      # overlap text copied from the end of the previous chunk
    body: list[str] = []            # new text for the current chunk
    body_page = 1
    body_heading: str | None = None

    def size() -> int:
        return len(carry) + sum(len(b) + 1 for b in body)

    def emit(keep_overlap: bool):
        nonlocal carry, body
        text = "\n".join(([carry] if carry else []) + body).strip()
        if len(text) >= 40:  # skip fragments like a lone running header
            chunks.append(Chunk(len(chunks), body_page, body_heading, text))
        tail = text[-overlap:] if keep_overlap and overlap else ""
        carry = tail[tail.find(" ") + 1:] if " " in tail else tail
        body = []

    for para in paragraphs(pages):
        if para.is_heading:
            if body:
                emit(keep_overlap=False)  # a new section starts a fresh chunk
            carry = ""
            heading = para.text
            continue
        for piece in _split_long(para.text, max_chars):
            if body and size() + len(piece) > max_chars:
                emit(keep_overlap=True)
            if not body:
                body_page, body_heading = para.page, heading
            body.append(piece)
    if body:
        emit(keep_overlap=False)
    return chunks
