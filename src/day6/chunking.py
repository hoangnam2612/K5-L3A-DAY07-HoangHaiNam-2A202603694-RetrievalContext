"""Implement fixed-size and structure-aware chunking."""
import bisect
import re
import statistics

from .models import Chunk, DocumentPage

_TERMINAL = ('.', '?', '!', ':', '"', "'", ')', '”', '’')
_SECTION_HEADING = re.compile(r"^\d+\.\d+ \S.{0,90}$")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def _make_chunk(page: DocumentPage, strategy: str, index: int, text: str, **extra) -> Chunk:
    """Build a Chunk with a stable ID and the page metadata needed for tracing and filtering."""
    meta = {k: page.metadata[k] for k in ("page_id", "chapter", "chapter_title", "section", "unit_id", "title", "period", "file")
            if k in page.metadata}
    meta.update({"chunking": strategy, "chunk_index": index, "n_chars": len(text)}, **extra)
    return Chunk(chunk_id=f"{page.document_id}-p{page.page:03d}-{strategy}-{index:03d}",
                 document_id=page.document_id, source=page.source, page=page.page,
                 text=text, metadata=meta)


def _add_lines(page: DocumentPage, chunks: list[Chunk]) -> list[Chunk]:
    """Record where each chunk sits on its page: metadata line_start / line_end (source line numbers).

    Lines are numbered 1.. in the text extracted from the page (before furniture removal), so a chunk can be
    traced as "page P, lines X-Y". The chunk is located in the page text with whitespace collapsed.
    """
    lm = page.metadata.get("line_map")
    if not lm:
        return chunks
    flat, positions = lm["flat"], [p for p, _ in lm["starts"]]
    numbers = [n for _, n in lm["starts"]]
    cursor = 0
    for c in chunks:
        needle = " ".join(c.text.split())
        i = flat.find(needle, cursor)
        if i < 0:
            i = flat.find(needle)
        if i < 0:
            continue  # leave the chunk without line info rather than guess
        cursor = i + 1
        last = i + len(needle) - 1
        c.metadata["line_start"] = numbers[bisect.bisect_right(positions, i) - 1]
        c.metadata["line_end"] = numbers[bisect.bisect_right(positions, last) - 1]
        c.metadata["page_lines"] = lm["total"]
    return chunks


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """TODO: split text into overlapping windows and retain page metadata."""
    # Implementation notes (added): plain character windows (may cut words/sentences on purpose,
    # this is the naive baseline); chunks never cross a page boundary, so `page` stays exact.
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size, got size={size}, overlap={overlap}")
    text, chunks, start = page.text, [], 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(_make_chunk(page, "fixed", len(chunks), piece.strip(),
                                      char_start=start, char_end=start + len(piece)))
        if start + size >= len(text):
            break
        start += size - overlap
    return _add_lines(page, chunks)


def _is_heading(paragraph: str) -> bool:
    return len(paragraph) < 90 and not paragraph.endswith(_TERMINAL)


def _split_oversized(paragraph: str, max_size: int) -> list[str]:
    """Split one long paragraph at sentence ends; hard-split on words only if a sentence is too long."""
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        while len(sentence) > max_size:  # a single sentence longer than the limit
            cut = sentence.rfind(" ", 0, max_size)
            cut = cut if cut > 0 else max_size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > max_size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """TODO: group paragraphs, split oversized paragraphs and retain metadata."""
    # Implementation notes (added): paragraphs come from normalize_text (blank-line separated).
    # A heading-like paragraph is glued to the paragraph that follows it. Consecutive blocks are
    # grouped while they fit in max_size; a block bigger than max_size is split at sentence ends.
    # `section` is updated from numbered headings ("1.2 The History ...") found inside the page,
    # so it is more precise than the page-level section copied by fixed_size.
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    section = page.metadata.get("section")
    blocks: list[tuple[str, str | None]] = []  # (text, section in force at the end of the block)
    pending_heading = ""
    in_outline = False  # "CHAPTER OUTLINE" repeats section titles; they are not real headings
    for paragraph in (p.strip() for p in page.text.split("\n\n")):
        if not paragraph:
            continue
        if paragraph == "CHAPTER OUTLINE":
            in_outline = True
        elif in_outline and _SECTION_HEADING.match(paragraph):
            pass
        else:
            in_outline = False
            if _SECTION_HEADING.match(paragraph):
                section = paragraph
        if _is_heading(paragraph):
            pending_heading = f"{pending_heading}\n\n{paragraph}".strip()
            continue
        paragraph = f"{pending_heading}\n\n{paragraph}".strip() if pending_heading else paragraph
        pending_heading = ""
        pieces = [paragraph] if len(paragraph) <= max_size else _split_oversized(paragraph, max_size)
        blocks.extend((piece, section) for piece in pieces)
    if pending_heading:  # trailing heading with no body on this page
        blocks.append((pending_heading, section))

    grouped: list[tuple[str, str | None]] = []
    for text, sec in blocks:
        if grouped and len(grouped[-1][0]) + 2 + len(text) <= max_size:
            grouped[-1] = (f"{grouped[-1][0]}\n\n{text}", sec)
        else:
            grouped.append((text, sec))

    chunks = []
    for index, (text, sec) in enumerate(grouped):
        chunk = _make_chunk(page, "para", index, text)
        if sec:
            chunk.metadata["section"] = sec
        chunks.append(chunk)
    return _add_lines(page, chunks)


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Count and length distribution of chunks, for the experiment log."""
    sizes = [len(c.text) for c in chunks]
    return {"chunks": len(sizes), "total_chars": sum(sizes), "min": min(sizes), "median": statistics.median(sizes),
            "mean": round(statistics.mean(sizes), 1), "max": max(sizes)}
