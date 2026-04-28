import argparse
import json
import os
import re
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_heading_line(line: str) -> str:
    text = _normalize_text(line)
    if not re.match(r"(?i)^chapter\s+\d{1,3}\b", text):
        text = re.sub(r"\s+\d{2,4}\s+(?=[A-Za-z]).*$", "", text).strip()
    text = re.sub(r"\s*\d{1,4}\s*$", "", text).strip()  # drop trailing page number
    return text


def _normalize_chapter_label(number: str, title: str) -> str:
    num = str(number or "").strip()
    ttl = _clean_heading_line(title)
    if not num:
        return ttl
    return f"{num} {ttl}".strip()


def _chapter_parts_from_heading(line: str) -> Tuple[str, str]:
    text = _clean_heading_line(line)
    patterns = [
        r"(?i)^chapter\s*(\d{1,3})\s*[:.\-–]?\s*(.+)$",
        r"^(\d{1,3})\s*[•·.\-\–]\s*([A-Za-z][A-Za-z0-9 ,:&()/\-'’]{3,180})$",
        r"^(\d{1,3})\s+([A-Z][A-Za-z0-9 ,:&()/\-'’]{3,180})$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        number = m.group(1).strip()
        title = _clean_heading_line(m.group(2))
        try:
            number_i = int(number)
        except ValueError:
            return "", ""
        if number_i <= 0 or number_i > 120:
            return "", ""
        if not re.search(r"[A-Za-z]{3,}", title):
            return "", ""
        # Reject accidental body fragments such as "441 This characteristic ..."
        if number_i > 120 or title.split(" ", 1)[0].lower() in {
            "this",
            "these",
            "they",
            "there",
            "when",
            "however",
            "conversely",
            "more",
            "although",
            "because",
        }:
            return "", ""
        return _normalize_chapter_label(number, title), title
    return "", ""


def _is_noise_heading(line: str) -> bool:
    low = _clean_heading_line(line).lower()
    if not low:
        return True
    noise = ("references", "contents", "index", "acknowledgments", "copyright")
    return any(tok in low for tok in noise)


def _extract_chapter_section_state(
    text: str,
    current_chapter: str,
    current_section: str,
) -> Tuple[str, str]:
    chapter = str(current_chapter or "").strip()
    section = str(current_section or "").strip()
    lines = [_clean_heading_line(x) for x in str(text or "").splitlines()]
    lines = [x for x in lines if x]

    for line in lines[:40]:
        if _is_noise_heading(line):
            continue
        candidate_chapter, candidate_section = _chapter_parts_from_heading(line)
        if candidate_chapter:
            chapter = candidate_chapter
            section = candidate_section
            return chapter, section

    # Soft section update: keep chapter unchanged, refresh section when clear heading appears.
    for line in lines[:40]:
        if _is_noise_heading(line):
            continue
        if len(line) < 6 or len(line) > 120:
            continue
        if "." in line:
            continue
        # Title-like line
        if re.match(r"^[A-Z][A-Za-z0-9 ,:&()/\-]{5,120}$", line):
            section = line
            break

    return chapter, section


def _build_structure_prefix(book_title: str, chapter: str, section: str) -> str:
    title = str(book_title or "").strip() or "Miller's Anesthesia"
    parts = [f"书籍: {title}"]
    if str(chapter or "").strip():
        parts.append(f"章节: {str(chapter).strip()}")
    if str(section or "").strip():
        parts.append(f"子节: {str(section).strip()}")
    return "[" + ", ".join(parts) + "]"


def _iter_outline_items(items: Any, depth: int = 0) -> List[Tuple[Any, int]]:
    out: List[Tuple[Any, int]] = []
    if not items:
        return out
    for item in items:
        if isinstance(item, list):
            out.extend(_iter_outline_items(item, depth + 1))
        else:
            out.append((item, depth))
    return out


def _extract_outline_chapter_ranges(reader: Any) -> List[Dict[str, object]]:
    try:
        outline = getattr(reader, "outline", None) or getattr(reader, "outlines", None)
    except Exception:
        outline = None
    ranges: List[Dict[str, object]] = []
    if not outline:
        return ranges

    seen: Dict[str, Dict[str, object]] = {}
    for item, depth in _iter_outline_items(outline):
        title = _clean_heading_line(str(getattr(item, "title", "") or ""))
        chapter, section = _chapter_parts_from_heading(title)
        if not chapter:
            continue
        try:
            page_index = int(reader.get_destination_page_number(item)) + 1
        except Exception:
            continue
        number = chapter.split(" ", 1)[0]
        candidate = {
            "start_page": page_index,
            "chapter": chapter,
            "section": section,
            "depth": depth,
            "chapter_source": "pdf_outline",
            "chapter_confidence": 0.95,
        }
        previous = seen.get(number)
        if previous is None or (depth, page_index) < (int(previous.get("depth", 999)), int(previous.get("start_page", 10**9))):
            seen[number] = candidate

    ranges = sorted(seen.values(), key=lambda x: int(x["start_page"]))
    return ranges


def _chapter_for_page(page_idx: int, outline_ranges: List[Dict[str, object]]) -> Dict[str, object]:
    if not outline_ranges:
        return {}
    starts = [int(x["start_page"]) for x in outline_ranges]
    pos = bisect_right(starts, int(page_idx)) - 1
    if pos < 0:
        return {}
    return outline_ranges[pos]


def _page_label(reader: Any, page_idx: int) -> str:
    labels = getattr(reader, "page_labels", None)
    if isinstance(labels, list) and 0 <= page_idx - 1 < len(labels):
        label = str(labels[page_idx - 1] or "").strip()
        if label:
            return label
    return str(page_idx)


def _chunk_text_blocks(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    raw_blocks = [blk.strip() for blk in re.split(r"\n\s*\n+", text) if blk.strip()]
    if not raw_blocks:
        raw_blocks = [text.strip()] if text.strip() else []
    if not raw_blocks:
        return []

    chunks: List[str] = []
    current = ""
    for block in raw_blocks:
        block = _normalize_text(block)
        if not block:
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if current and len(candidate) > chunk_chars:
            chunks.append(current.strip())
            carry = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{carry} {block}".strip() if carry else block
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def _extract_pdf_chunks(
    pdf_path: str,
    chunk_chars: int,
    overlap_chars: int,
    min_chars: int,
    book_title: str,
    inject_structure_prefix: bool,
) -> List[Dict[str, object]]:
    if PdfReader is None:
        raise ImportError("Missing dependency `pypdf`. Install requirements first.")

    reader = PdfReader(pdf_path)
    source_file = os.path.basename(pdf_path)
    records: List[Dict[str, object]] = []
    current_chapter = ""
    current_section = ""
    current_chapter_source = ""
    current_chapter_confidence = 0.0
    outline_ranges = _extract_outline_chapter_ranges(reader)

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_page_text = page.extract_text() or ""
        outline_chapter = _chapter_for_page(page_idx, outline_ranges)
        if outline_chapter:
            current_chapter = str(outline_chapter.get("chapter") or "")
            current_section = str(outline_chapter.get("section") or "")
            current_chapter_source = str(outline_chapter.get("chapter_source") or "pdf_outline")
            current_chapter_confidence = float(outline_chapter.get("chapter_confidence") or 0.95)
        fallback_chapter, fallback_section = _extract_chapter_section_state(
            raw_page_text,
            current_chapter=current_chapter,
            current_section=current_section,
        )
        if not outline_chapter and fallback_chapter:
            current_chapter = fallback_chapter
            current_section = fallback_section
            current_chapter_source = "text_heading"
            current_chapter_confidence = 0.55

        page_text = _normalize_text(raw_page_text)
        if len(page_text) < min_chars:
            continue
        page_chunks = _chunk_text_blocks(raw_page_text, chunk_chars, overlap_chars)
        for local_idx, chunk in enumerate(page_chunks):
            if len(chunk) < min_chars:
                continue
            fallback_chapter, fallback_section = _extract_chapter_section_state(
                chunk,
                current_chapter=current_chapter,
                current_section=current_section,
            )
            if not outline_chapter and fallback_chapter:
                current_chapter = fallback_chapter
                current_section = fallback_section
                current_chapter_source = "text_heading"
                current_chapter_confidence = 0.55
            text_payload = chunk
            if inject_structure_prefix:
                prefix = _build_structure_prefix(book_title, current_chapter, current_section)
                text_payload = f"{prefix}\n{chunk}"
            page_label = _page_label(reader, page_idx)
            records.append(
                {
                    "source": source_file,
                    "page": page_label,
                    "pdf_page": page_idx,
                    "page_label": page_label,
                    "chunk_id": len(records),
                    "page_chunk_index": local_idx,
                    "char_count": len(chunk),
                    "chapter": current_chapter,
                    "section": current_section,
                    "chapter_source": current_chapter_source,
                    "chapter_confidence": current_chapter_confidence,
                    "display_locator": (
                        f"[M10#{len(records) + 1} | 相关章节: {current_chapter or current_section or '章节定位不足'} | p.{page_label}]"
                    ),
                    "text": text_payload,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Miller PDF into JSONL retrieval passages.")
    parser.add_argument("--input-pdf", required=True, help="Path to the Miller PDF.")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL path. Each line contains one retrievable passage.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1200,
        help="Target chunk size in characters.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=120,
        help="Drop very short extracted chunks below this length.",
    )
    parser.add_argument(
        "--book-title",
        default="Miller's Anesthesia",
        help="Book title injected into each chunk prefix.",
    )
    parser.add_argument(
        "--disable-structure-prefix",
        action="store_true",
        help="Disable chapter/section metadata prefix injection to `text`.",
    )
    args = parser.parse_args()

    input_pdf = os.path.abspath(args.input_pdf)
    output_jsonl = os.path.abspath(args.output_jsonl)
    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    chunk_chars = max(300, int(args.chunk_chars))
    overlap_chars = max(0, min(int(args.overlap_chars), chunk_chars - 1))
    min_chars = max(20, int(args.min_chars))

    records = _extract_pdf_chunks(
        input_pdf,
        chunk_chars,
        overlap_chars,
        min_chars,
        book_title=args.book_title,
        inject_structure_prefix=(not args.disable_structure_prefix),
    )
    if not records:
        raise ValueError("No valid passages extracted from PDF.")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} passages -> {output_jsonl}")


if __name__ == "__main__":
    main()
