import argparse
import json
import os
import re
from typing import Dict, List

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def _extract_pdf_chunks(pdf_path: str, chunk_chars: int, overlap_chars: int, min_chars: int) -> List[Dict[str, object]]:
    if PdfReader is None:
        raise ImportError("Missing dependency `pypdf`. Install requirements first.")

    reader = PdfReader(pdf_path)
    source_file = os.path.basename(pdf_path)
    records: List[Dict[str, object]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = _normalize_text(page.extract_text() or "")
        if len(page_text) < min_chars:
            continue
        page_chunks = _chunk_text_blocks(page_text, chunk_chars, overlap_chars)
        for local_idx, chunk in enumerate(page_chunks):
            if len(chunk) < min_chars:
                continue
            records.append(
                {
                    "source": source_file,
                    "page": page_idx,
                    "chunk_id": len(records),
                    "page_chunk_index": local_idx,
                    "char_count": len(chunk),
                    "text": chunk,
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
    args = parser.parse_args()

    input_pdf = os.path.abspath(args.input_pdf)
    output_jsonl = os.path.abspath(args.output_jsonl)
    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    chunk_chars = max(300, int(args.chunk_chars))
    overlap_chars = max(0, min(int(args.overlap_chars), chunk_chars - 1))
    min_chars = max(20, int(args.min_chars))

    records = _extract_pdf_chunks(input_pdf, chunk_chars, overlap_chars, min_chars)
    if not records:
        raise ValueError("No valid passages extracted from PDF.")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} passages -> {output_jsonl}")


if __name__ == "__main__":
    main()
