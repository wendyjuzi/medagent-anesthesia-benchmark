import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


Q_COLON_RE = r"[:\uFF1A]"
CN_REASON = "\u3010\u4E34\u5E8A\u63A8\u7406\u3011"
CN_DECISION = "\u3010\u51B3\u7B56\u5E72\u9884\u3011"
CN_DECISION_MILLER = "\u3010\u51B3\u7B56\u5E72\u9884\uFF08Miller\uFF09\u3011"
CN_DECISION_VITALDB = "\u3010\u51B3\u7B56\u5E72\u9884\uFF08VitalDB\uFF09\u3011"
LEAK_TOKEN_RE = re.compile(
    r"(?is)\b("
    r"wait|strategy|constraint\s*check|analyze\s+the\s+input\s+data|"
    r"self-?correction|content\s+requirements|drafting|thinking\s+process|analysis:"
    r")\b"
)


def _clean_raw_output(text: str) -> str:
    out = text.strip()
    out = re.sub(r"^```(?:json|markdown|text)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```$", "", out)
    out = out.replace("\r\n", "\n").strip()
    return out


def _extract_qa_block(text: str) -> str:
    out = _clean_raw_output(text)

    # Remove think blocks when present.
    out = re.sub(r"(?is)<think>.*?</think>", "", out).strip()
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()

    # Keep only the last Q: section, avoiding draft Q lines in reasoning traces.
    q_matches = list(re.finditer(rf"(?im)^Q\s*{Q_COLON_RE}", out))
    if q_matches:
        out = out[q_matches[-1].start() :].strip()

    # Strict capture of final QA block.
    strict = re.search(
        rf"(Q\s*{Q_COLON_RE}.*?A\s*{Q_COLON_RE}.*?(?:{CN_DECISION_MILLER}|{CN_DECISION_VITALDB}|{CN_DECISION}|\[Decision Intervention\]).*?(?=\n\n|\n\*|\Z))",
        out,
        re.IGNORECASE | re.DOTALL,
    )
    if strict:
        return strict.group(1).strip()

    # Fallback: if Q and A exist, keep from Q onward.
    if re.search(rf"(?im)^Q\s*{Q_COLON_RE}", out) and re.search(rf"(?im)^A\s*{Q_COLON_RE}", out):
        return out.strip()

    return out.strip()


def _is_strict_qa(text: str) -> bool:
    out = text.strip()
    low = out.lower()
    banned = ["<think>", "</think>", "let's think", "**content requirements**", "**strategy**"]
    if LEAK_TOKEN_RE.search(low):
        return False
    if any(x in low for x in banned):
        return False
    if re.search(r"(?im)^\s*(\*|-|\d+\.)\s+", out):
        return False
    if not re.search(rf"(?im)^Q\s*{Q_COLON_RE}", out):
        return False
    if not re.search(rf"(?im)^A\s*{Q_COLON_RE}", out):
        return False
    has_reason = (CN_REASON in out) or ("[Clinical Reasoning]" in out)
    has_decision_dual = (CN_DECISION_MILLER in out) and (CN_DECISION_VITALDB in out)
    if (not has_reason) or (not has_decision_dual):
        return False
    if out.endswith(("\uFF1A", ":", "\uFF0C", ",", "\u3001", "\uFF08", "(", "\u3010", "[")):
        return False
    if "\ufffd" in out:
        return False
    if out.count("\u3010") != out.count("\u3011"):
        return False
    if out.count("\uFF08") != out.count("\uFF09"):
        return False
    if out.count("(") != out.count(")"):
        return False
    if not re.search(r"[\u3002\uFF01\uFF1F.!?]$", out):
        return False
    return True


def clean_record(rec: Dict[str, Any], field: str) -> Tuple[Dict[str, Any], bool, bool]:
    original = rec.get(field)
    if not isinstance(original, str) or not original.strip():
        return rec, False, False

    cleaned = _extract_qa_block(original)
    changed = cleaned != original
    strict_ok = _is_strict_qa(cleaned)
    rec[field] = cleaned
    return rec, changed, strict_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean QA JSONL and remove CoT leakage")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", default="", help="Output JSONL path. Default: <input>.cleaned.jsonl")
    parser.add_argument("--field", default="llm_output", help="Text field to clean")
    parser.add_argument("--drop-invalid", action="store_true", help="Drop rows whose cleaned QA is still invalid")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + ".cleaned.jsonl")

    total = 0
    changed = 0
    strict_ok = 0
    dropped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            rec, did_change, is_valid = clean_record(rec, args.field)
            if did_change:
                changed += 1
            if is_valid:
                strict_ok += 1
            elif args.drop_invalid:
                dropped += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== QA JSONL Cleaner Report ===")
    print(f"input:   {input_path}")
    print(f"output:  {output_path}")
    print(f"total:   {total}")
    print(f"changed: {changed}")
    print(f"strict:  {strict_ok}")
    print(f"dropped: {dropped}")


if __name__ == "__main__":
    main()
