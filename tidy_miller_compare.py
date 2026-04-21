import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _extract_miller_section(text: Optional[str]) -> str:
    if not text:
        return ""
    value = str(text)
    match = re.search(r"【决策干预（Miller）】[:：]\s*(.*?)(?:\n【|$)", value, re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def _one_line(text: Optional[str], max_len: int = 220) -> str:
    if not text:
        return ""
    value = " ".join(str(text).replace("\r", "\n").split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _compact_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = raw.get("snapshot") if isinstance(raw.get("snapshot"), dict) else {}
    pb = snapshot.get("patient_background") if isinstance(snapshot.get("patient_background"), dict) else {}
    ca = snapshot.get("clinical_assessment") if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    gpt = raw.get("gpt_api_miller10") if isinstance(raw.get("gpt_api_miller10"), dict) else {}
    local = raw.get("local_embedding_miller10") if isinstance(raw.get("local_embedding_miller10"), dict) else {}
    cmp_ = raw.get("comparison") if isinstance(raw.get("comparison"), dict) else {}
    anchor = snapshot.get("anchor_detail") if isinstance(snapshot.get("anchor_detail"), dict) else {}

    gpt_miller = gpt.get("miller_output") or _extract_miller_section(gpt.get("final_output"))
    local_miller = local.get("miller_output") or _extract_miller_section(local.get("final_output"))

    return {
        "index": raw.get("index"),
        "caseid": raw.get("caseid"),
        "time_sec": anchor.get("time_sec"),
        "surgery_group": pb.get("surgery_group"),
        "surgery_type": snapshot.get("surgery_type"),
        "risk_level": ca.get("risk_level"),
        "gpt_valid": bool(gpt.get("valid")),
        "gpt_strict_valid": bool(gpt.get("strict_valid")),
        "gpt_miller_valid": bool(gpt.get("miller_valid")),
        "gpt_error": gpt.get("error"),
        "gpt_miller_output": _one_line(gpt_miller, 500),
        "local_valid": bool(local.get("valid")),
        "local_strict_valid": bool(local.get("strict_valid")),
        "local_miller_valid": bool(local.get("miller_valid")),
        "local_error": local.get("error"),
        "local_miller_output": _one_line(local_miller, 500),
        "both_valid": bool(cmp_.get("both_valid")),
        "same_miller_output": bool(cmp_.get("same_miller_output")),
        "gpt_miller_len": cmp_.get("gpt_miller_len"),
        "local_miller_len": cmp_.get("local_miller_len"),
    }


def _build_markdown(rows: List[Dict[str, Any]], source_name: str) -> str:
    total = len(rows)
    gpt_valid = sum(1 for row in rows if row["gpt_valid"])
    local_valid = sum(1 for row in rows if row["local_valid"])
    both_valid = sum(1 for row in rows if row["both_valid"])
    same = sum(1 for row in rows if row["same_miller_output"])

    lines: List[str] = []
    lines.append(f"# Miller Compare Tidy Report")
    lines.append("")
    lines.append(f"- source: `{source_name}`")
    lines.append(f"- total: `{total}`")
    lines.append(f"- gpt_valid: `{gpt_valid}`")
    lines.append(f"- local_valid: `{local_valid}`")
    lines.append(f"- both_valid: `{both_valid}`")
    lines.append(f"- same_miller_output: `{same}`")
    lines.append("")
    lines.append("| index | caseid | time_sec | gpt_valid | local_valid | same_miller | gpt_error | local_error |")
    lines.append("|---:|---:|---:|:---:|:---:|:---:|---|---|")
    for row in rows:
        lines.append(
            "| {index} | {caseid} | {time_sec} | {gpt_valid} | {local_valid} | {same_miller_output} | {gpt_error} | {local_error} |".format(
                **{
                    "index": row.get("index", ""),
                    "caseid": row.get("caseid", ""),
                    "time_sec": row.get("time_sec", ""),
                    "gpt_valid": "Y" if row.get("gpt_valid") else "N",
                    "local_valid": "Y" if row.get("local_valid") else "N",
                    "same_miller_output": "Y" if row.get("same_miller_output") else "N",
                    "gpt_error": _one_line(row.get("gpt_error"), 90),
                    "local_error": _one_line(row.get("local_error"), 90),
                }
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tidy Miller compare JSONL into compact JSONL/CSV/Markdown.")
    parser.add_argument("--input", required=True, help="Path to demo_10_miller_compare.jsonl")
    parser.add_argument("--output-prefix", default="", help="Prefix for output files. Defaults to input stem in same folder.")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _load_jsonl(input_path)
    compact = [_compact_row(row) for row in rows]

    if args.output_prefix:
        base = Path(args.output_prefix)
    else:
        base = input_path.with_suffix("")

    compact_jsonl_path = base.parent / f"{base.name}.compact.jsonl"
    compact_csv_path = base.parent / f"{base.name}.compact.csv"
    report_md_path = base.parent / f"{base.name}.tidy_report.md"

    compact_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with compact_jsonl_path.open("w", encoding="utf-8") as f:
        for row in compact:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = list(compact[0].keys()) if compact else []
    with compact_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in compact:
            writer.writerow(row)

    report_md_path.write_text(_build_markdown(compact, input_path.name), encoding="utf-8")

    print(f"input:  {input_path}")
    print(f"rows:   {len(rows)}")
    print(f"jsonl:  {compact_jsonl_path}")
    print(f"csv:    {compact_csv_path}")
    print(f"report: {report_md_path}")


if __name__ == "__main__":
    main()
