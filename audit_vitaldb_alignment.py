import argparse
import json
import os
from typing import Any, Dict, List

from anes_pipeline import build_vitaldb_accuracy_report


def _load_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    if path.lower().endswith(".json"):
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        raise ValueError("JSON input must be a list of record objects")

    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VitalDB actions against Miller-style policy")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON file")
    parser.add_argument("--output", default="", help="Output report JSON path")
    args = parser.parse_args()

    records = _load_records(args.input)
    report = build_vitaldb_accuracy_report(records)

    if args.output:
        out_path = args.output
    else:
        root, _ = os.path.splitext(args.input)
        out_path = root + ".vitaldb_miller_alignment_report.json"

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== VitalDB vs Miller Report ===")
    print(f"input:   {args.input}")
    print(f"output:  {out_path}")
    print(f"total:   {report.get('total_evaluated', 0)}")
    print(f"aligned: {report.get('aligned', 0)}")
    print(f"partial: {report.get('partially_aligned', 0)}")
    print(f"misalgn: {report.get('misaligned', report.get('potentially_inaccurate', 0))}")
    print(f"uncert.: {report.get('uncertain', 0)}")
    ratio = float(report.get("misaligned_ratio", report.get("potentially_inaccurate_ratio", 0.0)))
    print(f"misaligned_ratio: {ratio:.2%}")
    print(f"high_risk_conflicts: {report.get('high_risk_conflicts', 0)}")


if __name__ == "__main__":
    main()
