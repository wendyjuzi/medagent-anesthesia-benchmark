import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from anes_pipeline import create_openai_client, generate_single_qa


def _load_records(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    raw = input_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if input_path.suffix.lower() == ".json":
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError("JSON input must be a list of records.")
        return [x for x in obj if isinstance(x, dict)]

    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _snapshot_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = record.get("snapshot") if isinstance(record.get("snapshot"), dict) else record
    if not isinstance(snapshot, dict):
        raise ValueError("Record does not contain a valid snapshot object.")
    return snapshot


def _build_cfg(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        llm_base_url=args.gpt_base_url,
        llm_api_key=args.gpt_api_key,
        api_key_env=args.gpt_api_key_env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate anesthesia QA directly with GPT API, without Miller embedding retrieval."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--output-field", default="llm_output_gpt_direct")

    parser.add_argument("--gpt-base-url", default="", help="Empty means official OpenAI API.")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--gpt-api-key", default="")
    parser.add_argument("--gpt-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    records = _load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No input records loaded.")

    client = create_openai_client(_build_cfg(args))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            qa = generate_single_qa(client, args.gpt_model, snapshot, retrieval=None)
            out = dict(record)
            out["generation_mode"] = "gpt_api_direct"
            out[args.output_field] = qa
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  - GPT direct generated {idx}/{len(records)}")

    print(f"Done: wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
