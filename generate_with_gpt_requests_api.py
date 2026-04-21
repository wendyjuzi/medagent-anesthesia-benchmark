import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests

from anes_pipeline import (
    SYSTEM_PROMPT,
    _decision_section,
    _decision_section_vitaldb,
    _extract_qa_block,
    _golden_action_hint,
    _is_action_aligned,
    _is_strict_qa,
    build_miller_retriever,
    build_user_prompt,
    create_embedding_client,
    retrieve_miller_context,
)


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


def _build_retrieval_cfg(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        enable_miller_rag=bool(args.enable_miller_rag),
        miller_corpus_path=args.miller_corpus_path,
        miller_index_path=args.miller_index_path,
        miller_top_k=max(1, min(5, int(args.miller_top_k))),
        miller_chunk_chars=max(300, int(args.miller_chunk_chars)),
        miller_chunk_overlap_chars=max(0, min(int(args.miller_chunk_overlap_chars), max(299, int(args.miller_chunk_chars) - 1))),
        miller_max_passage_chars=max(200, int(args.miller_max_passage_chars)),
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key_env=args.embedding_api_key_env,
        embedding_api_key=args.embedding_api_key,
        llm_base_url="",
        llm_api_key="",
        api_key_env="OPENAI_API_KEY",
    )


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    key = api_key.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post_chat(url: str, headers: Dict[str, str], model: str, user_prompt: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    obj = resp.json()
    return str(obj["choices"][0]["message"]["content"]).strip()


def _repair_via_requests(
    url: str,
    headers: Dict[str, str],
    model: str,
    raw_text: str,
    snapshot: Dict[str, Any],
    max_tokens: int,
) -> Optional[str]:
    hint = _golden_action_hint(snapshot)
    med_key = hint.get("medication_key", "")
    actual = hint.get("actual_intervention", "")
    kws = hint.get("keywords", [])
    kw_text = ", ".join(kws) if kws else "N/A"
    repair_sys = (
        "You are a medical text formatter. "
        "Return only final QA in Chinese. "
        "No thinking process, no bullets, no instruction echo."
    )
    repair_user = (
        "Rewrite to strict format. Output only:\n"
        "Q: ...\n"
        "A: 【临床推理】：...\n"
        "【决策干预（Miller）】：...\n"
        "【决策干预（VitalDB）】：...\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        "【决策干预（VitalDB）】必须与golden logged_action同药物类别，不得矛盾。\n"
        "Source text:\n"
        f"{raw_text}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": repair_sys},
            {"role": "user", "content": repair_user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        obj = resp.json()
        return str(obj["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None


def _generate_one(
    api_url: str,
    headers: Dict[str, str],
    model: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = build_user_prompt(snapshot, retrieval=retrieval)
    try:
        raw = _post_chat(api_url, headers, model, prompt, max_tokens)
    except Exception as e:  # noqa: BLE001
        return {
            "error": str(e),
            "raw_output": None,
            "final_output": None,
            "valid": False,
            "miller_output": "",
            "vitaldb_output": "",
        }

    cleaned = _extract_qa_block(raw)
    final = None
    if _is_strict_qa(cleaned) and _is_action_aligned(cleaned, snapshot):
        final = cleaned
    else:
        repaired = _repair_via_requests(api_url, headers, model, raw, snapshot, max_tokens)
        if repaired:
            repaired_cleaned = _extract_qa_block(repaired)
            if _is_strict_qa(repaired_cleaned) and _is_action_aligned(repaired_cleaned, snapshot):
                final = repaired_cleaned

    return {
        "error": None,
        "raw_output": raw,
        "final_output": final,
        "valid": bool(final),
        "miller_output": _decision_section(final) if final else "",
        "vitaldb_output": _decision_section_vitaldb(final) if final else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate anesthesia QA via requests-based GPT API, optionally with Miller retrieval."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--output-field", default="llm_output_gpt_requests")
    parser.add_argument("--print-miller", action="store_true", help="Print Miller section for each generated record.")

    parser.add_argument("--api-url", default="https://api2.aigcbest.top/v1/chat/completions")
    parser.add_argument("--api-key", default="", help="Bearer token for the gateway.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=700)

    parser.add_argument("--use-existing-retrieval", action="store_true")
    parser.add_argument("--enable-miller-rag", action="store_true")
    parser.add_argument("--miller-corpus-path", default="")
    parser.add_argument("--miller-index-path", default="")
    parser.add_argument("--miller-top-k", type=int, default=3)
    parser.add_argument("--miller-chunk-chars", type=int, default=1200)
    parser.add_argument("--miller-chunk-overlap-chars", type=int, default=200)
    parser.add_argument("--miller-max-passage-chars", type=int, default=800)
    parser.add_argument("--embedding-backend", default="auto", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-api-key", default="")
    args = parser.parse_args()

    records = _load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No input records loaded.")

    retriever = None
    embed_client = None
    retrieval_cfg = _build_retrieval_cfg(args)
    if args.enable_miller_rag and not args.use_existing_retrieval:
        embed_client = create_embedding_client(retrieval_cfg)
        retriever = build_miller_retriever(embed_client, retrieval_cfg)

    headers = _build_headers(args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            retrieval = None
            if args.use_existing_retrieval and isinstance(record.get("miller_retrieval"), dict):
                retrieval = record["miller_retrieval"]
            elif args.enable_miller_rag and retriever is not None and embed_client is not None:
                retrieval = retrieve_miller_context(snapshot, retriever, embed_client, retrieval_cfg)

            result = _generate_one(args.api_url, headers, args.model, snapshot, retrieval, args.max_tokens)
            out = dict(record)
            out["generation_mode"] = "gpt_requests_api"
            if retrieval is not None:
                out["miller_retrieval"] = retrieval
            out[args.output_field] = result.get("final_output")
            out[f"{args.output_field}_raw"] = result.get("raw_output")
            out[f"{args.output_field}_meta"] = {
                "valid": result.get("valid", False),
                "error": result.get("error"),
                "miller_output": result.get("miller_output", ""),
                "vitaldb_output": result.get("vitaldb_output", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  - GPT requests generated {idx}/{len(records)}")
            if args.print_miller:
                print("    Miller:", result.get("miller_output", ""))

    print(f"Done: wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
