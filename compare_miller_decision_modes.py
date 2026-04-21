import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests

from anes_pipeline import (
    SYSTEM_PROMPT,
    _decision_section,
    _extract_qa_block,
    _golden_action_hint,
    _is_action_aligned,
    _is_strict_qa,
    build_miller_retriever,
    build_user_prompt,
    create_embedding_client,
    create_openai_client,
    generate_single_qa,
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
        enable_miller_rag=True,
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
        llm_base_url=args.local_base_url,
        llm_api_key=args.local_api_key,
        api_key_env=args.local_api_key_env,
    )


def _build_local_llm_cfg(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        llm_base_url=args.local_base_url,
        llm_api_key=args.local_api_key,
        api_key_env=args.local_api_key_env,
    )


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    key = api_key.strip()
    if key:
        headers["Authorization"] = key if key.lower().startswith("bearer ") else f"Bearer {key}"
    return headers


def _resolve_api_key(explicit_key: str, env_name: str) -> str:
    key = (explicit_key or "").strip()
    if key:
        return key
    if env_name:
        return os.getenv(env_name, "").strip()
    return ""


def _post_gpt_requests(
    api_url: str,
    headers: Dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    obj = resp.json()
    return str(obj["choices"][0]["message"]["content"]).strip()


def _extract_miller_only(final_output: Optional[str]) -> str:
    if not final_output:
        return ""
    return _decision_section(final_output).strip()


def _repair_via_requests(
    api_url: str,
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
        resp = requests.post(api_url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        obj = resp.json()
        return str(obj["choices"][0]["message"]["content"]).strip()
    except Exception:  # noqa: BLE001
        return None


def _evaluate_validity(
    final_output: Optional[str],
    snapshot: Dict[str, Any],
    validity_mode: str,
) -> Dict[str, Any]:
    text = str(final_output or "").strip()
    strict_valid = bool(text) and _is_strict_qa(text) and _is_action_aligned(text, snapshot)
    miller_output = _extract_miller_only(text)
    miller_valid = bool(miller_output)
    if validity_mode == "strict":
        valid = strict_valid
    else:
        valid = miller_valid
    return {
        "valid": valid,
        "strict_valid": strict_valid,
        "miller_valid": miller_valid,
        "miller_output": miller_output,
    }


def _run_gpt_requests_mode(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    prompt = build_user_prompt(snapshot, retrieval=retrieval)
    try:
        raw = _post_gpt_requests(args.gpt_api_url, headers, args.gpt_model, prompt, args.max_tokens)
    except Exception as e:  # noqa: BLE001
        return {
            "error": str(e),
            "raw_output": None,
            "final_output": None,
            "valid": False,
            "strict_valid": False,
            "miller_valid": False,
            "miller_output": "",
        }

    cleaned = _extract_qa_block(raw)
    final = cleaned if (_is_strict_qa(cleaned) and _is_action_aligned(cleaned, snapshot)) else None
    if not final:
        repaired = _repair_via_requests(args.gpt_api_url, headers, args.gpt_model, raw, snapshot, args.max_tokens)
        if repaired:
            repaired_cleaned = _extract_qa_block(repaired)
            if repaired_cleaned:
                final = repaired_cleaned
    validity = _evaluate_validity(final, snapshot, args.validity_mode)
    return {
        "error": None,
        "raw_output": raw,
        "final_output": final,
        "valid": bool(validity["valid"]),
        "strict_valid": bool(validity["strict_valid"]),
        "miller_valid": bool(validity["miller_valid"]),
        "miller_output": str(validity["miller_output"]),
    }


def _run_local_mode(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    local_client: Any,
) -> Dict[str, Any]:
    final = generate_single_qa(local_client, args.local_model, snapshot, retrieval=retrieval)
    validity = _evaluate_validity(final, snapshot, args.validity_mode)
    return {
        "error": None if final else "local_generation_invalid_or_empty",
        "raw_output": final,
        "final_output": final,
        "valid": bool(validity["valid"]),
        "strict_valid": bool(validity["strict_valid"]),
        "miller_valid": bool(validity["miller_valid"]),
        "miller_output": str(validity["miller_output"]),
    }


def _compare_miller_outputs(gpt_out: Dict[str, Any], local_out: Dict[str, Any]) -> Dict[str, Any]:
    gpt_miller = str(gpt_out.get("miller_output") or "").strip()
    local_miller = str(local_out.get("miller_output") or "").strip()
    return {
        "both_valid": bool(gpt_out.get("valid")) and bool(local_out.get("valid")),
        "same_miller_output": bool(gpt_miller) and (gpt_miller == local_miller),
        "gpt_miller_len": len(gpt_miller),
        "local_miller_len": len(local_miller),
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    gpt_valid = sum(1 for row in rows if row["gpt_api_miller10"].get("valid"))
    local_valid = sum(1 for row in rows if row["local_embedding_miller10"].get("valid"))
    gpt_strict_valid = sum(1 for row in rows if row["gpt_api_miller10"].get("strict_valid"))
    local_strict_valid = sum(1 for row in rows if row["local_embedding_miller10"].get("strict_valid"))
    both_valid = sum(1 for row in rows if row["comparison"].get("both_valid"))
    same_miller = sum(1 for row in rows if row["comparison"].get("same_miller_output"))
    return {
        "total": total,
        "gpt_api_miller10_valid": gpt_valid,
        "local_embedding_miller10_valid": local_valid,
        "gpt_api_miller10_strict_valid": gpt_strict_valid,
        "local_embedding_miller10_strict_valid": local_strict_valid,
        "both_valid": both_valid,
        "same_miller_output": same_miller,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Miller decision outputs between GPT API and local embedding/local model modes."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output-jsonl", required=True, help="Output comparison JSONL path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--use-existing-retrieval", action="store_true")
    parser.add_argument(
        "--validity-mode",
        default="miller_only",
        choices=["miller_only", "strict"],
        help="miller_only: valid if Miller section exists; strict: enforce strict QA + VitalDB alignment.",
    )

    parser.add_argument("--gpt-api-url", default="https://api2.aigcbest.top/v1/chat/completions")
    parser.add_argument("--gpt-api-key", default="")
    parser.add_argument("--gpt-api-key-env", default="GPT_API_KEY")
    parser.add_argument("--gpt-model", required=True)

    parser.add_argument("--local-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--local-model", required=True)
    parser.add_argument("--local-api-key", default="local")
    parser.add_argument("--local-api-key-env", default="OPENAI_API_KEY")

    parser.add_argument("--miller-corpus-path", required=True)
    parser.add_argument("--miller-index-path", required=True)
    parser.add_argument("--miller-top-k", type=int, default=3)
    parser.add_argument("--miller-chunk-chars", type=int, default=1200)
    parser.add_argument("--miller-chunk-overlap-chars", type=int, default=200)
    parser.add_argument("--miller-max-passage-chars", type=int, default=800)
    parser.add_argument("--embedding-backend", default="local", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", required=True)
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

    retrieval_cfg = _build_retrieval_cfg(args)
    retriever = None
    embed_client = None
    if not args.use_existing_retrieval:
        embed_client = create_embedding_client(retrieval_cfg)
        retriever = build_miller_retriever(embed_client, retrieval_cfg)

    local_client = create_openai_client(_build_local_llm_cfg(args))
    gpt_api_key = _resolve_api_key(args.gpt_api_key, args.gpt_api_key_env)
    if not gpt_api_key:
        raise ValueError(
            "Missing GPT API key. Pass --gpt-api-key or set environment variable from --gpt-api-key-env."
        )
    gpt_headers = _build_headers(gpt_api_key)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            retrieval = None
            if args.use_existing_retrieval and isinstance(record.get("miller_retrieval"), dict):
                retrieval = record["miller_retrieval"]
            elif retriever is not None and embed_client is not None:
                retrieval = retrieve_miller_context(snapshot, retriever, embed_client, retrieval_cfg)

            gpt_out = _run_gpt_requests_mode(snapshot, retrieval, args, gpt_headers)
            local_out = _run_local_mode(snapshot, retrieval, args, local_client)
            comparison = _compare_miller_outputs(gpt_out, local_out)

            row = {
                "index": idx,
                "caseid": record.get("caseid"),
                "snapshot": snapshot,
                "miller_retrieval": retrieval,
                "gpt_api_miller10": gpt_out,
                "local_embedding_miller10": local_out,
                "comparison": comparison,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  - compared Miller outputs {idx}/{len(records)}")

    summary = _summarize(rows)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Miller Decision Comparison Summary ===")
    print(f"jsonl:   {output_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
