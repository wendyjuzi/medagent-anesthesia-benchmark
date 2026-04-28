import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import requests

from anes_pipeline import (
    SYSTEM_PROMPT,
    _decision_section,
    _decision_section_vitaldb,
    _extract_qa_block,
    _is_action_aligned,
    _is_unit_consistent_across_decisions,
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
        miller_chunk_overlap_chars=max(
            0,
            min(int(args.miller_chunk_overlap_chars), max(299, int(args.miller_chunk_chars) - 1)),
        ),
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
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    key = api_key.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post_chat(
    url: str,
    headers: Dict[str, str],
    model: str,
    user_prompt: str,
    max_tokens: int,
) -> Tuple[str, str]:
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
    choice = obj["choices"][0]
    content = str(choice["message"]["content"]).strip()
    finish_reason = str(choice.get("finish_reason", "")).strip().lower()
    return content, finish_reason


def _acceptance_flags(qa_text: Optional[str], snapshot: Dict[str, Any]) -> Dict[str, bool]:
    if not isinstance(qa_text, str) or not qa_text.strip():
        return {"aligned": False, "unit": False}
    return {
        "aligned": _is_action_aligned(qa_text, snapshot),
        "unit": _is_unit_consistent_across_decisions(qa_text, snapshot),
    }


def _has_miller_locator(text: Optional[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    miller_line = lines[2]
    if re.search(r"\[M10(?:\s*#\d+)?[^\]]*\]", miller_line, re.IGNORECASE):
        return True
    # Also treat locator without M10 index as valid, e.g. [M10 | 术中相关章节: ... | p.1974]
    has_chapter = bool(re.search(r"(术中相关章节|相关章节|chapter)\s*:", miller_line, re.IGNORECASE))
    has_page = bool(re.search(r"\bp\.\s*\d+\b", miller_line, re.IGNORECASE))
    return has_chapter and has_page


def _best_locator_from_retrieval(retrieval: Optional[Dict[str, Any]]) -> str:
    if not isinstance(retrieval, dict):
        return ""
    for key in ("results", "bm25_results", "dense_results"):
        vals = retrieval.get(key)
        if not isinstance(vals, list):
            continue
        for item in vals:
            if not isinstance(item, dict):
                continue
            loc = str(item.get("display_locator", "")).strip()
            if loc:
                return loc
    return ""


def _force_append_miller_locator(text: str, locator: str) -> str:
    if not text or not locator:
        return text
    out = _extract_qa_block(text)
    lines = [ln for ln in out.splitlines()]
    if len(lines) < 4:
        return out
    if _has_miller_locator("\n".join(lines[:4])):
        return out
    sep = "" if lines[2].rstrip().endswith(("\u3002", "\uff1b", ";")) else " "
    lines[2] = f"{lines[2].rstrip()}{sep}{locator}"
    return "\n".join(lines).strip()


def _strip_m10_index_token(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return text
    out = text
    # [M10#3 | ...] -> [M10 | ...]
    out = re.sub(r"\[\s*M10#\d+\s*\|\s*", "[M10 | ", out, flags=re.IGNORECASE)
    # Fallback: M10#3 | ... -> M10 | ...
    out = re.sub(r"\bM10#\d+\s*\|\s*", "M10 | ", out, flags=re.IGNORECASE)
    # [M10#3] -> [M10]
    out = re.sub(r"\[\s*M10#\d+\s*\]", "[M10]", out, flags=re.IGNORECASE)
    return out


def _strip_vitaldb_meta_phrases(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return text
    out = text
    m = re.search(r"(【决策干预（VitalDB）】[:：]\s*)(.*)$", out, re.IGNORECASE | re.DOTALL)
    if not m:
        return out

    prefix = m.group(1)
    body = m.group(2).strip()

    # Remove prompt/meta leakage phrases while keeping clinical action text.
    body = re.sub(
        r"[（(][^）)]*(?:logged_action|actual_intervention|golden|记录|对照)[^）)]*[）)]",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"按\s*logged_action\s*同类", "", body, flags=re.IGNORECASE)
    body = re.sub(r"按\s*actual_intervention\s*同类", "", body, flags=re.IGNORECASE)
    body = re.sub(r"严格\s*对照[^，,。；;]*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"(与|和)?(?:原始)?记录(?:动作|方向|剂量|数值|单位)?一致", "", body, flags=re.IGNORECASE)
    body = re.sub(r"(?:logged_action|actual_intervention|golden)", "", body, flags=re.IGNORECASE)
    body = re.sub(r"(并|且)\s*$", "", body)
    body = re.sub(r"\s{2,}", " ", body).strip(" ，,；;。")
    if not body:
        body = "维持当前给药策略。"

    return f"{out[:m.start(1)]}{prefix}{body}"


def _normalize_to_four_lines(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    out = _extract_qa_block(text).replace("\r\n", "\n").strip()
    # Rules-only normalization: re-split into exact 4 labeled lines without rewriting content.
    m = re.search(
        r"Q\s*[:：]\s*(.*?)\s*A\s*[:：]\s*【临床推理】\s*[:：]\s*(.*?)\s*【决策干预（Miller）】\s*[:：]\s*(.*?)\s*【决策干预（VitalDB）】\s*[:：]\s*(.*)$",
        out,
        re.DOTALL,
    )
    if not m:
        return _strip_vitaldb_meta_phrases(out)
    q = m.group(1).strip()
    reason = m.group(2).strip()
    miller = m.group(3).strip()
    vital = m.group(4).strip()
    normalized = (
        f"Q: {q}\n"
        f"A: 【临床推理】：{reason}\n"
        f"【决策干预（Miller）】：{miller}\n"
        f"【决策干预（VitalDB）】：{vital}"
    ).strip()
    return _strip_vitaldb_meta_phrases(normalized)


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
        raw, finish_reason = _post_chat(api_url, headers, model, prompt, max_tokens)
        if finish_reason == "length":
            retry_tokens = min(int(max_tokens * 1.8), 4000)
            raw_retry, finish_reason_retry = _post_chat(api_url, headers, model, prompt, retry_tokens)
            if len(raw_retry or "") >= len(raw or ""):
                raw = raw_retry
                finish_reason = finish_reason_retry
    except Exception as e:  # noqa: BLE001
        return {
            "error": str(e),
            "raw_output": None,
            "raw_finish_reason": "",
            "final_output": None,
            "valid": False,
            "miller_output": "",
            "vitaldb_output": "",
            "acceptance_flags_raw": {},
            "acceptance_flags_final": {},
        }

    cleaned = _extract_qa_block(raw)
    final = _normalize_to_four_lines(cleaned)
    if not isinstance(final, str) or not final.strip():
        final = cleaned if isinstance(cleaned, str) and cleaned.strip() else None

    # Rules-only locator patch: never ask model to rewrite evidence text.
    if final and not _has_miller_locator(final):
        locator = _best_locator_from_retrieval(retrieval)
        if locator:
            final = _force_append_miller_locator(final, locator)

    # User preference: keep locator content, but remove M10# index token in final output.
    final = _strip_m10_index_token(final)
    # Always sanitize VitalDB line from prompt/meta leakage phrases.
    final = _strip_vitaldb_meta_phrases(final)

    first_flags = _acceptance_flags(cleaned, snapshot)
    final_flags = _acceptance_flags(final, snapshot) if final else {}

    return {
        "error": None,
        "raw_output": raw,
        "raw_finish_reason": finish_reason,
        "final_output": final,
        "valid": bool(final),
        "acceptance_flags_raw": first_flags,
        "acceptance_flags_final": final_flags,
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
    parser.add_argument("--max-tokens", type=int, default=1200)

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
                "raw_finish_reason": result.get("raw_finish_reason", ""),
                "acceptance_flags_raw": result.get("acceptance_flags_raw", {}),
                "acceptance_flags_final": result.get("acceptance_flags_final", {}),
                "miller_output": result.get("miller_output", ""),
                "vitaldb_output": result.get("vitaldb_output", ""),
                "postprocess_mode": "rule_only",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  - GPT requests generated {idx}/{len(records)}")
            if args.print_miller:
                print("    Miller:", result.get("miller_output", ""))

    print(f"Done: wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
