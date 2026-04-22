import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from anes_pipeline import (
    SYSTEM_PROMPT,
    _extract_qa_block,
    _is_action_aligned,
    _is_strict_qa,
    _repair_qa_output,
    build_user_prompt,
    build_miller_retriever,
    create_embedding_client,
    retrieve_miller_context,
)

LABEL_REASONING = "【临床推理】"
LABEL_MILLER = "【决策干预（Miller）】"
LABEL_VITALDB = "【决策干预（VitalDB）】"


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


def _build_client(base_url: str, api_key: str, api_key_env: str) -> OpenAI:
    key = api_key.strip() or os.getenv(api_key_env, "").strip()
    if base_url.strip():
        if not key:
            key = "local"
        return OpenAI(api_key=key, base_url=base_url.strip().rstrip("/"))
    if not key:
        raise EnvironmentError(f"Missing API key for model client in env {api_key_env}")
    return OpenAI(api_key=key)


def _resolve_api_key(explicit_key: str, env_name: str) -> str:
    key = (explicit_key or "").strip()
    if key:
        return key
    if env_name:
        return os.getenv(env_name, "").strip()
    return ""


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _normalize_chat_completions_url(base_or_endpoint: str) -> str:
    value = str(base_or_endpoint or "").strip().rstrip("/")
    if not value:
        return ""
    lower = value.lower()
    if lower.endswith("/chat/completions"):
        return value
    if lower.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _has_miller_locator(text: str) -> bool:
    src = str(text or "")
    return bool(re.search(r"(?i)m10\s*#\d+", src)) or ("章节:" in src and "段落:" in src)


def _infer_chapter_from_text(text: str) -> Tuple[str, str]:
    src = str(text or "").strip()
    if not src:
        return "", ""
    head = " ".join(src.split())[:220]
    patterns = [
        r"^\s*(\d{1,3})\s*[•·\-\–]\s*([A-Za-z][A-Za-z0-9 ,:&()/\-]{3,120})",
        r"(?i)\bchapter\s+(\d{1,3})\s*[:\-–]?\s*([A-Za-z][A-Za-z0-9 ,:&()/\-]{3,120})",
    ]
    for pat in patterns:
        m = re.search(pat, head)
        if not m:
            continue
        chap = str(m.group(1) or "").strip()
        title = str(m.group(2) or "").strip(" .;,-")
        if chap:
            return chap, title
    return "", ""


def _build_miller_locator_from_item(item: Dict[str, Any], rank_fallback: int = 1) -> str:
    rank = item.get("rank", rank_fallback)
    chapter = str(item.get("chapter") or "").strip()
    section = str(item.get("section") or "").strip()
    if not chapter:
        inferred_chapter, inferred_section = _infer_chapter_from_text(item.get("text"))
        chapter = inferred_chapter or chapter
        if not section:
            section = inferred_section
    chapter = chapter or "未知"
    paragraph = str(item.get("paragraph") or item.get("page_chunk_index") or item.get("chunk_id") or "").strip() or "未知"
    if section:
        return f"[M10#{rank}|章节:{chapter}; 小节:{section}; 段落:{paragraph}]"
    return f"[M10#{rank}|章节:{chapter}; 段落:{paragraph}]"


def _best_miller_locator(retrieval: Optional[Dict[str, Any]]) -> str:
    if not isinstance(retrieval, dict):
        return ""
    results = retrieval.get("results", [])
    if not isinstance(results, list):
        return ""
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            return _build_miller_locator_from_item(item, rank_fallback=idx)
    return ""


def _inject_miller_locator_if_missing(text: str, retrieval: Optional[Dict[str, Any]]) -> str:
    src = str(text or "").strip()
    if not src or _has_miller_locator(src):
        return src
    locator = _best_miller_locator(retrieval)
    if not locator:
        return src
    return f"{src} {locator}".strip()


def _looks_like_instruction_conflict_text(text: str) -> bool:
    src = str(text or "").lower()
    if not src:
        return False
    markers = [
        "return strict json only",
        "do not output any text outside",
        "output exactly one qa pair",
        "this means i need to",
        "let me check instructions",
        "指令冲突",
        "让我仔细检查指令",
        "这意味着我需要",
    ]
    return any(m in src for m in markers)


def _json_payload_is_meta_or_low_quality(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return True
    keys = {"question", "reasoning", "miller_decision", "vitaldb_decision", "final_output"}
    if not keys.intersection(set(payload.keys())):
        return True
    merged = " ".join(str(payload.get(k) or "") for k in keys)
    if _looks_like_instruction_conflict_text(merged):
        return True
    if len(str(payload.get("miller_decision") or "").strip()) < 2:
        return True
    return False


def _post_chat(
    api_url: str,
    headers: Dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    obj = resp.json()
    return str(obj["choices"][0]["message"]["content"]).strip()


def _build_web_search_query(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    suffix: str,
) -> str:
    base = ""
    if isinstance(retrieval, dict):
        results = retrieval.get("results", [])
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    if text:
                        base = " ".join(text.split())[:220]
                        break
    if not base:
        pb = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
        assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
        risk_flags = assess.get("risk_flags", []) if isinstance(assess.get("risk_flags"), list) else []
        base = "; ".join(
            [
                str(pb.get("department") or "").strip(),
                str(pb.get("surgery_group") or "").strip(),
                str(snapshot.get("surgery_type") or "").strip(),
                ", ".join(str(x) for x in risk_flags[:2]),
            ]
        ).strip("; ")
    query = f"{base}; {suffix}".strip("; ")
    return " ".join(query.split())


def _miller10_hit_score(item: Dict[str, Any]) -> int:
    title = str(item.get("title") or "").lower()
    url = str(item.get("url") or "").lower()
    snippet = str(item.get("snippet") or "").lower()
    text = " ".join([title, url, snippet])

    score = 0
    if "miller" in text:
        score += 2
    if "anesthesia" in text or "anaesthesia" in text or "麻醉" in text:
        score += 1
    if "10th" in text or "tenth" in text or "第十版" in text:
        score += 2
    if "edition" in text or "版" in text:
        score += 1
    return score


def _filter_miller10_results(results: List[Dict[str, Any]], min_score: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        hit_score = _miller10_hit_score(item)
        if hit_score >= int(min_score):
            copied = dict(item)
            copied["miller10_hit_score"] = hit_score
            out.append(copied)
    return out


def _search_with_gpt_search_api(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    api_url: str,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    query = _build_web_search_query(snapshot, retrieval, args.gpt_search_query_suffix)
    top_k = max(1, int(args.gpt_search_top_k))
    user_prompt = (
        "Search online evidence for this anesthesia scenario.\n"
        f"Query: {query}\n\n"
        f"Return top {top_k} results as strict JSON object only:\n"
        '{"query":"...","results":[{"title":"...","url":"...","snippet":"...","score":0.0}]}\n'
        "Rules:\n"
        "- Prioritize Miller's Anesthesia 10th edition evidence.\n"
        "- Keep snippet concise and clinically relevant.\n"
        "- No markdown fences, no extra text outside JSON.\n"
    )
    search_system = (
        "You are a medical literature search assistant. "
        "Return only valid JSON results for evidence retrieval."
    )
    try:
        raw = _post_chat(
            api_url=api_url,
            headers=headers,
            model=args.gpt_search_model,
            prompt=user_prompt,
            max_tokens=max(200, int(args.gpt_search_max_tokens)),
            system_prompt=search_system,
        )
        parsed = _extract_json_obj(raw)
        items = parsed.get("results", [])
        results: List[Dict[str, Any]] = []
        if isinstance(items, list):
            for idx, item in enumerate(items[:top_k], start=1):
                if not isinstance(item, dict):
                    continue
                results.append(
                    {
                        "rank": idx,
                        "title": str(item.get("title") or "").strip(),
                        "url": str(item.get("url") or "").strip(),
                        "snippet": str(item.get("snippet") or "").strip(),
                        "score": float(item.get("score") or 0.0),
                        "provider": "gpt_search_api",
                    }
                )
        if args.force_gpt_search_miller10:
            filtered = _filter_miller10_results(results, int(args.gpt_search_miller10_min_score))
            if filtered:
                results = filtered
        return {
            "enabled": True,
            "provider": "gpt_search_api",
            "query": str(parsed.get("query") or query),
            "results": results,
            "error": None,
            "raw_response": raw,
            "force_miller10": bool(args.force_gpt_search_miller10),
            "miller10_hit_count": len(results),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "enabled": True,
            "provider": "gpt_search_api",
            "query": query,
            "results": [],
            "error": str(e),
            "force_miller10": bool(args.force_gpt_search_miller10),
            "miller10_hit_count": 0,
        }


def _format_web_context_block(web_search: Optional[Dict[str, Any]]) -> str:
    if not isinstance(web_search, dict) or not web_search.get("enabled"):
        return ""
    results = web_search.get("results", [])
    if not isinstance(results, list) or not results:
        return ""
    lines = [
        "",
        "Supplemental web search snippets (non-authoritative unless corroborated):",
    ]
    for item in results:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank", "?")
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if len(snippet) > 480:
            snippet = snippet[:477] + "..."
        lines.append(f"[Web#{rank}] {title} | {url}")
        lines.append(snippet)
    return "\n".join(lines)


def _canonicalize_line_label(line: str) -> str:
    src = str(line or "").strip()
    if not src:
        return ""
    src = src.replace("：", ":")
    src = re.sub(r"\s+", " ", src).strip()
    lower = src.lower()

    def _payload(value: str) -> str:
        return re.sub(r"^(?:q|a)\s*:\s*", "", value, flags=re.IGNORECASE).strip()

    if re.match(r"^(q|question)\s*:", lower):
        return "Q: " + _payload(src)
    if re.match(r"^a\s*:", lower) and "临床推理" not in src and "clinical reasoning" not in lower:
        return "A: " + _payload(src)

    miller_alias = [
        "【决策干预（miller）】",
        "【决策干预(miller)】",
        "【决策干预 m i l l e r】",
        "【miller决策】",
        "【miller建议】",
        "【miller】",
        "[miller]",
        "miller decision",
    ]
    vital_alias = [
        "【决策干预（vitaldb）】",
        "【决策干预(vitaldb)】",
        "【决策干预 vitaldb】",
        "【vitaldb决策】",
        "【vitaldb】",
        "[vitaldb]",
        "vitaldb decision",
    ]
    reasoning_alias = [
        "【临床推理】",
        "[临床推理]",
        "[clinical reasoning]",
        "clinical reasoning",
    ]

    payload = re.sub(r"^[\[【].*?[\]】]\s*:?\s*", "", src, flags=re.IGNORECASE).strip()
    if any(a in lower for a in miller_alias):
        return f"{LABEL_MILLER}：{payload}"
    if any(a in lower for a in vital_alias):
        return f"{LABEL_VITALDB}：{payload}"
    if any(a in lower for a in reasoning_alias):
        return f"A: {LABEL_REASONING}：{payload}"
    return src


def normalize_labels(text: str) -> str:
    src = _extract_qa_block(str(text or ""))
    lines = [ln for ln in src.replace("\r\n", "\n").split("\n") if str(ln).strip()]
    normalized = [_canonicalize_line_label(ln) for ln in lines]
    normalized = [ln for ln in normalized if ln.strip()]
    if normalized:
        if not re.match(r"(?i)^Q\s*:", normalized[0]):
            normalized.insert(0, "Q: 请基于术中快照给出最合理的麻醉干预？")
        if not any(ln.startswith("A:") for ln in normalized):
            normalized.insert(1, f"A: {LABEL_REASONING}：请结合血流动力学与麻醉深度综合判断。")
    return "\n".join(normalized).strip()


def _extract_section_by_label(text: str, label: str) -> str:
    m = re.search(rf"{re.escape(label)}\s*[:：]?\s*(.*?)(?=\n(?:Q:|A:|【)|$)", text, re.IGNORECASE | re.DOTALL)
    return str(m.group(1)).strip() if m else ""


def parse_sections_relaxed(text: str) -> Dict[str, Any]:
    normalized = normalize_labels(text)
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    q_line = ""
    for ln in lines:
        if re.match(r"(?i)^Q\s*:", ln):
            q_line = ln
            break

    reasoning = _extract_section_by_label(normalized, LABEL_REASONING)
    miller = _extract_section_by_label(normalized, LABEL_MILLER)
    vital = _extract_section_by_label(normalized, LABEL_VITALDB)

    missing_miller_label = (LABEL_MILLER not in normalized)
    inferred_miller = False
    if not miller:
        keywords = [
            "根据miller",
            "miller",
            "建议干预",
            "优先处理",
            "应考虑",
            "增加镇痛",
            "加深麻醉",
            "继续观察",
            "维持灌注",
            "升压",
        ]
        for ln in lines:
            low = ln.lower()
            if "vitaldb" in low or "临床推理" in low:
                continue
            if any(k in low for k in keywords):
                miller = re.sub(r"^(?:A:|Q:)\s*", "", ln, flags=re.IGNORECASE).strip()
                inferred_miller = bool(miller)
                if inferred_miller:
                    break

    return {
        "normalized_text": normalized,
        "lines": lines,
        "q_line": q_line,
        "reasoning_section": reasoning,
        "miller_section": miller,
        "vitaldb_section": vital,
        "missing_miller_label": bool(missing_miller_label),
        "inferred_miller": bool(inferred_miller),
    }


def semantic_repair(text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed)
    miller = str(out.get("miller_section") or "").strip()
    if miller:
        return out
    reasoning = str(out.get("reasoning_section") or "").strip()
    lines = out.get("lines", []) if isinstance(out.get("lines"), list) else []
    candidates: List[str] = []
    if reasoning:
        candidates.append(reasoning)
    candidates.extend([str(x).strip() for x in lines if str(x).strip()])
    keywords = [
        "miller",
        "建议",
        "优先",
        "应考虑",
        "升压",
        "镇痛",
        "麻醉",
        "灌注",
    ]
    for cand in candidates:
        low = cand.lower()
        if "vitaldb" in low:
            continue
        if any(k in low for k in keywords):
            out["miller_section"] = re.sub(r"^(?:A:|Q:|【[^】]+】[:：]?)\s*", "", cand, flags=re.IGNORECASE).strip()
            out["inferred_miller"] = True
            break
    return out


def _render_from_parsed(parsed: Dict[str, Any]) -> str:
    q = str(parsed.get("q_line") or "").strip()
    if not q:
        q = "Q: 请基于术中快照给出最合理的麻醉干预？"
    elif not re.match(r"(?i)^Q\s*:", q):
        q = "Q: " + re.sub(r"^(?:Q\s*[:：])\s*", "", q, flags=re.IGNORECASE).strip()
    reasoning = str(parsed.get("reasoning_section") or "").strip() or "请结合血流动力学与麻醉深度综合判断。"
    miller = str(parsed.get("miller_section") or "").strip() or "证据定位不足，建议基于Miller相关章节保守处理。"
    vital = str(parsed.get("vitaldb_section") or "").strip() or "建议与记录动作同类药物小步调整并复评。"
    return "\n".join([q, f"A: {LABEL_REASONING}：{reasoning}", f"{LABEL_MILLER}：{miller}", f"{LABEL_VITALDB}：{vital}"])


def rewrite_if_needed(
    client: OpenAI,
    model: str,
    raw_text: str,
    snapshot: Dict[str, Any],
    max_tokens: int,
    parsed: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], bool, str]:
    current = dict(parsed)
    # Step 4: only rewrite when still missing key fields.
    if current.get("miller_section") and current.get("vitaldb_section") and current.get("reasoning_section"):
        rendered = _render_from_parsed(current)
        return rendered, current, False, ""

    # First, targeted repair for missing Miller label.
    repaired = _repair_missing_miller_label(client, model, raw_text, max_tokens)
    if repaired:
        parsed2 = semantic_repair(repaired, parse_sections_relaxed(repaired))
        if parsed2.get("miller_section"):
            return _render_from_parsed(parsed2), parsed2, True, "missing_miller_label_repair"

    # Final fallback: strict template repair once.
    repaired2 = _repair_qa_output(client, model, raw_text, snapshot)
    if repaired2:
        parsed3 = semantic_repair(repaired2, parse_sections_relaxed(repaired2))
        if parsed3.get("miller_section"):
            return _render_from_parsed(parsed3), parsed3, True, "general_repair"

    rendered = _render_from_parsed(current)
    return rendered, current, False, ""


def final_validate(parsed: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    rendered = _render_from_parsed(parsed)
    strict_qa = bool(_is_strict_qa(rendered))
    action_aligned = bool(_is_action_aligned(rendered, snapshot))
    semantic_valid = bool(str(parsed.get("miller_section") or "").strip())
    format_valid = bool(strict_qa)
    strict_valid = bool(strict_qa and action_aligned)
    failure_type = _classify_invalid_reason(
        error=None,
        semantic_valid=semantic_valid,
        format_valid=format_valid,
        action_aligned=action_aligned,
        missing_miller_label=bool(parsed.get("missing_miller_label")),
    )
    # Keep semantic layer as top-level validity.
    valid = bool(semantic_valid)
    return {
        "rendered_output": rendered,
        "strict_qa": strict_qa,
        "format_valid": format_valid,
        "semantic_valid": semantic_valid,
        "strict_valid": strict_valid,
        "action_aligned": action_aligned,
        "valid": valid,
        "failure_type": failure_type if not valid else "",
        "invalid_reason": failure_type if not valid else "",
    }


def _render_qa_from_json_payload(payload: Dict[str, Any], retrieval: Optional[Dict[str, Any]]) -> str:
    def _clean_value(value: Any) -> str:
        out = str(value or "").strip()
        out = re.sub(r"^\s*(Q|A)\s*[:：]\s*", "", out, flags=re.IGNORECASE).strip()
        out = re.sub(r"^\s*【[^】]+】\s*[:：]\s*", "", out, flags=re.IGNORECASE).strip()
        if _looks_like_instruction_conflict_text(out):
            return ""
        return out

    question = str(
        payload.get("question")
        or payload.get("q")
        or "请基于术中快照给出最合理的麻醉干预？"
    ).strip()
    reasoning = _clean_value(payload.get("reasoning") or payload.get("clinical_reasoning") or "")
    miller = _clean_value(payload.get("miller_decision") or payload.get("miller_output") or "")
    miller = _inject_miller_locator_if_missing(miller, retrieval)
    vital = _clean_value(payload.get("vitaldb_decision") or payload.get("vitaldb_output") or "")
    parsed = {
        "q_line": f"Q: {question}",
        "reasoning_section": reasoning,
        "miller_section": miller,
        "vitaldb_section": vital,
        "missing_miller_label": not bool(miller),
        "inferred_miller": False,
    }
    return _render_from_parsed(parsed)


def _build_structured_json_prompt(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    web_search: Optional[Dict[str, Any]],
) -> str:
    snap_text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    retrieval_lines: List[str] = []
    if isinstance(retrieval, dict):
        query = str(retrieval.get("query_rewritten") or retrieval.get("query") or "").strip()
        if query:
            retrieval_lines.append(f"retrieval_query: {query}")
        results = retrieval.get("results", [])
        if isinstance(results, list):
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                rank = item.get("rank", "?")
                chapter = str(item.get("chapter") or "").strip() or "未知"
                paragraph = str(item.get("paragraph") or item.get("chunk_id") or "未知").strip()
                text = str(item.get("text") or "").strip()
                if len(text) > 360:
                    text = text[:357] + "..."
                retrieval_lines.append(f"[M10#{rank}|章节:{chapter}; 段落:{paragraph}] {text}")
    web_lines: List[str] = []
    if isinstance(web_search, dict):
        ws = web_search.get("results", [])
        if isinstance(ws, list):
            for item in ws[:2]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                web_lines.append(f"[WEB] {title} | {snippet}")

    retrieval_block = "\n".join(retrieval_lines) if retrieval_lines else "none"
    web_block = "\n".join(web_lines) if web_lines else "none"
    return (
        "You are a strict clinical formatter.\n"
        "Task: read the snapshot and evidence, then output JSON only.\n"
        "Do not discuss instructions, conflicts, or policies.\n\n"
        "[Snapshot JSON]\n"
        f"{snap_text}\n\n"
        "[Miller retrieval evidence]\n"
        f"{retrieval_block}\n\n"
        "[Optional web snippets]\n"
        f"{web_block}\n\n"
        "Output MUST be a single JSON object with EXACT keys:\n"
        '{"question":"...","reasoning":"...","miller_decision":"...","vitaldb_decision":"...","final_output":"..."}\n'
        "Rules:\n"
        "- No markdown/code fence/explanations.\n"
        "- Keep Chinese clinical text concise.\n"
        "- miller_decision must include at least one locator token like [M10#1|章节:...; 段落:...].\n"
        "- If uncertain, still give conservative decision grounded by the first Miller locator.\n"
        "- final_output must be an empty string \"\"; do not add extra keys.\n"
    )


def _repair_to_strict_json(
    client: OpenAI,
    model: str,
    raw_text: str,
    max_tokens: int,
) -> Dict[str, Any]:
    repair_system = "You are a strict JSON formatter. Return JSON only. No explanations."
    repair_user = (
        "Convert the following model output into strict JSON with EXACT keys:\n"
        '{"question":"...","reasoning":"...","miller_decision":"...","vitaldb_decision":"...","final_output":"..."}\n'
        "No extra keys. No markdown. If missing fields, fill with conservative short Chinese text.\n\n"
        f"Source text:\n{raw_text}"
    )
    try:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": repair_user},
                ],
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": repair_user},
                ],
            )
    except Exception:
        return {}
    if not resp.choices:
        return {}
    content = resp.choices[0].message.content
    if content is None:
        return {}
    return _extract_json_obj(str(content))


def _extract_labeled_section(text: str, labels: List[str], stop_labels: List[str]) -> str:
    src = str(text or "")
    for lb in labels:
        pattern = rf"{re.escape(lb)}\s*[:：]?\s*(.*?)(?=\n(?:{'|'.join(re.escape(x) for x in stop_labels)})\s*[:：]?|$)"
        m = re.search(pattern, src, re.IGNORECASE | re.DOTALL)
        if m:
            value = str(m.group(1) or "").strip()
            if value:
                return value
    return ""


def _line_payload(line: str) -> str:
    s = str(line or "").strip()
    return re.sub(r"^(A:|Q:|【[^】]+】[:：]?)\s*", "", s, flags=re.IGNORECASE).strip()


def _parse_sections_with_fallback(text: str) -> Dict[str, Any]:
    src = _extract_qa_block(str(text or ""))
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]

    miller_labels = ["【决策干预（Miller）】", "【Miller决策】", "【Miller建议】", "【Miller】", "【决策干预】"]
    vital_labels = ["【决策干预（VitalDB）】", "【VitalDB决策】", "【VitalDB】"]
    reasoning_labels = ["【临床推理】", "[Clinical Reasoning]"]

    stop_labels = miller_labels + vital_labels + reasoning_labels + ["Q", "A"]
    reasoning = _extract_labeled_section(src, reasoning_labels, stop_labels)
    miller = _extract_labeled_section(src, miller_labels, stop_labels)
    vital = _extract_labeled_section(src, vital_labels, stop_labels)

    missing_miller_label = "【决策干预（Miller）】" not in src
    inferred_miller = False

    if not miller:
        lower_keywords = [
            "根据miller",
            "miller",
            "建议干预",
            "优先处理",
            "应考虑",
            "增加镇痛",
            "加深麻醉",
            "继续观察",
            "维持灌注",
            "升压",
        ]
        a_idx = next((i for i, ln in enumerate(lines) if re.match(r"(?i)^A\s*[:：]", ln)), -1)
        v_idx = next((i for i, ln in enumerate(lines) if any(lb in ln for lb in vital_labels)), len(lines))
        search_from = (a_idx + 1) if a_idx >= 0 else 0
        search_to = max(search_from, v_idx)
        for ln in lines[search_from:search_to]:
            low = ln.lower()
            if any(k in low for k in lower_keywords):
                cand = _line_payload(ln)
                if cand:
                    miller = cand
                    inferred_miller = True
                    break
        if not miller and len(lines) >= 3:
            cand = _line_payload(lines[2])
            if cand:
                miller = cand
                inferred_miller = True

    if not vital:
        for ln in lines:
            if any(lb in ln for lb in vital_labels):
                cand = _line_payload(ln)
                if cand:
                    vital = cand
                    break

    if not reasoning:
        for ln in lines:
            if "临床推理" in ln.lower():
                cand = _line_payload(ln)
                if cand:
                    reasoning = cand
                    break

    q_line = ""
    for ln in lines:
        if re.match(r"(?i)^Q\s*[:：]", ln):
            q_line = ln
            break

    return {
        "normalized_text": src,
        "lines": lines,
        "q_line": q_line,
        "reasoning_section": reasoning,
        "miller_section": miller,
        "vitaldb_section": vital,
        "missing_miller_label": bool(missing_miller_label),
        "inferred_miller": bool(inferred_miller),
    }


def _repair_missing_miller_label(
    client: OpenAI,
    model: str,
    raw_text: str,
    max_tokens: int,
) -> Optional[str]:
    repair_system = (
        "You are a strict medical QA formatter. "
        "Return only one final Chinese QA block with exact labels and no extra text."
    )
    repair_user = (
        "Rewrite the text into EXACTLY 4 lines:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句，尽量包含证据定位如[M10#1|章节:...; 段落:...]>\n"
        "【决策干预（VitalDB）】：<1-2句>\n\n"
        "Rules:\n"
        "- Keep labels exactly as written.\n"
        "- Do not add markdown, bullets, JSON, or extra commentary.\n"
        "- If Miller content exists but label is missing, map it into 【决策干预（Miller）】 line.\n"
        "- If uncertain, write a conservative Miller suggestion; do not omit the Miller line.\n\n"
        f"Source:\n{raw_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": repair_system},
                {"role": "user", "content": repair_user},
            ],
        )
    except Exception:
        return None
    if not resp.choices:
        return None
    content = resp.choices[0].message.content
    if content is None:
        return None
    return str(content).strip()


def _classify_invalid_reason(
    error: Optional[str],
    semantic_valid: bool,
    format_valid: bool,
    action_aligned: bool,
    missing_miller_label: bool,
) -> str:
    if str(error or "").strip():
        return "api_error"
    if not semantic_valid:
        return "no_semantic_miller_decision"
    if not format_valid and missing_miller_label:
        return "missing_miller_label"
    if not format_valid:
        return "strict_format_fail"
    if not action_aligned:
        return "action_align_fail"
    return ""


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
        llm_base_url=args.a_base_url,
        llm_api_key=args.a_api_key,
        api_key_env=args.a_api_key_env,
    )


def _prepare_retrieval(
    record: Dict[str, Any],
    retriever: Any,
    embed_client: Any,
    retrieval_cfg: Any,
    use_existing: bool,
) -> Optional[Dict[str, Any]]:
    if use_existing and isinstance(record.get("miller_retrieval"), dict):
        return record["miller_retrieval"]
    if not retrieval_cfg.enable_miller_rag:
        return None
    snapshot = record.get("snapshot") if isinstance(record.get("snapshot"), dict) else record
    if not isinstance(snapshot, dict):
        return None
    return retrieve_miller_context(snapshot, retriever, embed_client, retrieval_cfg)


def _run_single_model(
    client: OpenAI,
    model: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    web_search: Optional[Dict[str, Any]],
    max_tokens: int,
    structured_output_mode: str = "off",
) -> Dict[str, Any]:
    if structured_output_mode == "json_then_render":
        prompt = _build_structured_json_prompt(snapshot, retrieval, web_search)
        system_prompt = (
            "You are a strict clinical JSON generator. "
            "Return one JSON object only. Never explain your reasoning or instruction conflicts."
        )
    else:
        prompt = build_user_prompt(snapshot, retrieval=retrieval)
        prompt += _format_web_context_block(web_search)
        system_prompt = SYSTEM_PROMPT
    try:
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if structured_output_mode == "json_then_render":
                request_kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**request_kwargs)
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
    except Exception as e:  # noqa: BLE001
        return {
            "error": str(e),
            "raw_output": None,
            "final_output": None,
            "strict_qa": False,
            "format_valid": False,
            "semantic_valid": False,
            "strict_valid": False,
            "action_aligned": False,
            "repaired": False,
            "repair_path": "",
            "valid": False,
            "invalid_reason": "api_error",
            "quality_flags": ["api_error"],
            "miller_section": "",
            "vitaldb_section": "",
            "miller_section_inferred": False,
            "missing_miller_label": True,
        }

    if not resp.choices:
        return {
            "error": "empty_choices",
            "raw_output": None,
            "final_output": None,
            "strict_qa": False,
            "format_valid": False,
            "semantic_valid": False,
            "strict_valid": False,
            "action_aligned": False,
            "repaired": False,
            "repair_path": "",
            "valid": False,
            "invalid_reason": "api_error",
            "quality_flags": ["api_error"],
            "miller_section": "",
            "vitaldb_section": "",
            "miller_section_inferred": False,
            "missing_miller_label": True,
        }

    content = resp.choices[0].message.content
    raw = content.strip() if isinstance(content, str) else str(content or "").strip()
    model_text = _extract_qa_block(raw)
    json_obj: Dict[str, Any] = {}
    if structured_output_mode == "json_then_render":
        json_obj = _extract_json_obj(raw)
        if json_obj and _json_payload_is_meta_or_low_quality(json_obj):
            json_obj = {}
        if not json_obj:
            json_obj = _repair_to_strict_json(client, model, raw, max_tokens)
        if json_obj and _json_payload_is_meta_or_low_quality(json_obj):
            json_obj = {}
        if isinstance(json_obj, dict) and json_obj:
            model_text = _render_qa_from_json_payload(json_obj, retrieval)

    # Step 1 + Step 2
    parsed = parse_sections_relaxed(model_text)
    # Step 3
    parsed = semantic_repair(model_text, parsed)
    parsed["miller_section"] = _inject_miller_locator_if_missing(parsed.get("miller_section"), retrieval)
    # Step 4
    final_text, parsed, repaired_used, repair_path = rewrite_if_needed(
        client=client,
        model=model,
        raw_text=raw,
        snapshot=snapshot,
        max_tokens=max_tokens,
        parsed=parsed,
    )
    parsed["miller_section"] = _inject_miller_locator_if_missing(parsed.get("miller_section"), retrieval)
    # Step 5
    validated = final_validate(parsed, snapshot)
    final_text = str(validated.get("rendered_output") or final_text)

    quality_flags: List[str] = []
    if not validated["format_valid"]:
        quality_flags.append("format_invalid")
    if bool(parsed.get("missing_miller_label")):
        quality_flags.append("missing_miller_label")
    if not validated["action_aligned"]:
        quality_flags.append("action_not_aligned")
    if structured_output_mode == "json_then_render":
        quality_flags.append("json_mode")
        if bool(json_obj):
            quality_flags.append("json_parsed")
        else:
            quality_flags.append("json_parse_failed")

    return {
        "error": None,
        "raw_output": raw,
        "final_output": final_text,
        "strict_qa": bool(validated["strict_qa"]),
        "format_valid": bool(validated["format_valid"]),
        "semantic_valid": bool(validated["semantic_valid"]),
        "strict_valid": bool(validated["strict_valid"]),
        "action_aligned": bool(validated["action_aligned"]),
        "repaired": repaired_used,
        "repair_path": repair_path,
        "valid": bool(validated["valid"]),
        "invalid_reason": str(validated["invalid_reason"] or ""),
        "failure_type": str(validated["failure_type"] or ""),
        "quality_flags": quality_flags,
        "miller_section": str(parsed.get("miller_section") or "").strip(),
        "vitaldb_section": str(parsed.get("vitaldb_section") or "").strip(),
        "miller_section_inferred": bool(parsed.get("inferred_miller")),
        "missing_miller_label": bool(parsed.get("missing_miller_label")),
        "normalized_output": str(parsed.get("normalized_text") or ""),
        "structured_json_used": bool(structured_output_mode == "json_then_render" and bool(json_obj)),
        "structured_json": json_obj if json_obj else None,
    }


def _compare_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_final = str(a.get("final_output") or "").strip()
    b_final = str(b.get("final_output") or "").strip()
    a_miller = str(a.get("miller_section") or "").strip().lower()
    b_miller = str(b.get("miller_section") or "").strip().lower()
    a_vital = str(a.get("vitaldb_section") or "").strip().lower()
    b_vital = str(b.get("vitaldb_section") or "").strip().lower()
    return {
        "both_valid": bool(a.get("valid")) and bool(b.get("valid")),
        "same_final_output": bool(a_final) and (a_final == b_final),
        "same_miller_decision": bool(a_miller) and (a_miller == b_miller),
        "same_vitaldb_decision": bool(a_vital) and (a_vital == b_vital),
    }


def _summarize(rows: List[Dict[str, Any]], label_a: str, label_b: str) -> Dict[str, Any]:
    total = len(rows)
    valid_a = sum(1 for row in rows if row[label_a].get("valid"))
    valid_b = sum(1 for row in rows if row[label_b].get("valid"))
    semantic_valid_a = sum(1 for row in rows if row[label_a].get("semantic_valid"))
    semantic_valid_b = sum(1 for row in rows if row[label_b].get("semantic_valid"))
    format_valid_a = sum(1 for row in rows if row[label_a].get("format_valid"))
    format_valid_b = sum(1 for row in rows if row[label_b].get("format_valid"))
    strict_valid_a = sum(1 for row in rows if row[label_a].get("strict_valid"))
    strict_valid_b = sum(1 for row in rows if row[label_b].get("strict_valid"))
    both_valid = sum(1 for row in rows if row["comparison"].get("both_valid"))
    same_final = sum(1 for row in rows if row["comparison"].get("same_final_output"))
    same_miller = sum(1 for row in rows if row["comparison"].get("same_miller_decision"))
    same_vitaldb = sum(1 for row in rows if row["comparison"].get("same_vitaldb_decision"))
    invalid_reason_a: Dict[str, int] = {}
    invalid_reason_b: Dict[str, int] = {}
    failure_type_a: Dict[str, int] = {}
    failure_type_b: Dict[str, int] = {}
    for row in rows:
        reason_a = str(row[label_a].get("invalid_reason") or "").strip()
        reason_b = str(row[label_b].get("invalid_reason") or "").strip()
        ftype_a = str(row[label_a].get("failure_type") or "").strip()
        ftype_b = str(row[label_b].get("failure_type") or "").strip()
        if reason_a:
            invalid_reason_a[reason_a] = invalid_reason_a.get(reason_a, 0) + 1
        if reason_b:
            invalid_reason_b[reason_b] = invalid_reason_b.get(reason_b, 0) + 1
        if ftype_a:
            failure_type_a[ftype_a] = failure_type_a.get(ftype_a, 0) + 1
        if ftype_b:
            failure_type_b[ftype_b] = failure_type_b.get(ftype_b, 0) + 1
    return {
        "total": total,
        label_a: {
            "valid": valid_a,
            "semantic_valid": semantic_valid_a,
            "format_valid": format_valid_a,
            "strict_valid": strict_valid_a,
            "invalid_reasons": invalid_reason_a,
            "failure_types": failure_type_a,
        },
        label_b: {
            "valid": valid_b,
            "semantic_valid": semantic_valid_b,
            "format_valid": format_valid_b,
            "strict_valid": strict_valid_b,
            "invalid_reasons": invalid_reason_b,
            "failure_types": failure_type_b,
        },
        "both_valid": both_valid,
        "same_final_output": same_final,
        "same_miller_decision": same_miller,
        "same_vitaldb_decision": same_vitaldb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two model outputs on the same anesthesia snapshots.")
    parser.add_argument("--input", required=True, help="Input JSONL/JSON containing records with `snapshot`.")
    parser.add_argument("--output-jsonl", default="", help="Output comparison JSONL path.")
    parser.add_argument("--summary-json", default="", help="Output summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to compare; 0 means all.")
    parser.add_argument("--max-tokens", type=int, default=700, help="Generation max tokens per model call.")
    parser.add_argument(
        "--structured-output-mode",
        default="off",
        choices=["off", "json_then_render"],
        help="off: keep legacy text output parsing; json_then_render: ask model for JSON first, then render to 4-line labels.",
    )

    parser.add_argument("--a-label", default="local_model")
    parser.add_argument("--a-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--a-model", required=True)
    parser.add_argument("--a-api-key", default="local")
    parser.add_argument("--a-api-key-env", default="OPENAI_API_KEY")

    parser.add_argument("--b-label", default="gpt_api")
    parser.add_argument("--b-base-url", default="")
    parser.add_argument("--b-model", required=True)
    parser.add_argument("--b-api-key", default="")
    parser.add_argument("--b-api-key-env", default="OPENAI_API_KEY")

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
    parser.add_argument("--enable-gpt-search-retrieval", action="store_true")
    parser.add_argument("--gpt-search-target", default="b", choices=["a", "b", "both"])
    parser.add_argument("--gpt-search-model", default="gpt-5-search-api")
    parser.add_argument("--gpt-search-api-url", default="")
    parser.add_argument("--gpt-search-api-key", default="")
    parser.add_argument("--gpt-search-api-key-env", default="GPT_API_KEY")
    parser.add_argument("--gpt-search-top-k", type=int, default=3)
    parser.add_argument("--gpt-search-max-tokens", type=int, default=700)
    parser.add_argument(
        "--gpt-search-query-suffix",
        default="Miller's Anesthesia 10th edition intraoperative anesthesia hemodynamic depth management evidence",
    )
    parser.add_argument("--force-gpt-search-miller10", action="store_true")
    parser.add_argument("--gpt-search-miller10-min-score", type=int, default=2)
    args = parser.parse_args()

    records = _load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records loaded from input.")

    label_a = args.a_label.strip() or "model_a"
    label_b = args.b_label.strip() or "model_b"
    if label_a == label_b:
        raise ValueError("--a-label and --b-label must be different")

    client_a = _build_client(args.a_base_url, args.a_api_key, args.a_api_key_env)
    client_b = _build_client(args.b_base_url, args.b_api_key, args.b_api_key_env)

    retriever = None
    embed_client = None
    retrieval_cfg = _build_retrieval_cfg(args)
    if args.enable_miller_rag and not args.use_existing_retrieval:
        embed_client = create_embedding_client(retrieval_cfg)
        retriever = build_miller_retriever(embed_client, retrieval_cfg)

    search_api_url = ""
    search_headers: Dict[str, str] = {}
    if args.enable_gpt_search_retrieval:
        search_api_url = _normalize_chat_completions_url(args.gpt_search_api_url or args.b_base_url)
        if not search_api_url:
            raise ValueError("gpt-search retrieval enabled but no API URL available; set --gpt-search-api-url or --b-base-url")
        search_api_key = _resolve_api_key(args.gpt_search_api_key, args.gpt_search_api_key_env)
        if not search_api_key:
            search_api_key = _resolve_api_key(args.b_api_key, args.b_api_key_env)
        search_headers = _build_headers(search_api_key)

    out_jsonl = args.output_jsonl or str(Path(args.input).with_suffix("")) + ".model_compare.jsonl"
    summary_json = args.summary_json or str(Path(args.input).with_suffix("")) + ".model_compare.summary.json"
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for idx, record in enumerate(records, start=1):
            snapshot = record.get("snapshot") if isinstance(record.get("snapshot"), dict) else record
            if not isinstance(snapshot, dict):
                continue
            retrieval = _prepare_retrieval(record, retriever, embed_client, retrieval_cfg, args.use_existing_retrieval)
            web_search_shared: Optional[Dict[str, Any]] = None
            web_search_a: Optional[Dict[str, Any]] = None
            web_search_b: Optional[Dict[str, Any]] = None
            if args.enable_gpt_search_retrieval:
                web_search_shared = _search_with_gpt_search_api(
                    snapshot=snapshot,
                    retrieval=retrieval,
                    args=args,
                    api_url=search_api_url,
                    headers=search_headers,
                )
                if args.gpt_search_target in {"a", "both"}:
                    web_search_a = web_search_shared
                if args.gpt_search_target in {"b", "both"}:
                    web_search_b = web_search_shared

            out_a = _run_single_model(
                client_a,
                args.a_model,
                snapshot,
                retrieval,
                web_search_a,
                args.max_tokens,
                structured_output_mode=args.structured_output_mode,
            )
            out_b = _run_single_model(
                client_b,
                args.b_model,
                snapshot,
                retrieval,
                web_search_b,
                args.max_tokens,
                structured_output_mode=args.structured_output_mode,
            )
            comparison = _compare_pair(out_a, out_b)
            row = {
                "index": idx,
                "caseid": record.get("caseid"),
                "snapshot": snapshot,
                "miller_retrieval": retrieval,
                "web_search_a": web_search_a,
                "web_search_b": web_search_b,
                label_a: out_a,
                label_b: out_b,
                "comparison": comparison,
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  - compared {idx}/{len(records)}")

    summary = _summarize(rows, label_a, label_b)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Model Comparison Summary ===")
    print(f"input:   {args.input}")
    print(f"jsonl:   {out_jsonl}")
    print(f"summary: {summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
