import argparse
import copy
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

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
    evaluate_vitaldb_vs_miller,
    retrieve_miller_context,
)

RULES_DIR = Path(__file__).resolve().parent / "rules"
MILLER_SUPPORT_RULES_PATH = RULES_DIR / "miller_support_rules.yaml"
_SUPPORT_RULES_CACHE: Optional[Dict[str, Any]] = None


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


def _resolve_api_key(explicit_key: str, env_name: str) -> str:
    key = (explicit_key or "").strip()
    if key:
        return key
    if env_name:
        return os.getenv(env_name, "").strip()
    return ""


def _resolve_api_key_for_url(api_url: str, explicit_key: str, env_name: str) -> str:
    key = _resolve_api_key(explicit_key, env_name)
    if key:
        return key
    lower_url = (api_url or "").strip().lower()
    if "127.0.0.1" in lower_url or "localhost" in lower_url:
        return "local"
    return ""


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    key = api_key.strip()
    if key:
        headers["Authorization"] = key if key.lower().startswith("bearer ") else f"Bearer {key}"
    return headers


def _build_retrieval_cfg(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        enable_miller_rag=True,
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


def _build_web_search_query(snapshot: Dict[str, Any], retrieval: Optional[Dict[str, Any]], suffix: str) -> str:
    if isinstance(retrieval, dict):
        for key in ("query_rewritten", "query", "query_raw"):
            value = str(retrieval.get(key) or "").strip()
            if value:
                base = value
                break
        else:
            base = ""
    else:
        base = ""

    if not base:
        pb = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
        preop = snapshot.get("preop_context", [])
        risk_flags = (
            snapshot.get("clinical_assessment", {}).get("risk_flags", [])
            if isinstance(snapshot.get("clinical_assessment"), dict)
            else []
        )
        base = "; ".join(
            [
                str(pb.get("department") or "").strip(),
                str(pb.get("surgery_group") or "").strip(),
                str(snapshot.get("surgery_type") or "").strip(),
                ", ".join(str(x) for x in preop[:2]) if isinstance(preop, list) else str(preop),
                ", ".join(str(x) for x in risk_flags[:2]) if isinstance(risk_flags, list) else str(risk_flags),
            ]
        ).strip("; ")

    query = f"{base}; {suffix}".strip("; ")
    return " ".join(query.split())


def _http_get_json(url: str, params: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    obj = resp.json()
    return obj if isinstance(obj, dict) else {}


def _http_post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    obj = resp.json()
    return obj if isinstance(obj, dict) else {}


def _search_tavily(query: str, args: argparse.Namespace, api_key: str) -> List[Dict[str, Any]]:
    # Keep placeholders defined to avoid NameError in legacy f-strings below.
    actual = ""
    med_key = ""
    kw_text = ""
    raw_text = ""
    repair_sys = (
        "You are a strict medical QA formatter. "
        "Return only final QA in Chinese. "
        "No thinking process, no bullets, no instruction echo, no markdown fences."
    )
    repair_user = (
        "Rewrite to strict output format. Output only:\n"
        "Q: ...\n"
        "A: 【临床推理】：...\n"
        "【决策干预（Miller）】：...\n"
        "【决策干预（VitalDB）】：...\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        "【决策干预（VitalDB）】 must be same drug class/category as golden logged_action.\n"
        "Source text:\n"
        f"{raw_text}"
    )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, int(args.web_search_top_k)),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    if args.web_search_domains:
        payload["include_domains"] = [x.strip() for x in args.web_search_domains.split(",") if x.strip()]
    obj = _http_post_json("https://api.tavily.com/search", payload, int(args.web_search_timeout_sec))
    items = obj.get("results", [])
    out: List[Dict[str, Any]] = []
    if isinstance(items, list):
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "rank": idx,
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(item.get("content") or "").strip(),
                    "score": float(item.get("score") or 0.0),
                    "provider": "tavily",
                }
            )
    return out


def _search_serpapi(query: str, args: argparse.Namespace, api_key: str) -> List[Dict[str, Any]]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max(1, int(args.web_search_top_k)),
    }
    obj = _http_get_json("https://serpapi.com/search.json", params, int(args.web_search_timeout_sec))
    items = obj.get("organic_results", [])
    out: List[Dict[str, Any]] = []
    if isinstance(items, list):
        for idx, item in enumerate(items[: max(1, int(args.web_search_top_k))], start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "rank": idx,
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("link") or "").strip(),
                    "snippet": str(item.get("snippet") or "").strip(),
                    "score": 0.0,
                    "provider": "serpapi",
                }
            )
    return out


def _search_custom(query: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.web_search_url:
        raise ValueError("custom web search requires --web-search-url")
    payload = {
        "query": query,
        "top_k": max(1, int(args.web_search_top_k)),
        "domains": [x.strip() for x in args.web_search_domains.split(",") if x.strip()] if args.web_search_domains else [],
    }
    obj = _http_post_json(args.web_search_url, payload, int(args.web_search_timeout_sec))
    items = obj.get("results", [])
    out: List[Dict[str, Any]] = []
    if isinstance(items, list):
        for idx, item in enumerate(items[: max(1, int(args.web_search_top_k))], start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "rank": idx,
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(item.get("snippet") or item.get("content") or "").strip(),
                    "score": float(item.get("score") or 0.0),
                    "provider": "custom",
                }
            )
    return out


def _run_web_search(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query = _build_web_search_query(snapshot, retrieval, args.web_search_query_suffix)
    api_key = _resolve_api_key(args.web_search_api_key, args.web_search_api_key_env)
    provider = args.web_search_provider
    try:
        if provider == "tavily":
            if not api_key:
                raise ValueError("missing API key for tavily")
            results = _search_tavily(query, args, api_key)
        elif provider == "serpapi":
            if not api_key:
                raise ValueError("missing API key for serpapi")
            results = _search_serpapi(query, args, api_key)
        elif provider == "custom":
            results = _search_custom(query, args)
        else:
            raise ValueError(f"unsupported provider: {provider}")
        return {"enabled": True, "provider": provider, "query": query, "results": results, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"enabled": True, "provider": provider, "query": query, "results": [], "error": str(e)}


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


def _count_results(web_search: Optional[Dict[str, Any]]) -> int:
    if not isinstance(web_search, dict):
        return 0
    items = web_search.get("results", [])
    if not isinstance(items, list):
        return 0
    return sum(1 for item in items if isinstance(item, dict))


def _search_with_gpt_search_api(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    api_url: str,
    headers: Dict[str, str],
    query_override: str = "",
    search_model: str = "",
    search_top_k: int = 0,
    force_miller10: bool = False,
    miller10_min_score: int = 3,
    strict_evidence_gate: bool = False,
) -> Dict[str, Any]:
    query = (query_override or "").strip() or _build_web_search_query(snapshot, retrieval, args.web_search_query_suffix)
    top_k = max(1, int(search_top_k or args.gpt_search_top_k))
    model_name = (search_model or args.gpt_search_model).strip()
    if force_miller10 and strict_evidence_gate:
        user_prompt = (
            "Search online evidence for this anesthesia scenario.\n"
            f"Query: {query}\n\n"
            f"Return top {top_k} results as strict JSON object only:\n"
            '{"query":"...","results":[{"title":"...","url":"...","snippet":"...","score":0.0}]}\n'
            "Hard Rules:\n"
            "- Only keep results that are explicitly about Miller's Anesthesia 10th edition.\n"
            "- If no Miller 10th evidence is found, return results as [].\n"
            "- Keep snippet concise and clinically relevant.\n"
            "- No markdown fences, no extra text outside JSON.\n"
        )
    elif force_miller10:
        user_prompt = (
            "Search online evidence for this anesthesia scenario.\n"
            f"Query: {query}\n\n"
            f"Return top {top_k} results as strict JSON object only:\n"
            '{"query":"...","results":[{"title":"...","url":"...","snippet":"...","score":0.0}]}\n'
            "Rules:\n"
            "- Prioritize Miller's Anesthesia 10th edition evidence first.\n"
            "- If exact Miller 10th hits are limited, include closest authoritative anesthesia sources.\n"
            "- Keep snippet concise and clinically relevant.\n"
            "- No markdown fences, no extra text outside JSON.\n"
        )
    else:
        user_prompt = (
            "Search online evidence for this anesthesia scenario, prioritizing Miller's Anesthesia 10th edition.\n"
            f"Query: {query}\n\n"
            f"Return top {top_k} results as strict JSON object only:\n"
            '{"query":"...","results":[{"title":"...","url":"...","snippet":"...","score":0.0}]}\n'
            "Rules:\n"
            "- Prioritize Miller's Anesthesia 10th edition or closely related authoritative anesthesia references.\n"
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
            model=model_name,
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
        if force_miller10:
            filtered = _filter_miller10_results(results, int(miller10_min_score))
            if strict_evidence_gate:
                results = filtered
            elif filtered:
                results = filtered

        if not results and raw and not force_miller10:
            results = [
                {
                    "rank": 1,
                    "title": "gpt_search_api_raw",
                    "url": "",
                    "snippet": str(raw).strip()[:800],
                    "score": 0.0,
                    "provider": "gpt_search_api",
                }
            ]
        error = None
        if force_miller10 and strict_evidence_gate and not results:
            error = "no_miller10_hits"
        return {
            "enabled": True,
            "provider": "gpt_search_api",
            "query": str(parsed.get("query") or query),
            "results": results,
            "error": error,
            "raw_response": raw,
            "force_miller10": bool(force_miller10),
            "miller10_hit_count": len(results),
            "strict_evidence_gate": bool(strict_evidence_gate),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "enabled": True,
            "provider": "gpt_search_api",
            "query": query,
            "results": [],
            "error": str(e),
            "force_miller10": bool(force_miller10),
            "miller10_hit_count": 0,
            "strict_evidence_gate": bool(strict_evidence_gate),
        }


def _format_web_context_block(web_search: Optional[Dict[str, Any]]) -> str:
    if not isinstance(web_search, dict) or not web_search.get("enabled"):
        return ""
    results = web_search.get("results", [])
    if not isinstance(results, list) or not results:
        return ""
    lines = [
        "",
        "Supplemental web search snippets (may include non-Miller sources; use cautiously):",
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
    lines.append(
        "Do not treat web snippets as authoritative Miller evidence unless they clearly match the retrieved Miller text."
    )
    return "\n".join(lines)


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


def _extract_miller_only(final_output: Optional[str]) -> str:
    if not final_output:
        return ""
    return _decision_section(final_output).strip()


def _extract_vitaldb_only(final_output: Optional[str]) -> str:
    if not final_output:
        return ""
    return _decision_section_vitaldb(final_output).strip()


def _has_required_sections(text: str) -> bool:
    value = str(text or "")
    required = ["【临床推理】", "【决策干预（Miller）】", "【决策干预（VitalDB）】"]
    return all(tag in value for tag in required)


def _has_required_sections_strict_v2(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    required = ["【临床推理】", "【决策干预（Miller）】", "【决策干预（VitalDB）】"]
    if not all(tag in value for tag in required):
        return False
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 4:
        return False
    if not lines[0].startswith("Q:"):
        return False
    if not lines[1].startswith("A:"):
        return False
    if not lines[2].startswith("【决策干预（Miller）】"):
        return False
    if not lines[3].startswith("【决策干预（VitalDB）】"):
        return False
    miller_line = lines[2]
    has_locator = bool(re.search(r"(?i)m10\s*#\d+", miller_line)) or ("章节:" in miller_line and "段落:" in miller_line)
    return has_locator


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower().replace("\r", "\n")


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _default_support_rules() -> Dict[str, Any]:
    return {
        "concepts": {
            "perfusion_first": {
                "out": ["灌注优先", "先稳灌注", "先纠正低血压", "perfusion first", "stabilize perfusion"],
                "ev": ["hypotension", "perfusion", "blood pressure", "hemodynamic", "vasopressor"],
            },
            "avoid_deepen_when_hypotension": {
                "out": ["避免加深", "不宜加深", "不要加深", "avoid deepening", "do not deepen"],
                "ev": ["hypotension", "map", "anesthetic depth", "depth of anesthesia", "hemodynamic"],
            },
            "bis_with_context": {
                "out": ["bis", "结合刺激", "不是单独触发", "not standalone trigger"],
                "ev": ["bis", "eeg", "stimulation", "artifact", "awareness"],
            },
            "opioid_titration": {
                "out": ["阿片", "瑞芬太尼", "opioid", "remifentanil", "analgesia"],
                "ev": ["opioid", "fentanyl", "remifentanil", "noxious stimulation"],
            },
            "vasopressor_logic": {
                "out": ["去氧肾上腺素", "麻黄碱", "去甲肾上腺素", "phenylephrine", "ephedrine", "norepinephrine"],
                "ev": ["phenylephrine", "ephedrine", "norepinephrine", "vasopressor", "hypotension"],
            },
        },
        "anchors": [
            ["map", "平均动脉压", "灌注压"],
            ["bis", "脑电", "麻醉深度"],
            ["hypotension", "低血压"],
            ["perfusion", "灌注"],
            ["opioid", "阿片", "瑞芬太尼", "remifentanil"],
            ["vasopressor", "升压药", "去甲肾上腺素", "麻黄碱", "去氧肾上腺素"],
        ],
        "scoring": {
            "no_output_score": -1,
            "no_evidence_score": -1,
            "no_claim_score": -1,
            "unsupported_score": -2,
            "weakly_supported_score": 1,
            "partially_supported_score": 2,
            "supported_strict_score": 3,
            "strict_min_anchor_hits": 1,
            "partial_ratio_threshold": 0.5,
        },
    }


def _load_support_rules() -> Dict[str, Any]:
    global _SUPPORT_RULES_CACHE  # noqa: PLW0603
    if _SUPPORT_RULES_CACHE is not None:
        return _SUPPORT_RULES_CACHE
    rules = _default_support_rules()
    try:
        if MILLER_SUPPORT_RULES_PATH.exists() and yaml is not None:
            loaded = yaml.safe_load(MILLER_SUPPORT_RULES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                rules = loaded
    except Exception:  # noqa: BLE001
        pass
    _SUPPORT_RULES_CACHE = rules
    return rules


def _semantic_support_by_judge(
    concepts: List[str],
    miller_output: str,
    evidence_text: str,
    cfg: Optional[Dict[str, Any]],
) -> Set[str]:
    if not concepts or not isinstance(cfg, dict) or not cfg.get("enabled"):
        return set()
    api_url = str(cfg.get("api_url") or "").strip()
    model = str(cfg.get("model") or "").strip()
    headers = cfg.get("headers") if isinstance(cfg.get("headers"), dict) else {}
    max_tokens = int(cfg.get("max_tokens") or 300)
    if not api_url or not model:
        return set()

    prompt = (
        "You are a strict medical evidence judge.\n"
        "Task: Determine which claimed concepts are semantically supported by evidence.\n"
        "Return JSON only: {\"supported_concepts\": [..], \"reason\": \"...\"}\n"
        f"Allowed concepts: {json.dumps(concepts, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Use semantic equivalence, not exact keyword matching.\n"
        "- If concept is unsupported or uncertain, do not include it.\n"
        "- Do not invent new concept names.\n\n"
        f"[Model claim text]\n{miller_output}\n\n"
        f"[Evidence text]\n{evidence_text[:6000]}\n"
    )
    try:
        raw = _post_chat(
            api_url=api_url,
            headers=headers,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            system_prompt="Return strict JSON only.",
        )
        obj = _extract_json_obj(raw)
        items = obj.get("supported_concepts", [])
        out: Set[str] = set()
        if isinstance(items, list):
            for item in items:
                value = str(item).strip()
                if value in concepts:
                    out.add(value)
        return out
    except Exception:  # noqa: BLE001
        return set()


def _concept_support_eval(
    miller_output: str,
    evidence_text: str,
    semantic_judge_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rules_cfg = _load_support_rules()
    concept_rules = rules_cfg.get("concepts", {}) if isinstance(rules_cfg.get("concepts"), dict) else {}
    anchors = rules_cfg.get("anchors", []) if isinstance(rules_cfg.get("anchors"), list) else []
    scoring = rules_cfg.get("scoring", {}) if isinstance(rules_cfg.get("scoring"), dict) else {}

    no_output_score = int(scoring.get("no_output_score", -1))
    no_evidence_score = int(scoring.get("no_evidence_score", -1))
    no_claim_score = int(scoring.get("no_claim_score", -1))
    unsupported_score = int(scoring.get("unsupported_score", -2))
    weak_score = int(scoring.get("weakly_supported_score", 1))
    partial_score = int(scoring.get("partially_supported_score", 2))
    strict_score = int(scoring.get("supported_strict_score", 3))
    strict_min_anchor_hits = int(scoring.get("strict_min_anchor_hits", 1))
    partial_ratio_threshold = float(scoring.get("partial_ratio_threshold", 0.5))

    out = _normalize_text(miller_output)
    ev = _normalize_text(evidence_text)
    if not out:
        return {
            "verdict": "no_miller_output",
            "score": no_output_score,
            "support_ratio": 0.0,
            "claimed_concepts": [],
            "supported_concepts": [],
            "anchor_hits": 0,
        }
    if not ev:
        return {
            "verdict": "no_evidence",
            "score": no_evidence_score,
            "support_ratio": 0.0,
            "claimed_concepts": [],
            "supported_concepts": [],
            "anchor_hits": 0,
        }

    claimed: Set[str] = set()
    supported: Set[str] = set()
    for concept, cfg in concept_rules.items():
        if not isinstance(cfg, dict):
            continue
        out_keywords = [str(x).strip().lower() for x in cfg.get("out", []) if str(x).strip()]
        ev_keywords = [str(x).strip().lower() for x in cfg.get("ev", []) if str(x).strip()]
        if _contains_any(out, out_keywords):
            claimed.add(concept)
            if _contains_any(ev, ev_keywords):
                supported.add(concept)

    if claimed and semantic_judge_cfg and semantic_judge_cfg.get("enabled"):
        unsupported_claims = sorted(list(claimed - supported))
        semantic_supported = _semantic_support_by_judge(
            unsupported_claims,
            miller_output=miller_output,
            evidence_text=evidence_text,
            cfg=semantic_judge_cfg,
        )
        supported.update(semantic_supported)

    anchor_hits = 0
    for kws in anchors:
        kw_list = [str(x).strip().lower() for x in kws if str(x).strip()]
        if _contains_any(out, kw_list) and _contains_any(ev, kw_list):
            anchor_hits += 1

    if not claimed:
        verdict = "no_explicit_claim"
        score = no_claim_score
        ratio = 0.0
    else:
        ratio = len(supported) / max(1, len(claimed))
        if ratio >= 1.0 and anchor_hits >= strict_min_anchor_hits:
            verdict, score = "supported_strict", strict_score
        elif ratio >= partial_ratio_threshold:
            verdict, score = "partially_supported", partial_score
        elif ratio > 0:
            verdict, score = "weakly_supported", weak_score
        else:
            verdict, score = "unsupported", unsupported_score

    return {
        "verdict": verdict,
        "score": score,
        "support_ratio": round(float(ratio), 4),
        "claimed_concepts": sorted(list(claimed)),
        "supported_concepts": sorted(list(supported)),
        "anchor_hits": anchor_hits,
        "semantic_judge_used": bool(semantic_judge_cfg and semantic_judge_cfg.get("enabled")),
    }


def _build_evidence_text(retrieval: Optional[Dict[str, Any]]) -> str:
    if not isinstance(retrieval, dict):
        return ""
    results = retrieval.get("results", [])
    if not isinstance(results, list):
        return ""
    chunks: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _alignment_from_generated_output(snapshot: Dict[str, Any], final_output: Optional[str]) -> Dict[str, Any]:
    text = str(final_output or "").strip()
    if not text:
        return {
            "verdict": "unavailable",
            "reason": "empty_final_output",
            "high_risk_conflict": False,
        }
    vitaldb_text = _extract_vitaldb_only(text)
    if not vitaldb_text:
        return {
            "verdict": "unavailable",
            "reason": "missing_vitaldb_section",
            "high_risk_conflict": False,
        }

    snap_eval = copy.deepcopy(snapshot)
    snap_eval["actual_intervention"] = vitaldb_text
    anchor = snap_eval.get("anchor_detail")
    if not isinstance(anchor, dict):
        anchor = {}
        snap_eval["anchor_detail"] = anchor
    # Avoid forcing class/direction from original logged anchor when evaluating generated decisions.
    anchor["medication_key"] = ""
    anchor["delta"] = None
    anchor["before"] = None
    anchor["after"] = None

    alignment = evaluate_vitaldb_vs_miller(snap_eval)
    if not isinstance(alignment, dict):
        return {
            "verdict": "unavailable",
            "reason": "alignment_eval_non_dict",
            "high_risk_conflict": False,
        }
    return alignment


def _verdict_score(verdict: str) -> int:
    table = {
        "aligned": 3,
        "partially_aligned": 2,
        "uncertain": 1,
        "misaligned": 0,
        "potentially_inaccurate": 0,
        "unavailable": -1,
    }
    return table.get(str(verdict or "").strip().lower(), 0)


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
        "You are a strict medical QA formatter. "
        "Return only final QA in Chinese. "
        "No thinking process, no bullets, no markdown, no instruction echo, no extra prefixes/suffixes."
    )
    repair_user = (
        "Rewrite to strict format. You MUST output EXACTLY this 4-line template:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句>\n"
        "【决策干预（VitalDB）】：<1-2句>\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        "Do not use markdown headings, bullets, code blocks, or JSON.\n"
        "If information is insufficient, keep all sections and write a conservative sentence instead of omitting sections.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        "【决策干预（VitalDB）】 must be same drug class/category as golden logged_action.\n"
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


def _format_contract_block(snapshot: Dict[str, Any]) -> str:
    hint = _golden_action_hint(snapshot)
    kws = hint.get("keywords", [])
    kw_text = ", ".join(str(k) for k in kws if str(k).strip()) if isinstance(kws, list) else ""
    keyword_line = (
        f"- In 【决策干预（VitalDB）】, prefer these action keywords when applicable: {kw_text}\n"
        if kw_text
        else ""
    )
    return (
        "\n\n[Output Contract - Must Follow]\n"
        "Output ONLY the final answer and NOTHING else.\n"
        "Use EXACTLY the following 4-line template:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句>\n"
        "【决策干预（VitalDB）】：<1-2句>\n"
        "Hard constraints:\n"
        "- Keep the three section labels exactly as written.\n"
        "- Do not omit brackets or colons.\n"
        "- No markdown, bullets, JSON, XML, or extra commentary.\n"
        + keyword_line
    )


def _format_contract_block_v2(snapshot: Dict[str, Any], retrieval: Optional[Dict[str, Any]]) -> str:
    hint = _golden_action_hint(snapshot)
    kws = hint.get("keywords", [])
    kw_text = ", ".join(str(k) for k in kws if str(k).strip()) if isinstance(kws, list) else ""
    keyword_line = (
        f"- In 【决策干预（VitalDB）】 prefer these action keywords when applicable: {kw_text}\n"
        if kw_text
        else ""
    )
    locator_lines: List[str] = []
    if isinstance(retrieval, dict):
        results = retrieval.get("results", [])
        if isinstance(results, list):
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                rank = item.get("rank", "?")
                chapter = str(item.get("chapter") or "").strip() or "未知"
                paragraph = str(item.get("paragraph") or "").strip() or str(item.get("chunk_id") or "未知")
                section = str(item.get("section") or "").strip()
                section_piece = f"; 小节:{section}" if section else ""
                locator_lines.append(f"[M10#{rank}|章节:{chapter}{section_piece}; 段落:{paragraph}]")
    locator_hint = ""
    if locator_lines:
        locator_hint = "Evidence locators (must cite at least one in Miller line): " + " ".join(locator_lines) + "\n"
    return (
        "\n\n[Output Contract - Must Follow]\n"
        "Output ONLY the final answer and NOTHING else.\n"
        "Use EXACTLY the following 4-line template:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句，且必须包含证据定位如[M10#1|章节:...; 段落:...]>\n"
        "【决策干预（VitalDB）】：<1-2句>\n"
        "Hard constraints:\n"
        "- Keep the three section labels exactly as written.\n"
        "- Do not omit brackets or colons.\n"
        "- No markdown, bullets, JSON, XML, or extra commentary.\n"
        "- Miller line must include at least one evidence locator token M10#N.\n"
        + keyword_line
        + locator_hint
    )


def _repair_via_requests_v2(
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
        "You are a strict medical QA formatter. "
        "Return only final QA in Chinese. "
        "No thinking process, no bullets, no markdown, no instruction echo, no extra prefixes/suffixes."
    )
    repair_user = (
        "Rewrite to strict format. You MUST output EXACTLY this 4-line template:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句，且必须包含证据定位标签如[M10#1|章节:...; 段落:...]>\n"
        "【决策干预（VitalDB）】：<1-2句>\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        "Do not use markdown headings, bullets, code blocks, or JSON.\n"
        "Miller line must include at least one M10 locator token.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        "【决策干预（VitalDB）】must be same drug class/category as golden logged_action.\n"
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
    valid = strict_valid if validity_mode == "strict" else miller_valid
    return {
        "valid": valid,
        "strict_valid": strict_valid,
        "miller_valid": miller_valid,
        "miller_output": miller_output,
    }


def _run_generation(
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    web_search: Optional[Dict[str, Any]],
    evaluation_retrieval: Optional[Dict[str, Any]],
    api_url: str,
    model: str,
    headers: Dict[str, str],
    max_tokens: int,
    validity_mode: str,
    force_miller10_for_search_model: bool = False,
    force_miller10_min_hits: int = 1,
    strict_format_gate: bool = False,
    strict_evidence_gate: bool = False,
    semantic_judge_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = build_user_prompt(snapshot, retrieval=retrieval)
    prompt += _format_web_context_block(web_search)
    prompt += _format_contract_block_v2(snapshot, retrieval if isinstance(retrieval, dict) else evaluation_retrieval)
    if strict_format_gate:
        strict_system_prompt = (
            SYSTEM_PROMPT
            + "\nYou are under STRICT formatting mode."
            + "\nYour output MUST strictly contain the three sections: "
            + "【临床推理】, 【决策干预（Miller）】, and 【决策干预（VitalDB）】."
            + "\nOutput ONLY one QA pair, with exact labels and colons, and no extra text."
        )
    else:
        strict_system_prompt = (
            SYSTEM_PROMPT
            + "\nPrefer using these section labels in the final answer: "
            + "【临床推理】, 【决策干预（Miller）】, and 【决策干预（VitalDB）】."
            + "\nOutput only one final QA pair. Avoid extra preface or postscript."
        )
    if "search" in str(model).lower():
        prompt += (
            "\nFor this model, perform online search if needed before final answer. "
            "Only output the final QA."
        )
        prompt += (
            "\nUse exact section labels in output: "
            "【临床推理】, 【决策干预（Miller）】, 【决策干预（VitalDB）】."
        )
        if force_miller10_for_search_model:
            if strict_evidence_gate:
                prompt += (
                    "\nHard constraint for this route:\n"
                    "- You MUST ground 【决策干预（Miller）】 on Miller's Anesthesia 10th edition evidence only.\n"
                    "- Prefer the provided web snippets that have been filtered for Miller 10th.\n"
                    "- If Miller 10th evidence is insufficient, explicitly state that in 【决策干预（Miller）】 and avoid unsupported claims.\n"
                )
                hit_count = _count_results(web_search)
                if hit_count < max(1, int(force_miller10_min_hits)):
                    prompt += (
                        "\nCurrent Miller10 evidence count is insufficient. "
                        "Do not fabricate textbook claims."
                    )
            else:
                prompt += (
                    "\nSoft constraint for this route:\n"
                    "- Prioritize Miller's Anesthesia 10th edition evidence in 【决策干预（Miller）】.\n"
                    "- If exact Miller10 evidence is limited, provide the most reasonable anesthesia decision and briefly state uncertainty.\n"
                )
    try:
        raw = _post_chat(api_url, headers, model, prompt, max_tokens, system_prompt=strict_system_prompt)
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
    loose_fallback = cleaned if str(cleaned or "").strip() else str(raw or "").strip()
    if strict_format_gate:
        final = cleaned if (_is_strict_qa(cleaned) and _is_action_aligned(cleaned, snapshot) and _has_required_sections_strict_v2(cleaned)) else None
    else:
        final = cleaned if (_has_required_sections_strict_v2(cleaned)) else None
    parse_mode = "strict_pass" if final else "strict_fail"
    if not final:
        repaired = _repair_via_requests_v2(api_url, headers, model, raw, snapshot, max_tokens)
        if repaired:
            repaired_cleaned = _extract_qa_block(repaired)
            if repaired_cleaned and _has_required_sections_strict_v2(repaired_cleaned):
                final = repaired_cleaned
                parse_mode = "repair_pass"
    if not final and loose_fallback:
        final = loose_fallback
        parse_mode = "loose_fallback"

    validity = _evaluate_validity(final, snapshot, validity_mode)
    vitaldb_output = _extract_vitaldb_only(final)
    alignment_eval = _alignment_from_generated_output(snapshot, final)
    support_eval = _concept_support_eval(
        str(validity["miller_output"]),
        _build_evidence_text(evaluation_retrieval if isinstance(evaluation_retrieval, dict) else retrieval),
        semantic_judge_cfg=semantic_judge_cfg,
    )
    return {
        "error": None,
        "raw_output": raw,
        "final_output": final,
        "valid": bool(validity["valid"]),
        "strict_valid": bool(validity["strict_valid"]),
        "miller_valid": bool(validity["miller_valid"]),
        "miller_output": str(validity["miller_output"]),
        "vitaldb_output": vitaldb_output,
        "miller_alignment_eval": alignment_eval,
        "miller10_support_eval": support_eval,
        "miller10_search_enforced": bool(force_miller10_for_search_model),
        "miller10_search_hit_count": _count_results(web_search),
        "strict_format_gate": bool(strict_format_gate),
        "strict_evidence_gate": bool(strict_evidence_gate),
        "parse_mode": parse_mode,
    }


def _compare_modes(rag_out: Dict[str, Any], direct_out: Dict[str, Any]) -> Dict[str, Any]:
    rag_miller = str(rag_out.get("miller_output") or "").strip()
    direct_miller = str(direct_out.get("miller_output") or "").strip()
    rag_score = int(bool(rag_out.get("strict_valid"))) * 2 + int(bool(rag_out.get("miller_valid")))
    direct_score = int(bool(direct_out.get("strict_valid"))) * 2 + int(bool(direct_out.get("miller_valid")))
    if rag_score > direct_score:
        winner = "rag"
    elif direct_score > rag_score:
        winner = "direct"
    else:
        winner = "tie"

    rag_verdict = str((rag_out.get("miller_alignment_eval") or {}).get("verdict", "unavailable")).strip().lower()
    direct_verdict = str((direct_out.get("miller_alignment_eval") or {}).get("verdict", "unavailable")).strip().lower()
    rag_verdict_score = _verdict_score(rag_verdict)
    direct_verdict_score = _verdict_score(direct_verdict)
    if rag_verdict_score > direct_verdict_score:
        verdict_winner = "rag"
    elif direct_verdict_score > rag_verdict_score:
        verdict_winner = "direct"
    else:
        verdict_winner = "tie"

    rag_support = rag_out.get("miller10_support_eval", {}) if isinstance(rag_out.get("miller10_support_eval"), dict) else {}
    direct_support = direct_out.get("miller10_support_eval", {}) if isinstance(direct_out.get("miller10_support_eval"), dict) else {}
    rag_support_score = int(rag_support.get("score", -1))
    direct_support_score = int(direct_support.get("score", -1))
    if rag_support_score > direct_support_score:
        support_winner = "rag"
    elif direct_support_score > rag_support_score:
        support_winner = "direct"
    else:
        support_winner = "tie"

    return {
        "both_valid": bool(rag_out.get("valid")) and bool(direct_out.get("valid")),
        "same_miller_output": bool(rag_miller) and (rag_miller == direct_miller),
        "rag_score_proxy": rag_score,
        "direct_score_proxy": direct_score,
        "winner_by_proxy": winner,
        "rag_miller_alignment_verdict": rag_verdict,
        "direct_miller_alignment_verdict": direct_verdict,
        "rag_miller_alignment_score": rag_verdict_score,
        "direct_miller_alignment_score": direct_verdict_score,
        "winner_by_miller_alignment": verdict_winner,
        "rag_miller10_support_verdict": str(rag_support.get("verdict", "unknown")),
        "direct_miller10_support_verdict": str(direct_support.get("verdict", "unknown")),
        "rag_miller10_support_score": rag_support_score,
        "direct_miller10_support_score": direct_support_score,
        "winner_by_miller10_support": support_winner,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    rag_valid = sum(1 for row in rows if row["gpt_rag_miller10"].get("valid"))
    direct_valid = sum(1 for row in rows if row["gpt_direct_no_rag"].get("valid"))
    rag_strict = sum(1 for row in rows if row["gpt_rag_miller10"].get("strict_valid"))
    direct_strict = sum(1 for row in rows if row["gpt_direct_no_rag"].get("strict_valid"))
    rag_miller = sum(1 for row in rows if row["gpt_rag_miller10"].get("miller_valid"))
    direct_miller = sum(1 for row in rows if row["gpt_direct_no_rag"].get("miller_valid"))
    same_miller = sum(1 for row in rows if row["comparison"].get("same_miller_output"))
    rag_win = sum(1 for row in rows if row["comparison"].get("winner_by_proxy") == "rag")
    direct_win = sum(1 for row in rows if row["comparison"].get("winner_by_proxy") == "direct")
    tie = sum(1 for row in rows if row["comparison"].get("winner_by_proxy") == "tie")
    rag_verdict_counts: Dict[str, int] = {}
    direct_verdict_counts: Dict[str, int] = {}
    rag_support_counts: Dict[str, int] = {}
    direct_support_counts: Dict[str, int] = {}
    for row in rows:
        comp = row.get("comparison", {}) if isinstance(row.get("comparison"), dict) else {}
        rv = str(comp.get("rag_miller_alignment_verdict", "unavailable")).strip().lower()
        dv = str(comp.get("direct_miller_alignment_verdict", "unavailable")).strip().lower()
        rag_verdict_counts[rv] = rag_verdict_counts.get(rv, 0) + 1
        direct_verdict_counts[dv] = direct_verdict_counts.get(dv, 0) + 1
        rs = str(comp.get("rag_miller10_support_verdict", "unknown")).strip().lower()
        ds = str(comp.get("direct_miller10_support_verdict", "unknown")).strip().lower()
        rag_support_counts[rs] = rag_support_counts.get(rs, 0) + 1
        direct_support_counts[ds] = direct_support_counts.get(ds, 0) + 1
    rag_win_alignment = sum(1 for row in rows if row["comparison"].get("winner_by_miller_alignment") == "rag")
    direct_win_alignment = sum(1 for row in rows if row["comparison"].get("winner_by_miller_alignment") == "direct")
    tie_alignment = sum(1 for row in rows if row["comparison"].get("winner_by_miller_alignment") == "tie")
    rag_win_support = sum(1 for row in rows if row["comparison"].get("winner_by_miller10_support") == "rag")
    direct_win_support = sum(1 for row in rows if row["comparison"].get("winner_by_miller10_support") == "direct")
    tie_support = sum(1 for row in rows if row["comparison"].get("winner_by_miller10_support") == "tie")
    return {
        "total": total,
        "gpt_rag_miller10_valid": rag_valid,
        "gpt_direct_no_rag_valid": direct_valid,
        "gpt_rag_miller10_strict_valid": rag_strict,
        "gpt_direct_no_rag_strict_valid": direct_strict,
        "gpt_rag_miller10_miller_valid": rag_miller,
        "gpt_direct_no_rag_miller_valid": direct_miller,
        "same_miller_output": same_miller,
        "rag_win_by_proxy": rag_win,
        "direct_win_by_proxy": direct_win,
        "tie_by_proxy": tie,
        "rag_miller_alignment_verdict_counts": rag_verdict_counts,
        "direct_miller_alignment_verdict_counts": direct_verdict_counts,
        "rag_win_by_miller_alignment": rag_win_alignment,
        "direct_win_by_miller_alignment": direct_win_alignment,
        "tie_by_miller_alignment": tie_alignment,
        "rag_miller10_support_verdict_counts": rag_support_counts,
        "direct_miller10_support_verdict_counts": direct_support_counts,
        "rag_win_by_miller10_support": rag_win_support,
        "direct_win_by_miller10_support": direct_win_support,
        "tie_by_miller10_support": tie_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GPT outputs between RAG (Miller retrieval) and Direct (no retrieval)."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output-jsonl", required=True, help="Output comparison JSONL path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--validity-mode", default="miller_only", choices=["miller_only", "strict"])
    parser.add_argument("--use-existing-retrieval", action="store_true")

    parser.add_argument("--rag-api-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--rag-api-key", default="")
    parser.add_argument("--rag-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--rag-model", required=True, help="Model for RAG route (recommended: local Qwen).")

    parser.add_argument("--direct-api-url", default="https://api2.aigcbest.top/v1/chat/completions")
    parser.add_argument("--direct-api-key", default="")
    parser.add_argument("--direct-api-key-env", default="GPT_API_KEY")
    parser.add_argument("--direct-model", required=True, help="Model for direct route (no retrieval).")
    parser.add_argument(
        "--strict-format-gate",
        action="store_true",
        help="Enable strict format/action gate. Default is relaxed.",
    )
    parser.add_argument(
        "--strict-evidence-gate",
        action="store_true",
        help="Enable strict Miller10 evidence gate. Default is relaxed.",
    )
    parser.add_argument(
        "--enable-semantic-judge",
        action="store_true",
        help="Enable LLM semantic judge to recover synonym/equivalence misses in support scoring.",
    )
    parser.add_argument(
        "--judge-api-url",
        default="",
        help="Semantic judge API URL. If empty and enabled, reuse --direct-api-url.",
    )
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-api-key-env", default="GPT_API_KEY")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-max-tokens", type=int, default=300)
    parser.add_argument(
        "--force-direct-miller10-search",
        action="store_true",
        help="Force direct search model to retrieve Miller's Anesthesia 10th evidence before generation.",
    )
    parser.add_argument(
        "--direct-miller10-query-suffix",
        default="Miller's Anesthesia 10th edition intraoperative anesthesia hemodynamic depth management evidence",
        help="Suffix appended to direct online search query when forcing Miller10 retrieval.",
    )
    parser.add_argument(
        "--direct-miller10-min-hits",
        type=int,
        default=1,
        help="Minimum Miller10 search snippets required before allowing strong Miller claims.",
    )
    parser.add_argument(
        "--direct-miller10-min-score",
        type=int,
        default=3,
        help="Minimum heuristic score for a web snippet to be treated as Miller10 hit.",
    )

    parser.add_argument(
        "--enable-gpt-search-retrieval",
        action="store_true",
        help="Use gpt-5-search-api (or compatible search model) for online retrieval snippets.",
    )
    parser.add_argument("--gpt-search-model", default="gpt-5-search-api")
    parser.add_argument("--gpt-search-api-url", default="", help="If empty, reuse --rag-api-url endpoint.")
    parser.add_argument("--gpt-search-api-key", default="")
    parser.add_argument("--gpt-search-api-key-env", default="GPT_API_KEY")
    parser.add_argument("--gpt-search-top-k", type=int, default=3)
    parser.add_argument("--gpt-search-max-tokens", type=int, default=700)
    parser.add_argument(
        "--gpt-search-target",
        default="direct",
        choices=["rag", "direct", "both"],
        help="Which route receives gpt-search snippets. For local-RAG vs gpt-search comparison, use `direct`.",
    )

    parser.add_argument("--miller-corpus-path", default="")
    parser.add_argument("--miller-index-path", default="")
    parser.add_argument("--miller-top-k", type=int, default=3)
    parser.add_argument("--miller-chunk-chars", type=int, default=1200)
    parser.add_argument("--miller-chunk-overlap-chars", type=int, default=200)
    parser.add_argument("--miller-max-passage-chars", type=int, default=800)
    parser.add_argument("--embedding-backend", default="local", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--enable-web-search", action="store_true")
    parser.add_argument("--web-search-provider", default="tavily", choices=["tavily", "serpapi", "custom"])
    parser.add_argument("--web-search-api-key", default="")
    parser.add_argument("--web-search-api-key-env", default="WEB_SEARCH_API_KEY")
    parser.add_argument("--web-search-url", default="", help="Required when --web-search-provider custom")
    parser.add_argument("--web-search-top-k", type=int, default=3)
    parser.add_argument("--web-search-timeout-sec", type=int, default=20)
    parser.add_argument(
        "--web-search-query-suffix",
        default="Miller's Anesthesia 10th edition intraoperative anesthesia decision evidence",
    )
    parser.add_argument(
        "--web-search-domains",
        default="",
        help="Optional comma-separated domain filters (tavily/custom). Example: elsevier.com,oxfordmedicine.com",
    )
    parser.add_argument(
        "--web-search-on-both-routes",
        action="store_true",
        help="If set, apply web search snippets to both RAG and direct routes. Default is RAG route only.",
    )
    args = parser.parse_args()

    rag_api_key = _resolve_api_key_for_url(args.rag_api_url, args.rag_api_key, args.rag_api_key_env)
    if not rag_api_key:
        raise ValueError("Missing RAG API key. Pass --rag-api-key or set env from --rag-api-key-env.")
    rag_headers = _build_headers(rag_api_key)

    direct_api_key = _resolve_api_key_for_url(args.direct_api_url, args.direct_api_key, args.direct_api_key_env)
    if not direct_api_key:
        raise ValueError("Missing direct API key. Pass --direct-api-key or set env from --direct-api-key-env.")
    direct_headers = _build_headers(direct_api_key)

    gpt_search_api_url = (args.gpt_search_api_url or args.rag_api_url).strip()
    gpt_search_headers: Optional[Dict[str, str]] = None
    if args.enable_gpt_search_retrieval:
        gpt_search_key = _resolve_api_key_for_url(
            gpt_search_api_url,
            args.gpt_search_api_key,
            args.gpt_search_api_key_env,
        )
        if not gpt_search_key:
            raise ValueError(
                "Missing gpt-search API key. Pass --gpt-search-api-key or set env from --gpt-search-api-key-env."
            )
        gpt_search_headers = _build_headers(gpt_search_key)

    semantic_judge_cfg: Dict[str, Any] = {"enabled": False}
    if args.enable_semantic_judge:
        judge_api_url = (args.judge_api_url or args.direct_api_url).strip()
        judge_api_key = _resolve_api_key_for_url(
            judge_api_url,
            args.judge_api_key,
            args.judge_api_key_env,
        )
        if not judge_api_key:
            raise ValueError("Missing semantic judge API key. Pass --judge-api-key or set --judge-api-key-env.")
        semantic_judge_cfg = {
            "enabled": True,
            "api_url": judge_api_url,
            "headers": _build_headers(judge_api_key),
            "model": args.judge_model,
            "max_tokens": max(128, int(args.judge_max_tokens)),
        }

    records = _load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No input records loaded.")

    retriever = None
    embed_client = None
    retrieval_cfg = _build_retrieval_cfg(args)
    if not args.use_existing_retrieval:
        if not args.miller_corpus_path or not args.miller_index_path or not args.embedding_model:
            raise ValueError(
                "For fresh retrieval, provide --miller-corpus-path, --miller-index-path, and --embedding-model."
            )
        embed_client = create_embedding_client(retrieval_cfg)
        retriever = build_miller_retriever(embed_client, retrieval_cfg)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            retrieval = None
            web_search = None
            if args.use_existing_retrieval and isinstance(record.get("miller_retrieval"), dict):
                retrieval = record["miller_retrieval"]
            elif retriever is not None and embed_client is not None:
                retrieval = retrieve_miller_context(snapshot, retriever, embed_client, retrieval_cfg)

            rag_web_search = None
            direct_web_search = None
            if args.enable_gpt_search_retrieval and gpt_search_headers is not None:
                web_search = _search_with_gpt_search_api(
                    snapshot=snapshot,
                    retrieval=None,
                    args=args,
                    api_url=gpt_search_api_url,
                    headers=gpt_search_headers,
                )
                if args.gpt_search_target in {"rag", "both"}:
                    rag_web_search = web_search
                if args.gpt_search_target in {"direct", "both"}:
                    direct_web_search = web_search
            elif args.enable_web_search:
                web_search = _run_web_search(snapshot, retrieval, args)
                rag_web_search = web_search
                if args.web_search_on_both_routes:
                    direct_web_search = web_search

            if args.force_direct_miller10_search and "search" in str(args.direct_model).lower():
                forced_query = _build_web_search_query(snapshot, None, args.direct_miller10_query_suffix)
                direct_web_search = _search_with_gpt_search_api(
                    snapshot=snapshot,
                    retrieval=None,
                    args=args,
                    api_url=args.direct_api_url,
                    headers=direct_headers,
                    query_override=forced_query,
                    search_model=args.direct_model,
                    search_top_k=max(int(args.gpt_search_top_k), int(args.direct_miller10_min_hits)),
                    force_miller10=True,
                    miller10_min_score=int(args.direct_miller10_min_score),
                    strict_evidence_gate=bool(args.strict_evidence_gate),
                )

            rag_out = _run_generation(
                snapshot,
                retrieval,
                rag_web_search,
                retrieval,
                args.rag_api_url,
                args.rag_model,
                rag_headers,
                args.max_tokens,
                args.validity_mode,
                force_miller10_for_search_model=False,
                strict_format_gate=bool(args.strict_format_gate),
                strict_evidence_gate=bool(args.strict_evidence_gate),
                semantic_judge_cfg=semantic_judge_cfg,
            )
            direct_out = _run_generation(
                snapshot,
                None,
                direct_web_search,
                retrieval,
                args.direct_api_url,
                args.direct_model,
                direct_headers,
                args.max_tokens,
                args.validity_mode,
                force_miller10_for_search_model=bool(args.force_direct_miller10_search),
                force_miller10_min_hits=max(1, int(args.direct_miller10_min_hits)),
                strict_format_gate=bool(args.strict_format_gate),
                strict_evidence_gate=bool(args.strict_evidence_gate),
                semantic_judge_cfg=semantic_judge_cfg,
            )
            comparison = _compare_modes(rag_out, direct_out)

            row = {
                "index": idx,
                "caseid": record.get("caseid"),
                "snapshot": snapshot,
                "miller_retrieval": retrieval,
                "web_search_rag": rag_web_search,
                "web_search_direct": direct_web_search,
                "gpt_rag_miller10": rag_out,
                "gpt_direct_no_rag": direct_out,
                "comparison": comparison,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  - compared RAG vs direct {idx}/{len(records)}")

    summary = _summarize(rows)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== GPT RAG vs Direct Summary ===")
    print(f"jsonl:   {output_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
