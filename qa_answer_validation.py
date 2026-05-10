"""Structured answer validation blocks for anesthesia QA generation."""

import re
from typing import Any, Callable, Dict, Optional, Sequence


def _extract_section(text: str, title: str, next_titles: Sequence[str]) -> str:
    norm_text = str(text or "").replace("\u3000", " ")
    norm_title = str(title or "").replace("\u3000", " ")
    start = norm_text.find(norm_title)
    if start < 0:
        alt = re.search(re.escape(norm_title).replace("\\ ", r"\s*"), norm_text)
        if not alt:
            return ""
        start = alt.start()
        title_len = alt.end() - alt.start()
    else:
        title_len = len(norm_title)
    if start < 0:
        return ""
    start = start + title_len
    rest = norm_text[start:]
    end_idx = len(rest)
    for nt in next_titles:
        nnt = str(nt or "").replace("\u3000", " ")
        pos = rest.find(nnt)
        if pos < 0:
            alt = re.search(re.escape(nnt).replace("\\ ", r"\s*"), rest)
            if alt:
                pos = alt.start()
        if pos >= 0:
            end_idx = min(end_idx, pos)
    return rest[:end_idx].strip("：: \n")


def _has_propofol_bolus_like_text(text: str) -> bool:
    t = str(text or "")
    return bool(
        re.search(r"(丙泊酚|propofol)", t, re.IGNORECASE)
        and re.search(r"(推注|bolus|追加|单次|静脉推注)", t, re.IGNORECASE)
    )


def _mentions_hemo_stabilization_first(text: str) -> bool:
    t = str(text or "")
    return bool(
        re.search(
            r"(先升压|优先升压|先纠正灌注|循环稳定后|血压稳定后|MAP恢复后|SBP恢复后|灌注恢复后)",
            t,
            re.IGNORECASE,
        )
    )


def _has_unsafe_bis_target(text: str) -> bool:
    t = str(text or "")
    target_words = r"(?:目标|维持|控制|平稳|调整|降至|回落至)"
    for m in re.finditer(
        rf"BIS[^\n。；;]{{0,30}}?{target_words}[^\n。；;]{{0,20}}?(\d{{2,3}})\s*[-~～至到]\s*(\d{{2,3}})",
        t,
        re.IGNORECASE,
    ):
        try:
            lo = int(m.group(1))
            hi = int(m.group(2))
        except Exception:
            continue
        if max(lo, hi) > 60:
            return True
    for m in re.finditer(
        rf"BIS[^\n。；;]{{0,30}}?{target_words}[^\n。；;]{{0,12}}?(\d{{2,3}})",
        t,
        re.IGNORECASE,
    ):
        try:
            v = int(m.group(1))
        except Exception:
            continue
        if v > 60:
            return True
    return False


def _has_etco2_drop_low_ventilation_mismatch(reasoning_text: str) -> bool:
    t = str(reasoning_text or "")
    mismatch = re.search(
        r"(EtCO2|二氧化碳)[^。\n]{0,40}(骤降|下降|降低|偏低)[^。\n]{0,80}(潮气量不足|低通气|通气不足)",
        t,
        re.IGNORECASE,
    )
    return bool(mismatch)


def validate_structured_answer(
    kind: str,
    text: str,
    include_review: bool,
    snapshot: Optional[Dict[str, Any]],
    vitaldb_logged_action_consistent_fn: Callable[[str, Optional[Dict[str, Any]]], bool],
    metadata_leak_fn: Callable[[str], bool],
    is_hypotension_risk_fn: Callable[[Optional[Dict[str, Any]]], bool],
) -> Dict[str, Any]:
    out = {"valid": True, "reasons": []}
    if not text or not str(text).strip():
        return {"valid": False, "reasons": ["empty_output"]}
    t = str(text).strip()
    sections = ["【临床推理】", "【宏观策略】", "【具体干预】"]
    if include_review:
        sections.append("【复评环节】")
    if kind == "miller":
        sections.append("【原文摘录】")
    for s in sections:
        if s not in t:
            out["valid"] = False
            out["reasons"].append(f"missing_{s}")

    route_keywords = ("静脉推注", "静脉泵注", "静脉滴注", "吸入", "雾化", "肌注", "皮下注", "口服")
    next_titles = []
    if include_review:
        next_titles.append("【复评环节】")
    if kind == "miller":
        next_titles.append("【原文摘录】")
    intervention = _extract_section(t, "【具体干预】", next_titles)
    if not any(k in intervention for k in route_keywords):
        out["valid"] = False
        out["reasons"].append("missing_route")
    if not bool(re.search(r"\d+(?:\.\d+)?\s*(?:mL/h|mL|mg|ug|μg/kg/min|mmHg|bpm|%|vol%|MAC)", intervention, re.IGNORECASE)):
        out["valid"] = False
        out["reasons"].append("missing_quantitative_dose")

    if kind == "vitaldb" and not vitaldb_logged_action_consistent_fn(intervention, snapshot):
        out["valid"] = False
        out["reasons"].append("vitaldb_logged_action_numeric_mismatch")
    if metadata_leak_fn(t):
        out["valid"] = False
        out["reasons"].append("internal_metadata_leak")

    if include_review:
        review = _extract_section(t, "【复评环节】", ["【原文摘录】"])
        if not bool(re.search(r"\d+(?:\.\d+)?\s*(?:s|sec|秒|min|分钟)", review, re.IGNORECASE)):
            out["valid"] = False
            out["reasons"].append("missing_recheck_time")
        if "预期演变" not in review:
            out["valid"] = False
            out["reasons"].append("missing_expected_evolution_field")
        evo_target_pattern = (
            r"(?:回升|下降|维持|恢复|达到|至|到)[^\n。；;]*\d+(?:\.\d+)?\s*"
            r"(?:mmHg|bpm|%|mL/h|mL|mg|ug|μg/kg/min|℃|vol%|MAC)"
            r"|"
            r"\d+(?:\.\d+)?\s*(?:mmHg|bpm|%|mL/h|mL|mg|ug|μg/kg/min|℃|vol%|MAC)[^\n。；;]*"
            r"(?:回升|下降|维持|恢复|达到|至|到)"
        )
        if not bool(re.search(evo_target_pattern, review, re.IGNORECASE)):
            out["valid"] = False
            out["reasons"].append("missing_expected_evolution_target")

    if kind == "miller":
        quote = _extract_section(t, "【原文摘录】", [])
        if not bool(re.search(r"(?i)M10#\d+", quote)):
            out["valid"] = False
            out["reasons"].append("missing_m10_locator")
        if not bool(re.search(r"(?i)\bp\.\s*\d+", quote)):
            out["valid"] = False
            out["reasons"].append("missing_page_locator")

    if is_hypotension_risk_fn(snapshot):
        if _has_propofol_bolus_like_text(intervention) and (not _mentions_hemo_stabilization_first(intervention)):
            out["valid"] = False
            out["reasons"].append("hemodynamic_lock_violation_propofol_under_hypotension")
    if _has_unsafe_bis_target(t):
        out["valid"] = False
        out["reasons"].append("unsafe_bis_target_above_60_in_general_anesthesia")

    reasoning = _extract_section(t, "【临床推理】", ["【宏观策略】", "【具体干预】", "【复评环节】", "【原文摘录】"])
    if _has_etco2_drop_low_ventilation_mismatch(reasoning):
        out["valid"] = False
        out["reasons"].append("etco2_drop_mechanism_mismatch_low_ventilation")
    return out
