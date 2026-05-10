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
    # Only treat explicit "target-setting" language as unsafe target evidence.
    # Avoid false positives from descriptive trend sentences such as "BIS由75降至46".
    target_words = r"(?:目标|维持在|控制在|保持在|设定为|计划维持)"
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


def _has_structured_intervention_block(text: str) -> bool:
    t = str(text or "")
    has_main = bool(re.search(r"(主干预|主处理|主操作)", t))
    has_sync = bool(re.search(r"(同步安全处理|同步处理|并行安全处理)", t))
    return has_main and has_sync


def _extract_main_intervention_block(intervention_text: str) -> str:
    t = str(intervention_text or "")
    m = re.search(
        r"(?:（A）|\(A\)|A[）\)])[\s\S]*?(?:主干预)?[:：]?\s*([\s\S]*?)(?:（B）|\(B\)|B[）\)])",
        t,
        re.IGNORECASE,
    )
    if m:
        return str(m.group(1) or "").strip()
    return t


def _looks_like_cumulative_window_phrase(text: str) -> bool:
    t = str(text or "")
    if re.search(r"(累计量|累积量|液量由)", t):
        return True
    # e.g. "由约33.526 mL调至约33.605 mL" in A-main
    if re.search(
        r"由[^。\n]{0,24}\d+(?:\.\d+)?\s*mL(?!\s*/\s*h)[^。\n]{0,12}(?:调至|至|到|->|→)[^。\n]{0,16}\d+(?:\.\d+)?\s*mL(?!\s*/\s*h)",
        t,
        re.IGNORECASE,
    ):
        return True
    return False


def _has_engineering_style_phrase(text: str) -> bool:
    t = str(text or "")
    return bool(
        re.search(
            r"(根据原始记录时间序列|根据记录时间序列|按原始记录时序|按记录时序|时间序列显示|时间轴显示|轨道数据显示|监测轨道显示|轨迹显示|logged_action|记录显示为|数据提示为|与VitalDB记录一致|原始记录显示|记录到|由记录可见|回顾记录可见|从记录看)",
            t,
            re.IGNORECASE,
        )
    )


def _has_record_replay_phrase(text: str) -> bool:
    t = str(text or "")
    return bool(
        re.search(
            r"(根据原始记录时间序列|根据记录时间序列|按原始记录时序|按记录时序|按时间序列|时间轴显示|轨道数据显示|监测轨道显示|轨迹显示|由记录可见|回顾记录可见|从记录看)",
            t,
            re.IGNORECASE,
        )
    )


def _has_order_label_phrase(text: str) -> bool:
    t = str(text or "")
    return bool(re.search(r"(下达医嘱|执行医嘱)\s*[：:]", t, re.IGNORECASE))


def _has_nested_main_heading_phrase(text: str) -> bool:
    t = str(text or "")
    return bool(re.search(r"(^|[\s，。；;])主干预\s*[：:]", t, re.IGNORECASE))


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

    route_keywords = (
        "静脉推注",
        "静脉泵注",
        "静脉滴注",
        "静脉持续输注",
        "持续输注",
        "静脉给药",
        "静滴",
        "静注",
        "泵注",
        "泵入",
        "吸入",
        "雾化",
        "肌注",
        "皮下注",
        "口服",
        "IV",
        "iv",
    )
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
    if not _has_structured_intervention_block(intervention):
        out["valid"] = False
        out["reasons"].append("missing_structured_intervention_block")
    a_main = _extract_main_intervention_block(intervention)
    if _looks_like_cumulative_window_phrase(a_main):
        out["valid"] = False
        out["reasons"].append("a_main_contains_cumulative_volume")
    if _has_record_replay_phrase(a_main):
        out["valid"] = False
        out["reasons"].append("a_main_contains_record_replay_phrase")
    if _has_order_label_phrase(a_main):
        out["valid"] = False
        out["reasons"].append("a_main_contains_order_label_phrase")
    if _has_nested_main_heading_phrase(a_main):
        out["valid"] = False
        out["reasons"].append("a_main_repeats_heading_phrase")
    if _looks_like_cumulative_window_phrase(intervention):
        out["valid"] = False
        out["reasons"].append("intervention_contains_cumulative_volume")
    if _has_engineering_style_phrase(intervention):
        out["valid"] = False
        out["reasons"].append("non_clinical_engineering_phrase")

    if kind == "vitaldb" and not vitaldb_logged_action_consistent_fn(intervention, snapshot):
        out["valid"] = False
        out["reasons"].append("vitaldb_logged_action_numeric_mismatch")
    if metadata_leak_fn(t):
        out["valid"] = False
        out["reasons"].append("internal_metadata_leak")

    if include_review:
        review = _extract_section(t, "【复评环节】", ["【原文摘录】"])
        if _looks_like_cumulative_window_phrase(review):
            out["valid"] = False
            out["reasons"].append("review_contains_cumulative_volume")
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
            r"|"
            r"(?:SpO2|EtCO2|HR|MAP|SBP|DBP|BIS)[^\n。；;]{0,40}(?:≥|<=|≤|>|<|约|至|到)\s*"
            r"\d+(?:\.\d+)?\s*(?:mmHg|bpm|%|mL/h|mL|mg|ug|μg/kg/min|℃|vol%|MAC)"
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
