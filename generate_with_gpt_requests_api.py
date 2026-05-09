import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import requests

from anes_pipeline import (
    _expected_action_unit,
    _golden_action_hint,
    build_miller_retriever,
    create_embedding_client,
    MEDICATION_DISPLAY,
    retrieve_miller_context,
)


RISK_LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}

CN_TERM_MAP = {
    "M": "男性",
    "F": "女性",
    "Male": "男性",
    "Female": "女性",
    "General surgery": "普外科",
    "Thoracic Surgery": "胸外科",
    "Thoracic surgery": "胸外科",
    "Cardiac Surgery": "心脏外科",
    "Neurosurgery": "神经外科",
    "Intraoperative": "术中",
    "relative timestamp": "相对时间",
    "Unknown": "暂缺",
    "Unknown surgery": "手术名称暂缺",
    "Advanced gastric cancer": "进展期胃癌",
    "Aortic aneurysm": "主动脉瘤",
    "Aortic aneurys": "主动脉瘤",
    "Hepatocellular carcinoma": "肝细胞癌",
    "Hepatic": "肝脏",
    "Liver": "肝脏",
    "Stomach": "胃",
    "Vascular": "血管外科",
    "Subtotal gastrectomy": "胃次全切除术",
    "Liver segmentectomy": "肝段切除术",
    "Aneurysmal repair": "动脉瘤修补术",
    "Lobectomy": "肺叶切除术",
    "VATS": "胸腔镜手术",
}

SURGERY_CN_MAP = {
    "Subtotal gastrectomy": "胃次全切除术",
    "Liver segmentectomy": "肝段切除术",
    "Aneurysmal repair": "动脉瘤修补术",
    "Lobectomy": "肺叶切除术",
    "Video-assisted thoracoscopic surgery": "胸腔镜手术",
}


def _sanitize_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    return obj


def _safe_json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(_sanitize_for_json(obj), ensure_ascii=False, allow_nan=False)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _split_csv_tokens(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw or "").split(","):
        cleaned = str(token).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _normalize_alarm_tag(tag: str) -> str:
    t = str(tag or "").strip()
    if not t:
        return ""
    tu = t.upper()
    alias = {
        "ETCO2": "ETCO2",
        "SPO2": "SPO2",
        "ECG": "ECG",
        "HR": "HR",
        "SBP": "SBP",
        "DBP": "DBP",
        "MAP": "MAP",
        "MBP": "MAP",
        "BT": "BT",
        "BIS": "BIS",
        "RSO2": "rSO2",
        "R-SO2": "rSO2",
        "CO": "CO",
        "CI": "CI",
        "SV": "SV",
        "SVV": "SVV",
        "PPV": "PPV",
        "CVP": "CVP",
        "SVR": "SVR",
        "ABG": "ABG",
        "TEG": "TEG",
        "ACT": "ACT",
        "URINE OUTPUT": "Urine Output",
        "BLOOD LOSS": "Blood Loss",
    }
    return alias.get(tu, t)


def _infer_adverse_event_types_from_flags(flags: Sequence[str]) -> List[str]:
    out: List[str] = []
    text = " | ".join(str(x) for x in flags if str(x).strip()).lower()
    if not text:
        return out

    def _add(tag: str) -> None:
        if tag not in out:
            out.append(tag)

    if ("大出血" in text) or ("失血量" in text and "1000" in text):
        _add("major_bleeding")
    elif "出血偏多" in text:
        _add("bleeding_warning")

    if ("无尿" in text) or ("危重少尿" in text):
        _add("anuria_critical")
    elif "尿量偏低" in text:
        _add("oliguria_warning")

    if "高钾" in text:
        _add("hyperkalemia_critical")
    if "低钾" in text:
        _add("hypokalemia_critical")

    if "高血糖明显异常" in text:
        _add("hyperglycemia_severe")
    elif "高血糖预警" in text:
        _add("hyperglycemia_warning")

    if ("恶性心律失常" in text) or ("af with rvr" in text) or ("svta" in text) or ("psvt" in text):
        _add("malignant_arrhythmia")
    elif "心律异常事件" in text:
        _add("arrhythmia_event")

    if "休克/低灌注模式" in text:
        _add("shock_pattern")
    if "疑似过敏" in text:
        _add("suspected_anaphylaxis_pattern")
    if "过敏史" in text:
        _add("allergy_history")

    if "abg低氧血症风险" in text:
        _add("abg_hypoxemia")
    if ("abg二氧化碳潴留风险" in text) or ("abg通气不足预警" in text):
        _add("abg_hypercapnia")
    if "abg酸中毒+高乳酸风险" in text:
        _add("abg_metabolic_acidosis_hyperlactatemia")
    elif ("abg酸中毒预警" in text) or ("abg乳酸升高预警" in text):
        _add("abg_metabolic_acidosis_warning")
    if "abg碱剩余显著负值" in text:
        _add("abg_be_negative_large")

    if "teg低凝风险" in text or "teg低凝/血小板功能不足风险" in text:
        _add("coagulation_low")
    if "teg高凝风险" in text:
        _add("coagulation_high")
    if "act异常" in text:
        _add("act_abnormal")

    return out


def _snapshot_meta(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    risk_level = str(assess.get("risk_level") or "low").strip().lower()
    if risk_level not in RISK_LEVEL_ORDER:
        risk_level = "low"
    sample_category = str(
        snapshot.get("sample_category")
        or assess.get("sample_category")
        or ("critical_alarm" if risk_level == "high" else "warning_signal" if risk_level == "moderate" else "stable_maintenance")
    ).strip()
    risk_flags = assess.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    adverse_event_flags = assess.get("adverse_event_flags", [])
    if not isinstance(adverse_event_flags, list):
        adverse_event_flags = []
    adverse_event_types = assess.get("adverse_event_types", [])
    if not isinstance(adverse_event_types, list):
        adverse_event_types = []
    if not adverse_event_types:
        merged_flags: List[str] = []
        merged_flags.extend([str(x) for x in adverse_event_flags if str(x).strip()])
        merged_flags.extend([str(x) for x in risk_flags if str(x).strip()])
        if merged_flags:
            adverse_event_types = _infer_adverse_event_types_from_flags(merged_flags)
    return {
        "risk_level": risk_level,
        "sample_category": sample_category,
        "risk_flags": risk_flags,
        "adverse_event_flags": adverse_event_flags,
        "adverse_event_types": adverse_event_types,
        "baseline_comparison": assess.get("baseline_comparison", {}) if isinstance(assess.get("baseline_comparison"), dict) else {},
        "recent_state_mean": assess.get("recent_state_mean", {}) if isinstance(assess.get("recent_state_mean"), dict) else {},
        "persistence_seconds": assess.get("persistence_seconds", {}) if isinstance(assess.get("persistence_seconds"), dict) else {},
        "sensitivity_policy": assess.get("sensitivity_policy", {}) if isinstance(assess.get("sensitivity_policy"), dict) else {},
    }


def _collect_indicator_alerts(
    snapshot: Dict[str, Any],
    min_hr_relative_change_pct: float = 20.0,
    include_event_flags: bool = True,
) -> List[Dict[str, Any]]:
    meta = _snapshot_meta(snapshot)
    recent = meta["recent_state_mean"]
    baseline = meta["baseline_comparison"]
    persist = meta["persistence_seconds"]
    sensitivity = meta["sensitivity_policy"] if isinstance(meta.get("sensitivity_policy"), dict) else {}
    personalized = sensitivity.get("personalized_thresholds", {}) if isinstance(sensitivity.get("personalized_thresholds"), dict) else {}

    map_now = _to_float(recent.get("MAP_mmhg"))
    sbp_now = _to_float(recent.get("SBP_mmhg"))
    dbp_now = _to_float(recent.get("DBP_mmhg"))
    hr_now = _to_float(recent.get("HR_bpm"))
    spo2_now = _to_float(recent.get("SpO2_pct"))
    bis_now = _to_float(recent.get("BIS"))
    etco2_now = _to_float(recent.get("EtCO2_mmhg"))
    co_now = _to_float(recent.get("CO_L_min"))
    if co_now is None:
        co_now = _to_float(recent.get("CO"))
    ci_now = _to_float(recent.get("CI_L_min_m2"))
    if ci_now is None:
        ci_now = _to_float(recent.get("CI"))
    sv_now = _to_float(recent.get("SV_ml"))
    if sv_now is None:
        sv_now = _to_float(recent.get("SV"))
    svv_now = _to_float(recent.get("SVV_pct"))
    if svv_now is None:
        svv_now = _to_float(recent.get("SVV"))
    ppv_now = _to_float(recent.get("PPV_pct"))
    if ppv_now is None:
        ppv_now = _to_float(recent.get("PPV"))
    cvp_now = _to_float(recent.get("CVP_mmhg"))
    if cvp_now is None:
        cvp_now = _to_float(recent.get("CVP"))
    svr_now = _to_float(recent.get("SVR_dyns_cm5"))
    if svr_now is None:
        svr_now = _to_float(recent.get("SVR"))
    bt_now = _to_float(recent.get("BT_c"))
    rso2_l_now = _to_float(recent.get("rSO2_L_pct"))
    rso2_r_now = _to_float(recent.get("rSO2_R_pct"))

    map_drop_pct = _to_float(baseline.get("MAP_drop_from_baseline_pct"))
    sbp_change_pct = _to_float(baseline.get("SBP_change_from_baseline_pct"))
    dbp_change_pct = _to_float(baseline.get("DBP_change_from_baseline_pct"))
    hr_change_pct = _to_float(baseline.get("HR_change_from_baseline_pct"))
    spo2_drop_pct = _to_float(baseline.get("SpO2_drop_from_baseline_pct"))
    rso2_l_drop_pct = _to_float(baseline.get("rSO2_L_drop_from_baseline_pct"))
    rso2_r_drop_pct = _to_float(baseline.get("rSO2_R_drop_from_baseline_pct"))

    map_lt_65 = _to_float(persist.get("map_lt_65")) or 0.0
    map_lt_55 = _to_float(persist.get("map_lt_55")) or 0.0
    sbp_lt_90 = _to_float(persist.get("sbp_lt_90")) or 0.0
    sbp_gt_180 = _to_float(persist.get("sbp_gt_180")) or 0.0
    dbp_lt_60 = _to_float(persist.get("dbp_lt_60")) or 0.0
    dbp_gt_100 = _to_float(persist.get("dbp_gt_100")) or 0.0
    hr_gt_100 = _to_float(persist.get("hr_gt_100")) or 0.0
    hr_lt_50 = _to_float(persist.get("hr_lt_50")) or 0.0
    bis_gt_60 = _to_float(persist.get("bis_gt_60")) or 0.0
    bis_lt_40 = _to_float(persist.get("bis_lt_40")) or 0.0
    spo2_lt_94 = _to_float(persist.get("spo2_lt_94")) or 0.0
    spo2_lt_90 = _to_float(persist.get("spo2_lt_90")) or 0.0
    spo2_le_attention = _to_float(persist.get("spo2_le_attention")) or 0.0
    etco2_missing = _to_float(persist.get("etco2_missing")) or 0.0
    etco2_zero_like = _to_float(persist.get("etco2_zero_like")) or 0.0
    bt_lt_36 = _to_float(persist.get("bt_lt_36")) or 0.0
    bt_ge_38 = _to_float(persist.get("bt_ge_38")) or 0.0
    rso2_l_lt_55 = _to_float(persist.get("rso2_l_lt_55")) or 0.0
    rso2_r_lt_55 = _to_float(persist.get("rso2_r_lt_55")) or 0.0

    hr_relative_limit = _to_float(personalized.get("hr_relative_change_pct"))
    if hr_relative_limit is None:
        hr_relative_limit = float(min_hr_relative_change_pct)
    map_low_limit = _to_float(personalized.get("map_low_mmhg"))
    if map_low_limit is None:
        map_low_limit = 65.0
    hr_tachy_limit = _to_float(personalized.get("hr_tachycardia_bpm"))
    if hr_tachy_limit is None:
        hr_tachy_limit = 100.0
    hr_brady_limit = _to_float(personalized.get("hr_bradycardia_bpm"))
    if hr_brady_limit is None:
        hr_brady_limit = 50.0
    spo2_attention_limit = _to_float(personalized.get("spo2_attention_pct"))
    if spo2_attention_limit is None:
        spo2_attention_limit = 95.0
    spo2_drop_limit = _to_float(personalized.get("spo2_drop_from_baseline_pct"))
    if spo2_drop_limit is None:
        spo2_drop_limit = 4.0
    spo2_attention_persist_limit = _to_float(personalized.get("spo2_attention_persist_sec"))
    if spo2_attention_persist_limit is None:
        spo2_attention_persist_limit = 20.0
    etco2_missing_alert_sec = 2.0
    svv_high_limit = _to_float(personalized.get("svv_high_pct"))
    if svv_high_limit is None:
        svv_high_limit = _to_float(personalized.get("svv_high"))
    if svv_high_limit is None:
        svv_high_limit = 13.0
    ppv_high_limit = _to_float(personalized.get("ppv_high_pct"))
    if ppv_high_limit is None:
        ppv_high_limit = 13.0
    cvp_low_limit = _to_float(personalized.get("cvp_low_mmhg"))
    if cvp_low_limit is None:
        cvp_low_limit = _to_float(personalized.get("cvp_low"))
    if cvp_low_limit is None:
        cvp_low_limit = 2.0
    cvp_high_limit = _to_float(personalized.get("cvp_high_mmhg"))
    if cvp_high_limit is None:
        cvp_high_limit = _to_float(personalized.get("cvp_high"))
    if cvp_high_limit is None:
        cvp_high_limit = 12.0
    co_low_limit = _to_float(personalized.get("co_low"))
    if co_low_limit is None:
        co_low_limit = 4.0
    co_high_limit = _to_float(personalized.get("co_high"))
    if co_high_limit is None:
        co_high_limit = 8.0
    ci_low_limit = _to_float(personalized.get("ci_low"))
    if ci_low_limit is None:
        ci_low_limit = 2.5
    ci_high_limit = _to_float(personalized.get("ci_high"))
    if ci_high_limit is None:
        ci_high_limit = 4.0
    sv_low_limit = _to_float(personalized.get("sv_low"))
    if sv_low_limit is None:
        sv_low_limit = 60.0
    sv_high_limit = _to_float(personalized.get("sv_high"))
    if sv_high_limit is None:
        sv_high_limit = 100.0
    svr_low_limit = _to_float(personalized.get("svr_low"))
    if svr_low_limit is None:
        svr_low_limit = 800.0
    svr_high_limit = _to_float(personalized.get("svr_high"))
    if svr_high_limit is None:
        svr_high_limit = 1600.0
    ci_lt_low = _to_float(persist.get("ci_lt_low")) or 0.0
    co_lt_low = _to_float(persist.get("co_lt_low")) or 0.0
    sv_lt_low = _to_float(persist.get("sv_lt_low")) or 0.0
    ppv_ge_13 = _to_float(persist.get("ppv_ge_13")) or 0.0
    svr_lt_low = _to_float(persist.get("svr_lt_low")) or 0.0

    alerts: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(
        rule: str,
        indicator: str,
        severity: str,
        current: Optional[float] = None,
        threshold: str = "",
        persistence_sec: Optional[float] = None,
        note: str = "",
    ) -> None:
        key = f"{indicator}|{rule}"
        if key in seen:
            return
        seen.add(key)
        item: Dict[str, Any] = {"indicator": indicator, "rule": rule, "severity": severity}
        if current is not None:
            item["current"] = round(float(current), 4)
        if threshold:
            item["threshold"] = threshold
        if persistence_sec is not None:
            item["persistence_sec"] = round(float(persistence_sec), 2)
        if note:
            item["note"] = note
        alerts.append(item)

    if include_event_flags:
        for flag in meta.get("adverse_event_flags", []):
            txt = str(flag).strip()
            if txt:
                _add("adverse_event_flag", "event", "critical", note=txt)

    if map_now is not None and map_now < float(map_low_limit):
        _add("map_below_floor", "MAP", "critical", current=map_now, threshold=f"<{map_low_limit:.1f} mmHg")
    if map_lt_55 >= 30.0:
        _add("map_lt_55_persistent", "MAP", "critical", current=map_now, threshold="<55 mmHg", persistence_sec=map_lt_55)
    if map_lt_65 >= 60.0:
        _add("map_lt_65_persistent", "MAP", "warning", current=map_now, threshold="<65 mmHg", persistence_sec=map_lt_65)
    if map_drop_pct is not None and map_drop_pct >= 20.0:
        _add("map_relative_drop", "MAP", "warning", current=map_drop_pct, threshold="drop >=20%")

    if sbp_now is not None and (sbp_now < 90.0 or sbp_now > 180.0):
        _add("sbp_out_of_range", "SBP", "warning", current=sbp_now, threshold="90-180 mmHg")
    if dbp_now is not None and (dbp_now < 60.0 or dbp_now > 100.0):
        _add("dbp_out_of_range", "DBP", "warning", current=dbp_now, threshold="60-100 mmHg")
    if sbp_lt_90 >= 60.0:
        _add("sbp_lt_90_persistent", "SBP", "warning", current=sbp_now, threshold="<90 mmHg", persistence_sec=sbp_lt_90)
    if sbp_gt_180 >= 60.0:
        _add("sbp_gt_180_persistent", "SBP", "warning", current=sbp_now, threshold=">180 mmHg", persistence_sec=sbp_gt_180)
    if dbp_lt_60 >= 60.0:
        _add("dbp_lt_60_persistent", "DBP", "warning", current=dbp_now, threshold="<60 mmHg", persistence_sec=dbp_lt_60)
    if dbp_gt_100 >= 60.0:
        _add("dbp_gt_100_persistent", "DBP", "warning", current=dbp_now, threshold=">100 mmHg", persistence_sec=dbp_gt_100)
    if sbp_change_pct is not None and abs(sbp_change_pct) >= 30.0:
        _add("sbp_relative_change", "SBP", "warning", current=sbp_change_pct, threshold="|change| >=30%")
    if dbp_change_pct is not None and abs(dbp_change_pct) >= 30.0:
        _add("dbp_relative_change", "DBP", "warning", current=dbp_change_pct, threshold="|change| >=30%")

    if hr_now is not None and (hr_now < float(hr_brady_limit) or hr_now > float(hr_tachy_limit)):
        _add("hr_out_of_range", "HR", "warning", current=hr_now, threshold=f"{hr_brady_limit:.0f}-{hr_tachy_limit:.0f} bpm")
    if hr_gt_100 >= 60.0:
        _add("hr_gt_100_persistent", "HR", "warning", current=hr_now, threshold=">100 bpm", persistence_sec=hr_gt_100)
    if hr_lt_50 >= 60.0:
        _add("hr_lt_50_persistent", "HR", "warning", current=hr_now, threshold="<50 bpm", persistence_sec=hr_lt_50)
    if hr_change_pct is not None and abs(hr_change_pct) >= float(hr_relative_limit):
        _add("hr_relative_change", "HR", "warning", current=hr_change_pct, threshold=f"|change| >={hr_relative_limit:.0f}%")

    if svv_now is not None and svv_now >= float(svv_high_limit):
        _add("svv_high", "SVV", "warning", current=svv_now, threshold=f">={svv_high_limit:.1f}%")
    if ppv_now is not None and ppv_now >= float(ppv_high_limit):
        _add("ppv_high", "PPV", "warning", current=ppv_now, threshold=f">={ppv_high_limit:.1f}%")
    if cvp_now is not None and cvp_now <= float(cvp_low_limit):
        _add("cvp_low", "CVP", "warning", current=cvp_now, threshold=f"<={cvp_low_limit:.1f} mmHg")
    if cvp_now is not None and cvp_now >= float(cvp_high_limit):
        _add("cvp_high", "CVP", "warning", current=cvp_now, threshold=f">={cvp_high_limit:.1f} mmHg")
    if co_now is not None and co_now < float(co_low_limit):
        _add("co_low", "CO", "warning", current=co_now, threshold=f"<{co_low_limit:.1f} L/min")
    if co_now is not None and co_now > float(co_high_limit):
        _add("co_high", "CO", "warning", current=co_now, threshold=f">{co_high_limit:.1f} L/min")
    if ci_now is not None and ci_now < float(ci_low_limit):
        _add("ci_low", "CI", "warning", current=ci_now, threshold=f"<{ci_low_limit:.1f} L/(min·m²)")
    if ci_now is not None and ci_now > float(ci_high_limit):
        _add("ci_high", "CI", "warning", current=ci_now, threshold=f">{ci_high_limit:.1f} L/(min·m²)")
    if sv_now is not None and sv_now < float(sv_low_limit):
        _add("sv_low", "SV", "warning", current=sv_now, threshold=f"<{sv_low_limit:.0f} mL")
    if sv_now is not None and sv_now > float(sv_high_limit):
        _add("sv_high", "SV", "warning", current=sv_now, threshold=f">{sv_high_limit:.0f} mL")
    if svr_now is not None and svr_now < float(svr_low_limit):
        _add("svr_low", "SVR", "warning", current=svr_now, threshold=f"<{svr_low_limit:.0f} dyn·s·cm⁻5")
    if svr_now is not None and svr_now > float(svr_high_limit):
        _add("svr_high", "SVR", "warning", current=svr_now, threshold=f">{svr_high_limit:.0f} dyn·s·cm⁻5")
    if ci_lt_low >= 60.0:
        _add("ci_lt_low_persistent", "CI", "warning", current=ci_now, threshold=f"<{ci_low_limit:.1f} L/(min·m²)", persistence_sec=ci_lt_low)
    if co_lt_low >= 60.0:
        _add("co_lt_low_persistent", "CO", "warning", current=co_now, threshold=f"<{co_low_limit:.1f} L/min", persistence_sec=co_lt_low)
    if sv_lt_low >= 60.0:
        _add("sv_lt_low_persistent", "SV", "warning", current=sv_now, threshold=f"<{sv_low_limit:.0f} mL", persistence_sec=sv_lt_low)
    if ppv_ge_13 >= 60.0:
        _add("ppv_high_persistent", "PPV", "warning", current=ppv_now, threshold=f">={ppv_high_limit:.1f}%", persistence_sec=ppv_ge_13)
    if svr_lt_low >= 60.0:
        _add("svr_lt_low_persistent", "SVR", "warning", current=svr_now, threshold=f"<{svr_low_limit:.0f} dyn·s·cm⁻5", persistence_sec=svr_lt_low)

    if spo2_now is not None and spo2_now < 90.0:
        _add("spo2_below_90", "SpO2", "critical", current=spo2_now, threshold="<90%")
    if spo2_now is not None and spo2_now < 94.0:
        _add("spo2_below_94", "SpO2", "warning", current=spo2_now, threshold="<94%")
    if spo2_lt_90 >= 30.0:
        _add("spo2_lt_90_persistent", "SpO2", "critical", current=spo2_now, threshold="<90%", persistence_sec=spo2_lt_90)
    if spo2_lt_94 >= 60.0:
        _add("spo2_lt_94_persistent", "SpO2", "warning", current=spo2_now, threshold="<94%", persistence_sec=spo2_lt_94)
    if spo2_le_attention >= spo2_attention_persist_limit:
        _add(
            "spo2_attention_persistent",
            "SpO2",
            "warning",
            current=spo2_now,
            threshold=f"<={spo2_attention_limit:.1f}%",
            persistence_sec=spo2_le_attention,
        )
    if spo2_drop_pct is not None and spo2_drop_pct >= spo2_drop_limit and spo2_now is not None and spo2_now <= spo2_attention_limit:
        _add(
            "spo2_drop_from_baseline",
            "SpO2",
            "warning",
            current=spo2_drop_pct,
            threshold=f"drop >={spo2_drop_limit:.1f}% and SpO2 <={spo2_attention_limit:.1f}%",
        )

    if bool(sensitivity.get("etco2_missing_triggered")) and (not bool(sensitivity.get("etco2_zeroing_suspected"))):
        _add("etco2_missing_triggered", "EtCO2", "critical", threshold=f"missing >={etco2_missing_alert_sec:.0f}s")
    if etco2_missing >= etco2_missing_alert_sec and etco2_zero_like < 6.0:
        _add(
            "etco2_missing_persistent",
            "EtCO2",
            "critical",
            current=etco2_now,
            threshold=f"missing >={etco2_missing_alert_sec:.0f}s",
            persistence_sec=etco2_missing,
        )
    if etco2_now is not None and etco2_now < 30.0:
        _add("etco2_low", "EtCO2", "warning", current=etco2_now, threshold="<30 mmHg")
    if etco2_now is not None and etco2_now > 50.0:
        _add("etco2_high", "EtCO2", "warning", current=etco2_now, threshold=">50 mmHg")

    bis_high_now = bis_now is not None and bis_now > 60.0
    bis_low_now = bis_now is not None and bis_now < 40.0
    if bis_high_now and bis_gt_60 >= 120.0:
        _add("bis_high_persistent", "BIS", "warning", current=bis_now, threshold=">60", persistence_sec=bis_gt_60)
    if bis_low_now and bis_lt_40 >= 120.0:
        _add("bis_low_persistent", "BIS", "warning", current=bis_now, threshold="<40", persistence_sec=bis_lt_40)
    if (bis_high_now or bis_low_now) and (
        (map_now is not None and map_now < float(map_low_limit))
        or (hr_now is not None and (hr_now < float(hr_brady_limit) or hr_now > float(hr_tachy_limit)))
        or (spo2_now is not None and spo2_now < 94.0)
        or (etco2_now is not None and (etco2_now < 30.0 or etco2_now > 50.0))
    ):
        _add("bis_with_hemo_oxy_abnormal", "BIS", "critical", current=bis_now, threshold="BIS abnormal + hemo/oxy abnormal")

    if bt_now is not None and (bt_now < 36.0 or bt_now >= 38.0):
        _add("temperature_out_of_range", "BT", "warning", current=bt_now, threshold="<36.0 or >=38.0℃")
    if bt_lt_36 >= 60.0 or bt_ge_38 >= 30.0:
        _add(
            "temperature_persistent",
            "BT",
            "warning",
            current=bt_now,
            threshold="<36.0℃ or >=38.0℃",
            persistence_sec=max(bt_lt_36, bt_ge_38),
        )

    rso2_now_vals = [x for x in [rso2_l_now, rso2_r_now] if x is not None]
    if rso2_now_vals and min(rso2_now_vals) < 55.0:
        _add("rso2_absolute_low", "rSO2", "warning", current=min(rso2_now_vals), threshold="<55%")
    if rso2_l_lt_55 >= 60.0 or rso2_r_lt_55 >= 60.0:
        _add(
            "rso2_persistent_low",
            "rSO2",
            "warning",
            current=min([x for x in [rso2_l_now, rso2_r_now] if x is not None] or [0.0]),
            threshold="<55%",
            persistence_sec=max(rso2_l_lt_55, rso2_r_lt_55),
        )
    rso2_drop_vals = [x for x in [rso2_l_drop_pct, rso2_r_drop_pct] if x is not None]
    if rso2_drop_vals and max(rso2_drop_vals) >= float(personalized.get("rso2_drop_from_baseline_pct", 20.0)):
        _add("rso2_relative_drop", "rSO2", "warning", current=max(rso2_drop_vals), threshold="drop >=20%")
    return alerts


def _has_objective_alert(
    snapshot: Dict[str, Any],
    min_hr_relative_change_pct: float = 20.0,
    min_objective_alert_count: int = 1,
    objective_alert_critical_only: bool = False,
) -> bool:
    alerts = _collect_indicator_alerts(
        snapshot,
        min_hr_relative_change_pct=min_hr_relative_change_pct,
        include_event_flags=False,
    )
    # BIS is supportive only: isolated BIS deviation should not pass objective-alert gating.
    # Keep BIS only when it is coupled with hemo/oxygen abnormalities.
    alerts = [
        x
        for x in alerts
        if not (
            str(x.get("indicator", "")).upper() == "BIS"
            and str(x.get("rule", "")).strip() != "bis_with_hemo_oxy_abnormal"
        )
    ]
    if objective_alert_critical_only:
        alerts = [x for x in alerts if str(x.get("severity", "")).lower() == "critical"]
    return len(alerts) >= max(1, int(min_objective_alert_count))


def _collect_present_alarm_tags(snapshot: Dict[str, Any], min_hr_relative_change_pct: float = 20.0) -> set[str]:
    alerts = _collect_indicator_alerts(
        snapshot,
        min_hr_relative_change_pct=min_hr_relative_change_pct,
        include_event_flags=False,
    )
    tags: set[str] = set()
    for a in alerts:
        indicator = str(a.get("indicator", "")).strip().upper()
        if not indicator:
            continue
        if indicator == "EVENT":
            continue
        if indicator == "RSO2":
            tags.add("rSO2")
            continue
        tags.add(_normalize_alarm_tag(indicator))

    meta = _snapshot_meta(snapshot)
    event_types = {str(x).strip().lower() for x in meta.get("adverse_event_types", []) if str(x).strip()}
    if {"malignant_arrhythmia", "arrhythmia_event"} & event_types:
        tags.add("ECG")
    if any(x.startswith("abg_") for x in event_types):
        tags.add("ABG")
    if {"coagulation_low", "coagulation_high"} & event_types:
        tags.add("TEG")
    if "act_abnormal" in event_types:
        tags.add("ACT")
    if {"anuria_critical", "oliguria_warning"} & event_types:
        tags.add("Urine Output")
    if {"major_bleeding", "bleeding_warning"} & event_types:
        tags.add("Blood Loss")
    return tags


def _filter_records_by_snapshot_policy(
    records: Sequence[Dict[str, Any]],
    allowed_categories: Sequence[str],
    min_risk_level: str,
    require_objective_alert: bool,
    min_hr_relative_change_pct: float,
    min_objective_alert_count: int,
    objective_alert_critical_only: bool,
    required_alarm_tags: Sequence[str],
    alarm_tag_match_mode: str,
    required_adverse_event_types: Sequence[str],
    adverse_event_match_mode: str,
) -> List[Dict[str, Any]]:
    allowed = {str(x).strip() for x in allowed_categories if str(x).strip()}
    required_event_types = {
        str(x).strip().lower() for x in required_adverse_event_types if str(x).strip()
    }
    required_alarm_tags_set = {
        _normalize_alarm_tag(str(x).strip()) for x in required_alarm_tags if str(x).strip()
    }
    match_mode = str(adverse_event_match_mode or "any").strip().lower()
    if match_mode not in {"any", "all"}:
        match_mode = "any"
    alarm_match_mode = str(alarm_tag_match_mode or "any").strip().lower()
    if alarm_match_mode not in {"any", "all"}:
        alarm_match_mode = "any"
    out: List[Dict[str, Any]] = []
    min_risk = str(min_risk_level or "low").strip().lower()
    if min_risk not in RISK_LEVEL_ORDER:
        min_risk = "low"
    min_rank = RISK_LEVEL_ORDER[min_risk]
    for rec in records:
        try:
            snap = _snapshot_from_record(rec)
        except Exception:
            continue
        meta = _snapshot_meta(snap)
        if allowed and meta["sample_category"] not in allowed:
            continue
        if RISK_LEVEL_ORDER.get(meta["risk_level"], 0) < min_rank:
            continue
        if required_event_types:
            present = {str(x).strip().lower() for x in meta.get("adverse_event_types", []) if str(x).strip()}
            if match_mode == "all":
                if not required_event_types.issubset(present):
                    continue
            else:
                if present.isdisjoint(required_event_types):
                    continue
        if require_objective_alert and (
            not _has_objective_alert(
                snap,
                min_hr_relative_change_pct=min_hr_relative_change_pct,
                min_objective_alert_count=min_objective_alert_count,
                objective_alert_critical_only=objective_alert_critical_only,
            )
        ):
            continue
        if required_alarm_tags_set:
            present_tags = _collect_present_alarm_tags(
                snap, min_hr_relative_change_pct=min_hr_relative_change_pct
            )
            if alarm_match_mode == "all":
                if not required_alarm_tags_set.issubset(present_tags):
                    continue
            else:
                if present_tags.isdisjoint(required_alarm_tags_set):
                    continue
        out.append(rec)
    return out


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
        miller_bis_intent_mode=str(args.miller_bis_intent_mode).strip().lower(),
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


def _format_num(value: Any, digits: int = 1, suffix: str = "") -> str:
    v = _to_float(value)
    if v is None:
        return "暂缺"
    return f"{v:.{digits}f}{suffix}"


def _safe_text(value: Any, default: str = "暂缺") -> str:
    s = str(value).strip() if value is not None else ""
    return s if s else default


def _to_cn_text(value: Any) -> str:
    s = _safe_text(value)
    if s in CN_TERM_MAP:
        return CN_TERM_MAP[s]
    out = s
    for en, cn in sorted(CN_TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if len(en) <= 1:
            continue
        out = re.sub(re.escape(en), cn, out, flags=re.IGNORECASE)
    out = re.sub(r"\bsec\b", "秒", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmin\b", "分钟", out, flags=re.IGNORECASE)
    return out


def _format_surgery_cn(value: Any) -> str:
    s = _safe_text(value)
    if s in SURGERY_CN_MAP:
        return SURGERY_CN_MAP[s]
    return _to_cn_text(s)


def _format_stage_cn(value: Any) -> str:
    s = _safe_text(value)
    m = re.search(r"relative timestamp:\s*(\d+)\s*sec", s, re.IGNORECASE)
    if m:
        sec = int(m.group(1))
        return f"术中（相对时间：{sec}秒，约{sec / 60:.1f}分钟）"
    return _to_cn_text(s)


def _format_preop_context(snapshot: Dict[str, Any]) -> str:
    preop = snapshot.get("preop_context", [])
    if isinstance(preop, list):
        items = [_to_cn_text(x) for x in preop if str(x).strip()]
        if items:
            return "；".join(items[:4])
    return "无特殊记录/暂缺"


def _infer_route(snapshot: Dict[str, Any]) -> str:
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).upper()
    itype = str(snapshot.get("interpreted_intervention_type", "")).lower()
    if any(k in med_key for k in ("SEVO", "DES", "ISO", "MAC")):
        return "吸入"
    if "BOLUS" in itype or "bolus_like_event" in itype:
        return "静脉推注"
    return "静脉泵注"


def _format_maintenance_state(snapshot: Dict[str, Any]) -> str:
    concurrent_active = snapshot.get("concurrent_medications_active", [])
    concurrent_all = snapshot.get("concurrent_medications", [])
    concurrent = concurrent_active if (isinstance(concurrent_active, list) and concurrent_active) else concurrent_all
    if isinstance(concurrent, list) and concurrent:
        med_parts: List[str] = []
        for item in concurrent:
            if not isinstance(item, dict):
                continue
            name = _safe_text(item.get("display_name"), _safe_text(item.get("med_base"), "未知药物"))
            rate_v = _to_float(item.get("rate_value"))
            rate_unit = _safe_text(item.get("rate_unit"), "")
            vol_v = _to_float(item.get("volume_ml"))
            bits: List[str] = []
            if rate_v is not None and abs(float(rate_v)) >= 0.01:
                if rate_unit:
                    bits.append(f"速率 {rate_v:.2f} {rate_unit}")
                else:
                    bits.append(f"速率 {rate_v:.2f}")
            if vol_v is not None:
                bits.append(f"累计量 {vol_v:.2f} mL")
            if bits:
                med_parts.append(f"{name}（{'；'.join(bits)}）")
        if med_parts:
            return "；".join(med_parts)

    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = _safe_text(anchor.get("medication_key"), "未知药物")
    med_key_upper = med_key.upper()
    med_name = MEDICATION_DISPLAY.get(med_key, _to_cn_text(med_key))
    route = _infer_route(snapshot)
    rate = _to_float(anchor.get("smoothed_rate_ml_per_h"))
    before = _to_float(anchor.get("before"))
    after = _to_float(anchor.get("after"))
    delta = _to_float(anchor.get("delta"))
    pieces: List[str] = [f"{med_name}（{route}）"]

    volatile_keys = {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE"}
    if med_key_upper in volatile_keys:
        if before is not None and after is not None:
            pieces.append(f"浓度 {before:.2f}→{after:.2f} vol%")
        elif delta is not None:
            pieces.append(f"浓度变化 {delta:+.3f} vol%")
        return "；".join(pieces)

    if med_key_upper == "MAC_RATE":
        if before is not None and after is not None:
            pieces.append(f"MAC {before:.2f}→{after:.2f}")
        elif delta is not None:
            pieces.append(f"MAC变化 {delta:+.3f}")
        return "；".join(pieces)

    if med_key_upper.endswith("_RATE"):
        paired_volume_ml = _to_float(anchor.get("paired_volume_ml"))
        paired_volume_key = _safe_text(anchor.get("paired_volume_key"), "")
        paired_label = MEDICATION_DISPLAY.get(paired_volume_key, _to_cn_text(paired_volume_key)) if paired_volume_key else ""
        if before is not None and after is not None:
            pieces.append(f"速率 {before:.2f}→{after:.2f} mL/h")
            if delta is not None:
                pieces.append(f"变化 {delta:+.2f} mL/h")
        elif delta is not None:
            pieces.append(f"速率变化 {delta:+.3f} mL/h")
        elif rate is not None:
            pieces.append(f"当前速率约 {rate:.2f} mL/h")
        if paired_volume_ml is not None and paired_label:
            pieces.append(f"同时间点{paired_label}约 {paired_volume_ml:.2f} mL")
        return "；".join(pieces)

    if rate is not None:
        pieces.append(f"当前平滑泵速约 {rate:.2f} mL/h")
    if before is not None and after is not None:
        pieces.append(f"累计量 {before:.2f}→{after:.2f} mL")
    elif delta is not None:
        unit = " mL" if med_key_upper.endswith("_VOL") else ""
        pieces.append(f"本次变化 {delta:+.3f}{unit}")
    return "；".join(pieces)


def _format_vital_trend_line(snapshot: Dict[str, Any], key: str, display: str) -> str:
    stats = snapshot.get("vital_stats", {}) if isinstance(snapshot.get("vital_stats"), dict) else {}
    item = stats.get(key, {}) if isinstance(stats.get(key), dict) else {}
    start = item.get("start")
    end = item.get("end")
    if _to_float(start) is None and _to_float(end) is None:
        return f"{display}：暂缺"
    unit = ""
    if key in {"MBP", "SBP", "DBP", "ETCO2", "CVP"}:
        unit = " mmHg"
    elif key == "CO":
        unit = " L/min"
    elif key == "CI":
        unit = " L/(min·m²)"
    elif key == "SV":
        unit = " mL"
    elif key == "SVR":
        unit = " dyn·s·cm⁻5"
    elif key == "HR":
        unit = " bpm"
    elif key in {"SPO2", "SVV", "PPV", "RSO2_L", "RSO2_R"}:
        unit = "%"
    elif key == "BT":
        unit = "℃"
    return f"{display}：{_format_num(start, 1, unit)} -> {_format_num(end, 1, unit)}"


def _build_document_monitoring_summary_lines(snapshot: Dict[str, Any]) -> List[str]:
    assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    recent = assess.get("recent_state_mean", {}) if isinstance(assess, dict) else {}
    baseline = assess.get("baseline_comparison", {}) if isinstance(assess, dict) else {}
    sensitivity = assess.get("sensitivity_policy", {}) if isinstance(assess, dict) else {}
    personalized = sensitivity.get("personalized_thresholds", {}) if isinstance(sensitivity, dict) else {}

    def _cur(keys: Sequence[str], unit: str = "", digits: int = 1) -> str:
        for key in keys:
            v = _to_float(recent.get(key))
            if v is not None:
                return f"{v:.{digits}f}{unit}"
        return "暂缺"

    def _cur_opt(keys: Sequence[str], unit: str = "", digits: int = 1) -> Optional[str]:
        for key in keys:
            v = _to_float(recent.get(key))
            if v is not None:
                return f"{v:.{digits}f}{unit}"
        return None

    map_low = _to_float(personalized.get("map_low_mmhg"))
    if map_low is None:
        map_low = 65.0
    hr_relative = _to_float(personalized.get("hr_relative_change_pct"))
    if hr_relative is None:
        hr_relative = 20.0
    spo2_warn = _to_float(personalized.get("spo2_attention_pct"))
    if spo2_warn is None:
        spo2_warn = 95.0
    spo2_low = _to_float(personalized.get("spo2_low_pct"))
    if spo2_low is None:
        spo2_low = 90.0
    etco2_missing_alert = _to_float(personalized.get("etco2_missing_alert_sec"))
    if etco2_missing_alert is None:
        etco2_missing_alert = 2.0
    temp_low = _to_float(personalized.get("bt_low_c"))
    if temp_low is None:
        temp_low = 36.0
    temp_fever = _to_float(personalized.get("bt_high_fever_c"))
    if temp_fever is None:
        temp_fever = 37.5
    temp_critical = _to_float(personalized.get("bt_high_critical_c"))
    if temp_critical is None:
        temp_critical = 38.0
    bis_low = _to_float(personalized.get("bis_low"))
    if bis_low is None:
        bis_low = 40.0
    bis_high = _to_float(personalized.get("bis_high"))
    if bis_high is None:
        bis_high = 60.0
    rso2_low = _to_float(personalized.get("rso2_low_pct"))
    if rso2_low is None:
        rso2_low = 55.0

    map_drop = _to_float(baseline.get("MAP_drop_from_baseline_pct"))
    hr_change = _to_float(baseline.get("HR_change_from_baseline_pct"))
    spo2_drop = _to_float(baseline.get("SpO2_drop_from_baseline_pct"))
    map_drop_text = f"{map_drop:.1f}%" if map_drop is not None else "未计算"
    hr_change_text = f"{hr_change:.1f}%" if hr_change is not None else "未计算"
    spo2_drop_text = f"{spo2_drop:.1f}%" if spo2_drop is not None else "未计算"

    advanced_hemo_parts: List[str] = []
    advanced_specs = [
        ("CO", ["CO_L_min", "CO"], " L/min"),
        ("CI", ["CI_L_min_m2", "CI"], " L/(min·m²)"),
        ("SV", ["SV_ml", "SV"], " mL"),
        ("SVV", ["SVV_pct", "SVV"], "%"),
        ("PPV", ["PPV_pct", "PPV"], "%"),
        ("CVP", ["CVP_mmhg", "CVP"], " mmHg"),
        ("SVR", ["SVR_dyns_cm5", "SVR"], " dyn·s·cm⁻5"),
    ]
    for label, keys, unit in advanced_specs:
        vtxt = _cur_opt(keys, unit)
        if vtxt is not None:
            advanced_hemo_parts.append(f"{label} {vtxt}")
    advanced_hemo_text = "；".join(advanced_hemo_parts) if advanced_hemo_parts else "高级容量参数本例未连续监测"

    bt_text = _cur_opt(["BT_c"], "℃") or "未连续监测"
    bis_text = _cur_opt(["BIS"], "") or "未连续监测"
    rso2_l_text = _cur_opt(["rSO2_L_pct", "rSO2_L"], "%")
    rso2_r_text = _cur_opt(["rSO2_R_pct", "rSO2_R"], "%")
    if rso2_l_text is None and rso2_r_text is None:
        rso2_text = "未连续监测"
    else:
        rso2_text = f"{rso2_l_text or '未监测'} / {rso2_r_text or '未监测'}"

    return [
        "监测优先级：呼吸(SpO2/EtCO2) > 循环(ECG/HR/SBP/DBP/MAP/CO/CI/SV/SVV/PPV/CVP/SVR) > 体温 > 脑功能(BIS/rSO2) > 凝血/ABG/出入量。",
        f"呼吸：SpO2 {_cur(['SpO2_pct'], '%')}（下降{spo2_drop_text}，警戒<{spo2_warn:.0f}%，硬阈值<{spo2_low:.0f}%）；EtCO2 {_cur(['EtCO2_mmhg'], ' mmHg')}（插管后应连续显示，缺失>{etco2_missing_alert:.0f}s且非校零应立即排查气道/回路）。",
        f"循环：HR {_cur(['HR_bpm'], ' bpm')}（基线变化{hr_change_text}）；SBP/DBP/MAP {_cur(['SBP_mmhg'], ' mmHg')} / {_cur(['DBP_mmhg'], ' mmHg')} / {_cur(['MAP_mmhg'], ' mmHg')}（MAP下降{map_drop_text}，灌注下限约{map_low:.0f} mmHg）；{advanced_hemo_text}。",
        f"体温/脑保护：体温 {bt_text}（<{temp_low:.1f}℃低体温，>{temp_fever:.1f}℃发热，≥{temp_critical:.1f}℃高热）；BIS {bis_text}（建议{bis_low:.0f}-{bis_high:.0f}，仅作提示）；rSO2-L/R {rso2_text}（<{rso2_low:.0f}%或较基线下降>20%异常）。",
        "化验/并发症：ABG（PaO2/PaCO2/pH/乳酸/K/BE）极端值、TEG/ACT异常、尿量<5 mL/h、出血量显著增加、过敏/休克/恶性心律失常/高血糖/低钾/高钾等均需及时预警。",
    ]


def _build_event_and_lab_alert_lines(snapshot: Dict[str, Any]) -> List[str]:
    meta = _snapshot_meta(snapshot)
    adverse_types = meta.get("adverse_event_types", []) if isinstance(meta.get("adverse_event_types"), list) else []
    risk_flags = meta.get("risk_flags", []) if isinstance(meta.get("risk_flags"), list) else []
    adverse_flags = meta.get("adverse_event_flags", []) if isinstance(meta.get("adverse_event_flags"), list) else []

    if not adverse_types and (risk_flags or adverse_flags):
        adverse_types = _infer_adverse_event_types_from_flags([*(str(x) for x in risk_flags), *(str(x) for x in adverse_flags)])

    pretty = {
        "major_bleeding": "大出血",
        "bleeding_warning": "出血预警",
        "anuria_critical": "无尿危重",
        "oliguria_warning": "少尿预警",
        "hyperkalemia_critical": "高钾危重",
        "hypokalemia_critical": "低钾危重",
        "hyperglycemia_severe": "高血糖重度异常",
        "hyperglycemia_warning": "高血糖预警",
        "malignant_arrhythmia": "恶性心律失常",
        "arrhythmia_event": "心律异常事件",
        "shock_pattern": "休克/低灌注模式",
        "suspected_anaphylaxis_pattern": "疑似过敏反应",
        "allergy_history": "过敏相关风险",
        "abg_hypoxemia": "ABG低氧血症风险",
        "abg_hypercapnia": "ABG二氧化碳潴留风险",
        "abg_metabolic_acidosis_hyperlactatemia": "ABG酸中毒+高乳酸风险",
        "abg_metabolic_acidosis_warning": "ABG酸中毒/乳酸预警",
        "abg_be_negative_large": "ABG碱剩余显著负值",
        "coagulation_low": "TEG低凝风险",
        "coagulation_high": "TEG高凝风险",
        "act_abnormal": "ACT异常",
    }
    event_names = [pretty.get(str(x), str(x)) for x in adverse_types[:8] if str(x).strip()]
    line1 = f"并发症标签：{'；'.join(event_names) if event_names else '当前未显式命中并发症标签'}。"
    line2 = "需重点监控：过敏、休克、大出血、恶性心律失常、高血糖、低钾/高钾、低氧血症及气道/回路异常。"
    line3 = "若出现ABG/TEG/ACT极端值或尿量/出血量快速恶化，应升级预警并立即复评。"
    return [line1, line2, line3]


def _build_sensitivity_policy_lines(snapshot: Dict[str, Any]) -> List[str]:
    assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    sensitivity = assess.get("sensitivity_policy", {}) if isinstance(assess, dict) else {}
    personalized = sensitivity.get("personalized_thresholds", {}) if isinstance(sensitivity, dict) else {}

    map_low = _to_float(personalized.get("map_low_mmhg"))
    if map_low is None:
        map_low = 65.0
    hr_relative = _to_float(personalized.get("hr_relative_change_pct"))
    if hr_relative is None:
        hr_relative = 20.0
    spo2_attention = _to_float(personalized.get("spo2_attention_pct"))
    if spo2_attention is None:
        spo2_attention = 95.0
    etco2_missing_alert_sec = _to_float(personalized.get("etco2_missing_alert_sec"))
    if etco2_missing_alert_sec is None:
        etco2_missing_alert_sec = 2.0
    bis_low = _to_float(personalized.get("bis_low"))
    if bis_low is None:
        bis_low = 40.0
    bis_high = _to_float(personalized.get("bis_high"))
    if bis_high is None:
        bis_high = 60.0
    spo2_low = _to_float(personalized.get("spo2_low_pct"))
    if spo2_low is None:
        spo2_low = 90.0
    spo2_warning = _to_float(personalized.get("spo2_attention_pct"))
    if spo2_warning is None:
        spo2_warning = 95.0
    svv_high = _to_float(personalized.get("svv_high_pct"))
    if svv_high is None:
        svv_high = 13.0
    cvp_low = _to_float(personalized.get("cvp_low_mmhg"))
    if cvp_low is None:
        cvp_low = 2.0
    cvp_high = _to_float(personalized.get("cvp_high_mmhg"))
    if cvp_high is None:
        cvp_high = 12.0
    co_low = _to_float(personalized.get("co_low_l_min"))
    if co_low is None:
        co_low = 4.0
    co_high = _to_float(personalized.get("co_high_l_min"))
    if co_high is None:
        co_high = 8.0
    ci_low = _to_float(personalized.get("ci_low_l_min_m2"))
    if ci_low is None:
        ci_low = 2.5
    ci_high = _to_float(personalized.get("ci_high_l_min_m2"))
    if ci_high is None:
        ci_high = 4.0
    sv_low = _to_float(personalized.get("sv_low_ml"))
    if sv_low is None:
        sv_low = 60.0
    sv_high = _to_float(personalized.get("sv_high_ml"))
    if sv_high is None:
        sv_high = 100.0
    svr_low = _to_float(personalized.get("svr_low"))
    if svr_low is None:
        svr_low = 800.0
    svr_high = _to_float(personalized.get("svr_high"))
    if svr_high is None:
        svr_high = 1600.0
    rso2_low = _to_float(personalized.get("rso2_low_pct"))
    if rso2_low is None:
        rso2_low = 55.0
    temp_low = _to_float(personalized.get("bt_low_c"))
    if temp_low is None:
        temp_low = 36.0
    temp_high = _to_float(personalized.get("bt_high_fever_c"))
    if temp_high is None:
        temp_high = 37.5
    temp_critical = _to_float(personalized.get("bt_high_critical_c"))
    if temp_critical is None:
        temp_critical = 38.0
    abg_missing_alert = _to_float(personalized.get("abg_missing_alert_sec"))
    if abg_missing_alert is None:
        abg_missing_alert = 8.0

    return [
        f"高敏感：EtCO2连续性（缺失>{etco2_missing_alert_sec:.0f}s即报警，校零除外）；SpO2警戒{spo2_warning:.0f}%、硬阈值<{spo2_low:.0f}%立即处理。",
        f"中敏感：MAP灌注下限约{map_low:.0f} mmHg；HR较个体基线变化≥{hr_relative:.0f}%需干预评估；BIS仅作趋势提示。",
        f"容量/循环：CO约{co_low:.1f}-{co_high:.1f} L/min；CI约{ci_low:.1f}-{ci_high:.1f} L/(min·m²)；SV约{sv_low:.0f}-{sv_high:.0f} mL；SVV>{svv_high:.0f}%或PPV>13-15%提示容量反应性；CVP约{cvp_low:.0f}-{cvp_high:.0f} mmHg；SVR偏低提示血管扩张，偏高提示血管收缩/后负荷增高。",
        f"体温/脑保护：核心体温{temp_low:.1f}-{temp_high:.1f}℃，>= {temp_critical:.1f}℃按发热/高热处理；BIS维持{bis_low:.0f}-{bis_high:.0f}；rSO2< {rso2_low:.0f}%或较基线下降>20%需警惕脑灌注不足。",
        f"化验/并发症：ABG/TEG/ACT极端值、尿量<5 mL/h、出血量增多、过敏/休克/恶性心律失常/高血糖/低钾/高钾等均需及时报警；ABG缺失本身若在高危手术或氧合通气不稳定场景也应尽快补测（阈值提示：{abg_missing_alert:.0f}s）。",
    ]


def _build_fixed_question(snapshot: Dict[str, Any]) -> str:
    patient = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
    age = _safe_text(patient.get("age"))
    sex = _to_cn_text(patient.get("sex"))
    weight = _safe_text(patient.get("weight_kg"))
    asa = _safe_text(patient.get("asa"))
    surgery = _format_surgery_cn(snapshot.get("surgery_type"))
    stage = _format_stage_cn(snapshot.get("intraop_stage"))
    intraop_summary = (
        snapshot.get("clinical_table_structured", {}).get("intraop_summary", {})
        if isinstance(snapshot.get("clinical_table_structured"), dict)
        else {}
    )
    ebl = _to_float(intraop_summary.get("intraop_ebl")) if isinstance(intraop_summary, dict) else None
    uo = _to_float(intraop_summary.get("intraop_uo")) if isinstance(intraop_summary, dict) else None

    trend_lines = [
        _format_vital_trend_line(snapshot, "BIS", "BIS"),
        _format_vital_trend_line(snapshot, "HR", "HR"),
        _format_vital_trend_line(snapshot, "SBP", "SBP"),
        _format_vital_trend_line(snapshot, "DBP", "DBP"),
        _format_vital_trend_line(snapshot, "MBP", "MAP"),
        _format_vital_trend_line(snapshot, "ETCO2", "EtCO2"),
        _format_vital_trend_line(snapshot, "CO", "CO"),
        _format_vital_trend_line(snapshot, "CI", "CI"),
        _format_vital_trend_line(snapshot, "SV", "SV"),
        _format_vital_trend_line(snapshot, "SVV", "SVV"),
        _format_vital_trend_line(snapshot, "PPV", "PPV"),
        _format_vital_trend_line(snapshot, "SVR", "SVR"),
        _format_vital_trend_line(snapshot, "CVP", "CVP"),
        _format_vital_trend_line(snapshot, "BT", "体温"),
        _format_vital_trend_line(snapshot, "RSO2_L", "rSO2-L"),
        _format_vital_trend_line(snapshot, "RSO2_R", "rSO2-R"),
        _format_vital_trend_line(snapshot, "SPO2", "SpO2"),
    ]
    trend_lines = [x for x in trend_lines if ("：暂缺" not in x)]
    if not trend_lines:
        trend_lines = ["关键体征趋势暂缺（建议补齐术中连续波形监测）"]

    surgery_extra = []
    if ebl is not None:
        surgery_extra.append(f"失血量约{ebl:.0f} mL")
    if uo is not None:
        surgery_extra.append(f"尿量约{uo:.0f} mL")
    surgery_extra_text = f"；{'; '.join(surgery_extra)}" if surgery_extra else ""

    lines = [
        "【患者档案】",
        f"• 年龄/性别/体重/ASA分级：{age}岁，{sex}，{weight} kg，ASA {asa}。",
        f"• 关键术前基础疾病：{_format_preop_context(snapshot)}。",
        "【手术状态】",
        f"• 手术名称：{surgery}。",
        f"• 手术阶段/当前进度：{stage}{surgery_extra_text}。",
        "【麻醉药物维持状态】",
        f"• 药物及当前给药状态：{_format_maintenance_state(snapshot)}。",
        "【体征序列】",
        "• 近5-10分钟动态趋势：",
    ]
    lines.extend([f"  - {x}" for x in trend_lines])
    lines.append("【问题】结合手术背景，此时最合理的干预措施是什么？")
    return "\n".join(lines)


def _build_answer_system_prompt(kind: str, include_review: bool = True) -> str:
    if kind == "miller":
        if include_review:
            return (
                "你是资深麻醉医生。仅输出中文。"
                "必须使用以下标题并按顺序输出："
                "【临床推理】、【宏观策略】、【具体干预】、【复评环节】、【原文摘录】。"
                "不得输出markdown代码块，不得输出额外说明。"
            )
        return (
            "你是资深麻醉医生。仅输出中文。"
            "必须使用以下标题并按顺序输出："
            "【临床推理】、【宏观策略】、【具体干预】、【原文摘录】。"
            "不得输出markdown代码块，不得输出额外说明。"
        )
    if include_review:
        return (
            "你是资深麻醉医生。仅输出中文。"
            "必须使用以下标题并按顺序输出："
            "【临床推理】、【宏观策略】、【具体干预】、【复评环节】。"
            "不得输出markdown代码块，不得输出额外说明。"
        )
    return (
        "你是资深麻醉医生。仅输出中文。"
        "必须使用以下标题并按顺序输出："
        "【临床推理】、【宏观策略】、【具体干预】。"
        "不得输出markdown代码块，不得输出额外说明。"
    )


def _build_answer_user_prompt(
    kind: str,
    question_text: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    include_review: bool = True,
) -> str:
    hint = _golden_action_hint(snapshot)
    med_key = hint.get("medication_key", "")
    actual = hint.get("actual_intervention", "")
    kws = ", ".join(hint.get("keywords", [])) if isinstance(hint.get("keywords"), list) else ""
    expected_unit = _expected_action_unit(snapshot) or "mL/h or mL"
    route = _infer_route(snapshot)
    meta = _snapshot_meta(snapshot)
    recent = meta.get("recent_state_mean", {}) if isinstance(meta.get("recent_state_mean"), dict) else {}
    baseline = meta.get("baseline_comparison", {}) if isinstance(meta.get("baseline_comparison"), dict) else {}
    map_now = _to_float(recent.get("MAP_mmhg"))
    sbp_now = _to_float(recent.get("SBP_mmhg"))
    map_drop = _to_float(baseline.get("MAP_drop_from_baseline_pct"))
    low_perf_risk = _is_hypotension_risk_snapshot(snapshot)
    retrieval_text = ""
    if kind == "miller" and isinstance(retrieval, dict):
        items = retrieval.get("results", []) if isinstance(retrieval.get("results"), list) else []
        short_items: List[str] = []
        for it in items[:3]:
            if not isinstance(it, dict):
                continue
            loc = str(it.get("display_locator") or it.get("locator") or "").strip()
            txt = str(it.get("text") or "").strip()
            short_items.append(f"{loc}\n{txt[:500]}")
        retrieval_text = "\n\n".join(short_items)

    constraints = [
        "【临床推理】要给出鉴别思路与当前最可能机制，严格按以下优先级："
        "1) 呼吸/氧合（SpO2/EtCO2，缺氧风险零容忍）；"
        "2) 血流动力学与器官灌注（MAP/HR/SBP/DBP/容量参数）；"
        "3) 体温；"
        "4) 内环境稳定（ABG/酸碱/电解质/乳酸/凝血/尿量/出血）；"
        "5) 其他辅助预警（如BIS/rSO2，仅作辅助）。",
        "【宏观策略】只写决策大方向。",
        f"【具体干预】必须包含：药名 + 给药途径（至少包含“{route}”或同义表达） + 数值剂量/速率 + 单位（优先 {expected_unit}）。",
    ]
    if include_review:
        constraints.append("【复评环节】必须包含复评时间、目标体征、预期演变、备用方案；其中“预期演变”需写明预计在多久后哪个指标变化到什么范围（如：1分钟后HR回升至约75 bpm）。")
    constraints.extend(
        [
            f"VitalDB版的主干预必须锚定原始记录动作，不得把未记录动作写成已发生的VitalDB干预。原始记录动作：{actual}；药物关键词：{kws}。",
            "【具体干预】第一句必须复述/改写logged_action中的同类药物、方向和数值；"
            "未在logged_action中出现的给药、补液、升压、减药等只能作为“同步安全处理”或“备用方案”，不得新增具体药物剂量作为主决策。",
            "最终输出不得出现任何内部元字段或提示词痕迹，如“med_key=”“logged_action=”“keywords=”等。",
            "若存在SpO2<90%或EtCO2异常，可优先写气道/通气排查等安全动作；但需明确这是临床安全处理，不是VitalDB记录的药物动作。",
            "BIS只做辅助提示，不得脱离MAP/HR/SpO2/EtCO2单独下结论；"
            "SpO2<90%或EtCO2信号异常必须优先处理气道/通气。",
            "血流动力学保护锁：当MAP <65-70 mmHg或SBP显著下降时，严禁优先建议单次推注丙泊酚等扩血管/抑制心肌药；"
            "必须先使用升压/容量策略稳定灌注，再评估是否需要加深麻醉。",
            "全麻BIS目标校准：目标范围应为40-60，不得把BIS目标设在>60的浅镇静区间。",
            "EtCO2病理树：EtCO2下降优先考虑采样/管路脱开、过度通气、心排量下降（低灌注/肺栓塞等）；"
            "EtCO2升高优先考虑低通气、气道阻力增加、代谢增高（如恶性高热）。"
            "不要将“低通气/潮气量不足”直接作为EtCO2下降的常见原因。",
            "除BIS、HR、MAP、SBP、DBP、SpO2、EtCO2、ASA、VitalDB及单位外，避免输出英文短语；疾病、手术、科室和机制说明尽量使用中文。",
            "不要写“缺失/暂无/不可用”等字样。",
        ]
    )
    if low_perf_risk:
        map_txt = f"{map_now:.1f}" if map_now is not None else "暂缺"
        sbp_txt = f"{sbp_now:.1f}" if sbp_now is not None else "暂缺"
        drop_txt = f"{map_drop:.1f}%" if map_drop is not None else "暂缺"
        constraints.append(
            f"当前已存在低灌注风险（MAP≈{map_txt} mmHg, SBP≈{sbp_txt} mmHg, MAP较基线下降≈{drop_txt}），"
            "先升压后加深为硬规则。"
        )

    numbered_constraints = "\n".join([f"{i}) {c}" for i, c in enumerate(constraints, start=1)])
    base = (
        "请基于以下固定问题上下文给出结构化回答。\n\n"
        f"{question_text}\n\n"
        "写作约束：\n"
        f"{numbered_constraints}\n"
    )
    if kind == "miller":
        idx = len(constraints) + 1
        idx2 = idx + 1
        base += (
            f"{idx}) 【原文摘录】必须给英文原句或关键短句，并附定位：[M10#1 | 术中相关章节: xxx | p.xxx]。\n"
            f"{idx2}) 仅可依据提供的Miller证据，不可编造文献。\n\n"
            f"Miller证据（Top-K）：\n{retrieval_text if retrieval_text else '证据定位不足'}\n"
        )
    return base


def _post_chat_with_system(
    url: str,
    headers: Dict[str, str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    obj = resp.json()
    return str(obj["choices"][0]["message"]["content"]).strip()


def _extract_section(text: str, title: str, next_titles: Sequence[str]) -> str:
    norm_text = str(text or "").replace("\u3000", " ")
    norm_title = str(title or "").replace("\u3000", " ")
    start = norm_text.find(norm_title)
    if start < 0:
        # tolerant match: ignore spaces around title tokens
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


def _is_hypotension_risk_snapshot(snapshot: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snapshot, dict):
        return False
    meta = _snapshot_meta(snapshot)
    recent = meta.get("recent_state_mean", {}) if isinstance(meta.get("recent_state_mean"), dict) else {}
    baseline = meta.get("baseline_comparison", {}) if isinstance(meta.get("baseline_comparison"), dict) else {}
    map_now = _to_float(recent.get("MAP_mmhg"))
    sbp_now = _to_float(recent.get("SBP_mmhg"))
    map_drop = _to_float(baseline.get("MAP_drop_from_baseline_pct"))
    sbp_change = _to_float(baseline.get("SBP_change_from_baseline_pct"))
    if map_now is not None and map_now < 70.0:
        return True
    if sbp_now is not None and sbp_now < 90.0:
        return True
    if map_drop is not None and map_drop >= 20.0:
        return True
    if sbp_change is not None and sbp_change <= -30.0:
        return True
    return False


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


def _numeric_values_in_text(text: str) -> List[float]:
    vals: List[float] = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", str(text or "")):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            continue
    return vals


def _has_number_near(values: Sequence[float], target: float, tolerance: float = 0.75) -> bool:
    return any(abs(float(v) - float(target)) <= tolerance for v in values)


def _vitaldb_logged_action_consistent(intervention_text: str, snapshot: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snapshot, dict):
        return True
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key") or "").upper()
    before = _to_float(anchor.get("before"))
    after = _to_float(anchor.get("after"))
    delta = _to_float(anchor.get("delta"))
    if not med_key or med_key in {"ARR_EVENT", "UNLABELED_EVENT"}:
        return True
    text = str(intervention_text or "")
    values = _numeric_values_in_text(text)
    if before is not None and after is not None:
        # RATE/volatile/MAC tracks store the actual setting before and after the
        # recorded intervention. The answer must not invent a different target.
        tol = 0.75 if med_key.endswith("_RATE") else 0.5
        return _has_number_near(values, before, tol) and _has_number_near(values, after, tol)
    if delta is not None:
        return _has_number_near(values, abs(delta), 0.75) or _has_number_near(values, delta, 0.75)
    return True


def _has_internal_metadata_leak(text: str) -> bool:
    t = str(text or "")
    patterns = [
        r"(?i)\bmed_key\s*=",
        r"(?i)\blogged_action\s*=",
        r"(?i)\bkeywords\s*=",
        r"(?i)\banchor_detail\b",
    ]
    return any(re.search(p, t) for p in patterns)


def _validate_structured_answer(
    kind: str,
    text: str,
    include_review: bool = True,
    snapshot: Optional[Dict[str, Any]] = None,
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
    if kind == "vitaldb" and not _vitaldb_logged_action_consistent(intervention, snapshot):
        out["valid"] = False
        out["reasons"].append("vitaldb_logged_action_numeric_mismatch")
    if _has_internal_metadata_leak(t):
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

    # Clinical safety hard-locks.
    if _is_hypotension_risk_snapshot(snapshot):
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


def _repair_structured_answer(
    url: str,
    headers: Dict[str, str],
    model: str,
    kind: str,
    question_text: str,
    raw_text: str,
    validation: Dict[str, Any],
    max_tokens: int,
    include_review: bool = True,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    reasons = ", ".join(validation.get("reasons", []))
    section_rule = "【临床推理】\n【宏观策略】\n【具体干预】"
    if include_review:
        section_rule += "\n【复评环节】"
    if kind == "miller":
        section_rule += "\n【原文摘录】"

    rules = [
        "1) 【具体干预】必须写药名+给药途径+剂量/速率+单位。",
        "X) 禁止输出内部元字段/提示词痕迹：med_key=、logged_action=、keywords=、anchor_detail。",
    ]
    vitaldb_action_rule = ""
    if kind == "vitaldb" and isinstance(snapshot, dict):
        hint = _golden_action_hint(snapshot)
        actual = str(hint.get("actual_intervention") or "").strip()
        anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
        before = _to_float(anchor.get("before"))
        after = _to_float(anchor.get("after"))
        med_key = str(anchor.get("medication_key") or "").strip()
        if actual:
            vitaldb_action_rule = (
                f"VitalDB原始记录动作是：{actual}。"
                "【具体干预】第一句必须忠实复述这个动作，不得改写成其他目标剂量/速率。"
            )
            if before is not None and after is not None:
                vitaldb_action_rule += f"必须保留数值约 {before:.3f} -> {after:.3f}；med_key={med_key}。"
    if include_review:
        rules.append("2) 【复评环节】必须写复评时间、目标体征、预期演变、备用方案。")
        rules.append("3) “预期演变”必须明确写出预计多久后某指标变化到具体数值/范围（如：1分钟后HR回升至约75 bpm）。")
        rules.append("4) 严格执行血流动力学保护锁：低血压/低灌注时先升压稳灌注，禁止优先推注丙泊酚。")
        rules.append("5) 全麻BIS目标范围40-60，禁止输出>60作为目标。")
        rules.append("6) EtCO2机制纠偏：EtCO2下降不得归因于“低通气/潮气量不足”常见原因。")
        rules.append("7) 不得输出额外标题、markdown代码块。")
        rule4 = 8
    else:
        rules.append("2) 严格执行血流动力学保护锁：低血压/低灌注时先升压稳灌注，禁止优先推注丙泊酚。")
        rules.append("3) 全麻BIS目标范围40-60，禁止输出>60作为目标。")
        rules.append("4) EtCO2机制纠偏：EtCO2下降不得归因于“低通气/潮气量不足”常见原因。")
        rules.append("5) 不得输出额外标题、markdown代码块。")
        rule4 = 6
    rules_text = "\n".join(rules)

    sys = "你是医学文本格式修复器，只输出修复后的最终内容，不解释。"
    usr = (
        f"请把下面文本修复为严格结构：\n{section_rule}\n"
        "要求：\n"
        f"{rules_text}\n"
        f"{vitaldb_action_rule}\n"
        f"{rule4}) 当前问题上下文如下：\n{question_text}\n\n"
        f"当前失败原因：{reasons}\n\n"
        f"待修复文本：\n{raw_text}"
    )
    try:
        return _post_chat_with_system(url, headers, model, sys, usr, max_tokens)
    except Exception:
        return None


def _compose_final_output(
    question_text: str,
    vitaldb_text: str,
    miller_text: str,
    include_miller: bool = True,
) -> str:
    if not include_miller:
        return (
            "Q (Input Context)\n"
            f"{question_text}\n\n"
            "Answer（VitalDB版）\n"
            f"{vitaldb_text}"
        )
    return (
        "Q (Input Context)\n"
        f"{question_text}\n\n"
        "Answer（VitalDB版）\n"
        f"{vitaldb_text}\n\n"
        "Answer（Miller版）\n"
        f"{miller_text}"
    )


def _generate_branch(
    url: str,
    headers: Dict[str, str],
    model: str,
    kind: str,
    question_text: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    max_tokens: int,
    include_review: bool = True,
) -> Dict[str, Any]:
    system_prompt = _build_answer_system_prompt(kind, include_review=include_review)
    user_prompt = _build_answer_user_prompt(kind, question_text, snapshot, retrieval, include_review=include_review)

    try:
        raw = _post_chat_with_system(url, headers, model, system_prompt, user_prompt, max_tokens)
    except Exception as e:  # noqa: BLE001
        return {
            "error": str(e),
            "raw_output": "",
            "repaired_output": "",
            "final_output": "",
            "valid": False,
            "validation_raw": {"valid": False, "reasons": ["api_error"]},
            "validation_final": {"valid": False, "reasons": ["api_error"]},
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    validation_raw = _validate_structured_answer(
        kind,
        raw,
        include_review=include_review,
        snapshot=snapshot,
    )
    final_text = raw
    repaired_text = ""
    validation_final = dict(validation_raw)

    if not validation_raw.get("valid", False):
        repaired = _repair_structured_answer(
            url=url,
            headers=headers,
            model=model,
            kind=kind,
            question_text=question_text,
            raw_text=raw,
            validation=validation_raw,
            max_tokens=max_tokens,
            include_review=include_review,
            snapshot=snapshot,
        )
        if repaired and str(repaired).strip():
            repaired_text = str(repaired).strip()
            final_text = repaired_text
            validation_final = _validate_structured_answer(
                kind,
                final_text,
                include_review=include_review,
                snapshot=snapshot,
            )

    return {
        "error": None,
        "raw_output": raw,
        "repaired_output": repaired_text,
        "final_output": final_text,
        "valid": bool(validation_final.get("valid", False)),
        "validation_raw": validation_raw,
        "validation_final": validation_final,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _generate_one(
    api_url: str,
    headers: Dict[str, str],
    model: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]],
    max_tokens: int,
    vitaldb_only: bool = False,
    include_review: bool = True,
) -> Dict[str, Any]:
    question_text = _build_fixed_question(snapshot)
    vitaldb_result = _generate_branch(
        url=api_url,
        headers=headers,
        model=model,
        kind="vitaldb",
        question_text=question_text,
        snapshot=snapshot,
        retrieval=None,
        max_tokens=max_tokens,
        include_review=include_review,
    )
    if vitaldb_only:
        miller_result = {
            "error": None,
            "raw_output": "",
            "repaired_output": "",
            "final_output": "",
            "valid": True,
            "validation_raw": {"valid": True, "reasons": ["skipped_vitaldb_only"]},
            "validation_final": {"valid": True, "reasons": ["skipped_vitaldb_only"]},
            "system_prompt": "",
            "user_prompt": "",
        }
    else:
        miller_result = _generate_branch(
            url=api_url,
            headers=headers,
            model=model,
            kind="miller",
            question_text=question_text,
            snapshot=snapshot,
            retrieval=retrieval,
            max_tokens=max_tokens,
            include_review=include_review,
        )

    vitaldb_text = str(vitaldb_result.get("final_output") or "").strip()
    miller_text = str(miller_result.get("final_output") or "").strip()
    final_output = _compose_final_output(question_text, vitaldb_text, miller_text, include_miller=(not vitaldb_only))

    error_parts: List[str] = []
    if vitaldb_result.get("error"):
        error_parts.append(f"vitaldb_error={vitaldb_result.get('error')}")
    if (not vitaldb_only) and miller_result.get("error"):
        error_parts.append(f"miller_error={miller_result.get('error')}")

    raw_output_obj: Dict[str, Any] = {"vitaldb": vitaldb_result.get("raw_output", "")}
    if not vitaldb_only:
        raw_output_obj["miller"] = miller_result.get("raw_output", "")

    branch_meta_obj: Dict[str, Any] = {"vitaldb": vitaldb_result}
    if not vitaldb_only:
        branch_meta_obj["miller"] = miller_result

    return {
        "error": " | ".join(error_parts) if error_parts else None,
        "valid": bool(vitaldb_result.get("valid", False) and (True if vitaldb_only else miller_result.get("valid", False))),
        "valid_vitaldb": bool(vitaldb_result.get("valid", False)),
        "valid_miller": bool(True if vitaldb_only else miller_result.get("valid", False)),
        "question_text": question_text,
        "vitaldb_output": vitaldb_text,
        "miller_output": "" if vitaldb_only else miller_text,
        "final_output": final_output,
        "raw_output": raw_output_obj,
        "branch_meta": branch_meta_obj,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate anesthesia QA via requests-based GPT API, optionally with Miller retrieval."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--output-field", default="llm_output_gpt_requests")
    parser.add_argument("--vitaldb-only", action="store_true", help="Generate only VitalDB answer branch and skip Miller branch.")
    parser.add_argument(
        "--disable-reeval-section",
        action="store_true",
        help="Disable `【复评环节】` in prompt/validation/output structure.",
    )
    parser.add_argument("--print-miller", action="store_true", help="Print Miller section for each generated record.")
    parser.add_argument(
        "--sample-categories",
        default="warning_signal,critical_alarm",
        help="Comma-separated categories to keep: stable_maintenance,warning_signal,critical_alarm. Default keeps warning+critical.",
    )
    parser.add_argument(
        "--min-risk-level",
        default="moderate",
        choices=["low", "moderate", "high"],
        help="Minimum risk level to keep. Default: moderate.",
    )
    parser.add_argument(
        "--disable-objective-alert-filter",
        action="store_true",
        help="Disable objective alert filter. By default, records must satisfy objective alert conditions.",
    )
    parser.add_argument(
        "--min-hr-relative-change-pct",
        type=float,
        default=20.0,
        help="HR relative baseline change threshold for alert filtering. Default: 20%%.",
    )
    parser.add_argument(
        "--min-objective-alert-count",
        type=int,
        default=1,
        help="Minimum number of objective physiologic alerts required for record keeping. Default: 1.",
    )
    parser.add_argument(
        "--objective-alert-critical-only",
        action="store_true",
        help="When objective alert filter is enabled, only count critical-severity physiologic alerts.",
    )
    parser.add_argument(
        "--require-adverse-event-types",
        default="",
        help=(
            "Comma-separated adverse event type filter. "
            "Examples: major_bleeding,shock_pattern,malignant_arrhythmia,hyperkalemia_critical,"
            "hypokalemia_critical,suspected_anaphylaxis_pattern."
        ),
    )
    parser.add_argument(
        "--adverse-event-match-mode",
        default="any",
        choices=["any", "all"],
        help="Event type matching mode for --require-adverse-event-types. Default: any.",
    )
    parser.add_argument(
        "--require-alarm-tags",
        default="",
        help=(
            "Comma-separated alarm tags to require. "
            "Supported tags include: EtCO2,SpO2,ECG,HR,SBP,DBP,MAP,BT,BIS,rSO2,CO,CI,SV,SVV,PPV,CVP,SVR,"
            "ABG,TEG,ACT,Urine Output,Blood Loss."
        ),
    )
    parser.add_argument(
        "--alarm-tag-match-mode",
        default="any",
        choices=["any", "all"],
        help="Matching mode for --require-alarm-tags. Default: any.",
    )

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
    parser.add_argument(
        "--miller-bis-intent-mode",
        default="dynamic",
        choices=["dynamic", "full", "paired_only", "off"],
        help="BIS retrieval intent mode: dynamic(推荐) / full / paired_only / off.",
    )
    parser.add_argument("--embedding-backend", default="auto", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-api-key", default="")
    args = parser.parse_args()

    records = _load_records(args.input)
    before_filter_n = len(records)
    allowed_categories = _split_csv_tokens(args.sample_categories)
    required_event_types = _split_csv_tokens(args.require_adverse_event_types)
    required_alarm_tags = _split_csv_tokens(args.require_alarm_tags)
    records = _filter_records_by_snapshot_policy(
        records=records,
        allowed_categories=allowed_categories,
        min_risk_level=args.min_risk_level,
        require_objective_alert=(not bool(args.disable_objective_alert_filter)),
        min_hr_relative_change_pct=float(args.min_hr_relative_change_pct),
        min_objective_alert_count=int(args.min_objective_alert_count),
        objective_alert_critical_only=bool(args.objective_alert_critical_only),
        required_alarm_tags=required_alarm_tags,
        alarm_tag_match_mode=args.alarm_tag_match_mode,
        required_adverse_event_types=required_event_types,
        adverse_event_match_mode=args.adverse_event_match_mode,
    )
    print(
        ">>> Snapshot policy filter: "
        f"input={before_filter_n}, kept={len(records)}, "
        f"categories={allowed_categories or ['ALL']}, min_risk={args.min_risk_level}, "
        f"objective_alert_required={not bool(args.disable_objective_alert_filter)}, "
        f"min_objective_alert_count={int(args.min_objective_alert_count)}, "
        f"objective_alert_critical_only={bool(args.objective_alert_critical_only)}, "
        f"required_alarm_tags={required_alarm_tags or ['ALL']}, alarm_tag_match_mode={args.alarm_tag_match_mode}, "
        f"required_events={required_event_types or ['ALL']}, event_match_mode={args.adverse_event_match_mode}"
    )
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records after filtering. Relax --sample-categories / --min-risk-level or disable objective alert filter.")

    retriever = None
    embed_client = None
    retrieval_cfg = _build_retrieval_cfg(args)
    if args.vitaldb_only and args.enable_miller_rag:
        print(">>> vitaldb-only enabled: ignore Miller retrieval settings")
    if (not args.vitaldb_only) and args.enable_miller_rag and not args.use_existing_retrieval:
        embed_client = create_embedding_client(retrieval_cfg)
        retriever = build_miller_retriever(embed_client, retrieval_cfg)

    headers = _build_headers(args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            indicator_alerts = _collect_indicator_alerts(
                snapshot,
                min_hr_relative_change_pct=float(args.min_hr_relative_change_pct),
            )
            retrieval = None
            if (not args.vitaldb_only) and args.use_existing_retrieval and isinstance(record.get("miller_retrieval"), dict):
                retrieval = record["miller_retrieval"]
            elif (not args.vitaldb_only) and args.enable_miller_rag and retriever is not None and embed_client is not None:
                retrieval = retrieve_miller_context(snapshot, retriever, embed_client, retrieval_cfg)

            result = _generate_one(
                args.api_url,
                headers,
                args.model,
                snapshot,
                retrieval,
                args.max_tokens,
                vitaldb_only=bool(args.vitaldb_only),
                include_review=(not bool(args.disable_reeval_section)),
            )
            out = dict(record)
            out["generation_mode"] = "gpt_requests_api"
            if retrieval is not None and (not args.vitaldb_only):
                out["miller_retrieval"] = retrieval
            elif args.vitaldb_only:
                out.pop("miller_retrieval", None)
            out[args.output_field] = result.get("final_output")
            out[f"{args.output_field}_question"] = result.get("question_text", "")
            out[f"{args.output_field}_vitaldb"] = result.get("vitaldb_output", "")
            if not args.vitaldb_only:
                out[f"{args.output_field}_miller"] = result.get("miller_output", "")
            else:
                out.pop(f"{args.output_field}_miller", None)
            out[f"{args.output_field}_raw"] = result.get("raw_output")
            out[f"{args.output_field}_meta"] = {
                "valid": result.get("valid", False),
                "valid_vitaldb": result.get("valid_vitaldb", False),
                "error": result.get("error"),
                "question_text": result.get("question_text", ""),
                "vitaldb_output": result.get("vitaldb_output", ""),
                "branch_meta": result.get("branch_meta", {}),
                "indicator_alerts": indicator_alerts,
                "has_objective_alert": bool(indicator_alerts),
                "snapshot_meta": _snapshot_meta(snapshot),
            }
            if not args.vitaldb_only:
                out[f"{args.output_field}_meta"]["valid_miller"] = result.get("valid_miller", False)
                out[f"{args.output_field}_meta"]["miller_output"] = result.get("miller_output", "")
            f.write(_safe_json_dumps(out) + "\n")
            print(f"  - GPT requests generated {idx}/{len(records)}")
            if args.print_miller and (not args.vitaldb_only):
                print("    Miller:", result.get("miller_output", ""))

    print(f"Done: wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()



