import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import random
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import matplotlib
import numpy as np
import pandas as pd
import vitaldb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import yaml
except Exception:
    yaml = None


# ------------------------------------------
# Track candidates
# ------------------------------------------
MEDICATION_TRACK_CANDIDATES: Dict[str, List[str]] = {
    "PHE_RATE": [
        "Orchestra/PHE_RATE",
    ],
    "EPI_RATE": [
        "Orchestra/EPI_RATE",
    ],
    "PPF20_VOL": [
        "Orchestra/PPF20_VOL",
    ],
    "REMI_VOL": [
        "Orchestra/REMI_VOL",
    ],
}
ADDITIONAL_MEDICATION_TRACK_CANDIDATES: Dict[str, List[str]] = {
    "NOR_RATE": ["Orchestra/NOR_RATE"],
    "EPH_VOL": ["Orchestra/EPH_VOL", "Orchestra/EPHE_VOL", "Orchestra/EPHEDRINE_VOL"],
    "EPH_RATE": ["Orchestra/EPH_RATE", "Orchestra/EPHE_RATE", "Orchestra/EPHEDRINE_RATE"],
    "DOPA_RATE": ["Orchestra/DOPA_RATE"],
    "ESMO_RATE": ["Orchestra/ESMO_RATE"],
    "NICA_RATE": ["Orchestra/NICA_RATE"],
    "NPS_RATE": ["Orchestra/NPS_RATE"],
    "NTG_VOL": ["Orchestra/NTG_VOL", "Orchestra/TNG_VOL", "Orchestra/NITRO_VOL"],
    "NTG_RATE": ["Orchestra/NTG_RATE", "Orchestra/TNG_RATE", "Orchestra/NITRO_RATE"],
    "MIL_VOL": ["Orchestra/MIL_VOL", "Orchestra/MILR_VOL", "Orchestra/MILRINONE_VOL"],
    "MIL_RATE": ["Orchestra/MIL_RATE", "Orchestra/MILR_RATE", "Orchestra/MILRINONE_RATE"],
    "ATRO_VOL": ["Orchestra/ATRO_VOL", "Orchestra/ATROPINE_VOL"],
    "ATRO_RATE": ["Orchestra/ATRO_RATE", "Orchestra/ATROPINE_RATE"],
    "URA_RATE": ["Orchestra/URA_RATE"],
    "REMI_RATE": ["Orchestra/REMI_RATE"],
    "RFTN20_VOL": ["Orchestra/RFTN20_VOL"],
    "RFTN50_VOL": ["Orchestra/RFTN50_VOL"],
    "RFTN20_RATE": ["Orchestra/RFTN20_RATE"],
    "RFTN50_RATE": ["Orchestra/RFTN50_RATE"],
    "ROC_VOL": ["Orchestra/ROC_VOL"],
    "ROC_RATE": ["Orchestra/ROC_RATE"],
    # Volatile anesthetic anchors (ET/FI concentration changes).
    "SEVO_ET_RATE": ["Primus/ETSEVO", "Primus/ET_SEVO", "Solar8000/SEVO_ET"],
    "SEVO_FI_RATE": ["Primus/FISEVO", "Primus/FI_SEVO", "Solar8000/SEVO_FI"],
    "DES_ET_RATE": ["Primus/ETDES", "Primus/ET_DES", "Solar8000/DES_ET"],
    "DES_FI_RATE": ["Primus/FIDES", "Primus/FI_DES", "Solar8000/DES_FI"],
    "ISO_ET_RATE": ["Primus/ETISO", "Primus/ET_ISO", "Solar8000/ISO_ET"],
    "ISO_FI_RATE": ["Primus/FIISO", "Primus/FI_ISO", "Solar8000/ISO_FI"],
    "MAC_RATE": ["Primus/MAC", "Solar8000/MAC"],
}

NON_PROPOFOL_MED_KEYS: List[str] = [
    "NOR_RATE",
    "EPH_VOL",
    "EPH_RATE",
    "PHE_RATE",
    "EPI_RATE",
    "DOPA_RATE",
    "ESMO_RATE",
    "NICA_RATE",
    "NPS_RATE",
    "NTG_VOL",
    "NTG_RATE",
    "MIL_VOL",
    "MIL_RATE",
    "ATRO_VOL",
    "ATRO_RATE",
    "URA_RATE",
    "REMI_VOL",
    "REMI_RATE",
    "RFTN20_VOL",
    "RFTN50_VOL",
    "RFTN20_RATE",
    "RFTN50_RATE",
    "ROC_VOL",
    "ROC_RATE",
    "SEVO_ET_RATE",
    "SEVO_FI_RATE",
    "DES_ET_RATE",
    "DES_FI_RATE",
    "ISO_ET_RATE",
    "ISO_FI_RATE",
    "MAC_RATE",
    "VENT_FIO2",
    "VENT_PEEP",
    "VENT_TV",
]

VASOACTIVE_MED_KEYS = {
    "NOR_RATE",
    "EPH_VOL",
    "EPH_RATE",
    "PHE_RATE",
    "EPI_RATE",
    "DOPA_RATE",
}

ANESTHETIC_DEPTH_MED_KEYS = {
    "PPF20_VOL",
    "SEVO_ET_RATE",
    "SEVO_FI_RATE",
    "DES_ET_RATE",
    "DES_FI_RATE",
    "ISO_ET_RATE",
    "ISO_FI_RATE",
    "MAC_RATE",
}

MILLER_POLICY_THRESHOLDS = {
    "critical_window_sec": 30.0,
    "hemodynamic_window_sec": 60.0,
    "slow_trend_window_sec": 120.0,
    "map_relative_drop_pct": 20.0,
}

MED_CLASS_BY_KEY = {
    "PHE_RATE": "vasopressor",
    "NOR_RATE": "vasopressor",
    "EPH_VOL": "vasopressor",
    "EPH_RATE": "vasopressor",
    "EPI_RATE": "inopressor",
    "DOPA_RATE": "inopressor",
    "ESMO_RATE": "anti_sympathetic",
    "NICA_RATE": "anti_sympathetic",
    "NPS_RATE": "anti_sympathetic",
    "NTG_VOL": "vasodilator",
    "NTG_RATE": "vasodilator",
    "MIL_VOL": "inodilator",
    "MIL_RATE": "inodilator",
    "ATRO_VOL": "chronotropic",
    "ATRO_RATE": "chronotropic",
    "URA_RATE": "anti_sympathetic",
    "PPF20_VOL": "hypnotic_iv",
    "REMI_VOL": "opioid_analgesic",
    "REMI_RATE": "opioid_analgesic",
    "RFTN20_VOL": "opioid_analgesic",
    "RFTN50_VOL": "opioid_analgesic",
    "RFTN20_RATE": "opioid_analgesic",
    "RFTN50_RATE": "opioid_analgesic",
    "ROC_VOL": "neuromuscular",
    "ROC_RATE": "neuromuscular",
    "SEVO_ET_RATE": "hypnotic_volatile",
    "SEVO_FI_RATE": "hypnotic_volatile",
    "DES_ET_RATE": "hypnotic_volatile",
    "DES_FI_RATE": "hypnotic_volatile",
    "ISO_ET_RATE": "hypnotic_volatile",
    "ISO_FI_RATE": "hypnotic_volatile",
    "MAC_RATE": "hypnotic_volatile",
    "ARR_EVENT": "arrhythmia",
    "UNLABELED_EVENT": "unknown",
}

ACTION_DRUG_BY_MED_KEY = {
    "PHE_RATE": "phenylephrine",
    "EPH_VOL": "ephedrine",
    "EPH_RATE": "ephedrine",
    "NOR_RATE": "norepinephrine",
    "EPI_RATE": "epinephrine",
    "NTG_VOL": "nitroglycerin",
    "NTG_RATE": "nitroglycerin",
    "MIL_VOL": "milrinone",
    "MIL_RATE": "milrinone",
    "ATRO_VOL": "atropine",
    "ATRO_RATE": "atropine",
    "PPF20_VOL": "propofol",
    "REMI_VOL": "remifentanil",
    "REMI_RATE": "remifentanil",
    "RFTN20_VOL": "remifentanil",
    "RFTN50_VOL": "remifentanil",
    "RFTN20_RATE": "remifentanil",
    "RFTN50_RATE": "remifentanil",
}

LLM_MAX_TOKENS_DEFAULT = 2048
LEAK_TOKEN_RE = re.compile(
    r"(?is)\b("
    r"wait|strategy|constraint\s*check|analyze\s+the\s+input\s+data|"
    r"self-?correction|content\s+requirements|drafting|thinking\s+process|analysis:"
    r")\b"
)

VITAL_TRACK_CANDIDATES: Dict[str, List[str]] = {
    "HR": [
        "Solar8000/HR",
        "IntelliVue/HR",
        "SNUADC/HR",
        "Primus/HR",
    ],
    "MBP": [
        # Prefer invasive ART first, then fallback to NIBP.
        "Solar8000/ART_MBP",
        "IntelliVue/ABP_MBP",
        "SNUADC/ART_MBP",
        "Solar8000/NIBP_MBP",
        "IntelliVue/NIBP_MBP",
    ],
    "SPO2": [
        "Solar8000/PLETH_SPO2",
        "Solar8000/SPO2",
        "IntelliVue/SpO2",
        "Primus/SPO2",
    ],
    "BIS": ["BIS/BIS"],
}

VITAL_DISPLAY = {
    "HR": "Heart Rate (HR)",
    "MBP": "Mean Arterial Pressure (MBP)",
    "SPO2": "SpO2",
    "BIS": "BIS",
}

VITAL_UNIT = {
    "HR": "bpm",
    "MBP": "mmHg",
    "SPO2": "%",
    "BIS": "",
}

MEDICATION_DISPLAY = {
    "PHE_RATE": "去氧肾上腺素泵速",
    "EPH_VOL": "麻黄碱累计量",
    "EPH_RATE": "麻黄碱泵速",
    "EPI_RATE": "肾上腺素泵速",
    "NTG_VOL": "硝酸甘油累计量",
    "NTG_RATE": "硝酸甘油泵速",
    "MIL_VOL": "米力农累计量",
    "MIL_RATE": "米力农泵速",
    "ATRO_VOL": "阿托品累计量",
    "ATRO_RATE": "阿托品泵速",
    "PPF20_VOL": "丙泊酚累计量",
    "REMI_VOL": "瑞芬太尼累计量",
    "NOR_RATE": "去甲肾上腺素泵速",
    "DOPA_RATE": "多巴胺泵速",
    "ESMO_RATE": "艾司洛尔泵速",
    "NICA_RATE": "尼卡地平泵速",
    "NPS_RATE": "硝普钠泵速",
    "URA_RATE": "乌拉地尔泵速",
    "REMI_RATE": "瑞芬太尼泵速",
    "RFTN20_VOL": "瑞芬太尼20浓度累计量",
    "RFTN50_VOL": "瑞芬太尼50浓度累计量",
    "RFTN20_RATE": "瑞芬太尼20浓度速率",
    "RFTN50_RATE": "瑞芬太尼50浓度速率",
    "ROC_VOL": "罗库溴铵累计量",
    "ROC_RATE": "罗库溴铵泵速",
    "SEVO_ET_RATE": "七氟烷呼气末浓度",
    "SEVO_FI_RATE": "七氟烷吸入浓度",
    "DES_ET_RATE": "地氟烷呼气末浓度",
    "DES_FI_RATE": "地氟烷吸入浓度",
    "ISO_ET_RATE": "异氟烷呼气末浓度",
    "ISO_FI_RATE": "异氟烷吸入浓度",
    "MAC_RATE": "吸入麻醉MAC",
}

ANES_THRESHOLDS = {
    "map_hypotension_mmhg": 65.0,
    "map_severe_hypotension_mmhg": 55.0,
    "map_relative_drop_pct": 20.0,
    "hr_tachycardia_bpm": 100.0,
    "hr_bradycardia_bpm": 50.0,
    "spo2_low_pct": 94.0,
    "spo2_severe_low_pct": 90.0,
    "bis_light": 60.0,
    "bis_deep": 40.0,
    "critical_window_sec": 30.0,
    "hemodynamic_window_sec": 60.0,
    "slow_trend_window_sec": 120.0,
}

RULES_DIR = Path(__file__).resolve().parent / "rules"
CLINICAL_CONFLICT_RULES_PATH = RULES_DIR / "clinical_conflict_rules.yaml"
_CLINICAL_RULES_CACHE: Optional[Dict[str, Any]] = None


def _default_clinical_conflict_rules() -> Dict[str, Any]:
    return {
        "classes_worsen_perfusion": [
            "hypnotic_iv",
            "hypnotic_volatile",
            "anti_sympathetic",
            "vasodilator",
            "inodilator",
        ],
        "conflict_rules": [
            {
                "id": "oxygenation_worsen_perfusion",
                "all": ["strategy_oxygenation_first", "action_escalation", "class_worsen_perfusion"],
                "reason": "低氧场景下优先氧合，但VitalDB策略偏向加深麻醉或降压。",
                "high_risk": True,
            },
            {
                "id": "perfusion_worsen_perfusion",
                "all": ["strategy_perfusion_first", "action_escalation", "class_worsen_perfusion"],
                "reason": "低灌注场景下应先稳灌注，VitalDB策略可能进一步压低血压。",
                "high_risk": True,
            },
            {
                "id": "map_low_escalate_hypnotic_or_vasodilator",
                "all": ["map_low", "action_escalation", "class_hypnotic_or_vasodilator_inodilator"],
                "reason": "MAP<65时继续升级催眠/吸入麻醉或扩血管药，方向上不符合灌注优先。",
                "high_risk": True,
            },
            {
                "id": "bis_high_and_low_map_hypnotic_escalation",
                "all": ["map_below_75", "bis_high", "class_hypnotic", "action_escalation"],
                "reason": "BIS高但MAP已接近/低于灌注安全边界时，单纯加深催眠药风险偏高。",
                "high_risk": True,
            },
            {
                "id": "reduce_depth_but_hypnotic_escalation",
                "all": ["strategy_reduce_depth", "class_hypnotic", "action_escalation"],
                "reason": "BIS低+低灌注时应减浅麻醉，但VitalDB记录为加深麻醉。",
                "high_risk": True,
            },
            {
                "id": "phenylephrine_in_bradycardia",
                "all": ["action_escalation", "drug_phenylephrine", "hr_lt_50"],
                "reason": "去氧肾上腺素在严重心动过缓时可诱发反射性进一步降心率，应避免。",
                "high_risk": True,
            },
            {
                "id": "ephedrine_in_tachycardia",
                "all": ["action_escalation", "drug_ephedrine", "hr_gt_100"],
                "reason": "麻黄碱在心动过速状态下会进一步推高心率，存在心肌缺血/室性心律失常风险。",
                "high_risk": True,
            },
            {
                "id": "epinephrine_non_rescue",
                "all": ["action_escalation", "drug_epinephrine", "map_not_lt_55"],
                "reason": "肾上腺素不宜作为非抢救性常规升压手段。",
                "high_risk": False,
            },
            {
                "id": "nitroglycerin_when_map_low",
                "all": ["action_escalation", "drug_nitroglycerin", "map_low"],
                "reason": "MAP<65时升级硝酸甘油可导致回心血量骤降并加重循环崩溃风险。",
                "high_risk": True,
            },
            {
                "id": "milrinone_when_map_low",
                "all": ["action_escalation", "drug_milrinone", "map_low"],
                "reason": "低血压未纠正前升级米力农可能因扩血管效应导致血压进一步下降。",
                "high_risk": True,
            },
            {
                "id": "atropine_when_tachycardia",
                "all": ["action_escalation", "drug_atropine", "hr_gt_100"],
                "reason": "阿托品在已心动过速时不合适，可能进一步加重心率失控。",
                "high_risk": False,
            },
            {
                "id": "propofol_in_severe_hypotension",
                "all": ["action_escalation", "drug_propofol", "severe_hypotension"],
                "reason": "重度低灌注状态下继续加深丙泊酚可能显著恶化循环。",
                "high_risk": True,
            },
            {
                "id": "remifentanil_brady_hypotension",
                "all": ["action_escalation", "drug_remifentanil", "map_low", "hr_lt_50"],
                "reason": "不明原因心动过缓合并低血压时升级瑞芬太尼可加重缓慢性循环抑制。",
                "high_risk": True,
            },
            {
                "id": "remifentanil_in_severe_hypotension",
                "all": ["action_escalation", "drug_remifentanil", "map_lt_55"],
                "reason": "重度低血压时升级瑞芬太尼可能进一步抑制交感反应，应先纠正灌注。",
                "high_risk": True,
            },
            {
                "id": "norepinephrine_without_volume_optimization",
                "all": ["action_escalation", "drug_norepinephrine", "severe_hypotension", "hr_gt_110", "map_drop_ge_relative"],
                "reason": "疑似低容量未纠正时直接强化去甲升压，可能增加微循环灌注不足风险。",
                "high_risk": True,
            },
        ],
        "alignment_rules": [
            {
                "id": "vasopressor_when_perfusion_first",
                "all": ["class_vasopressor_or_inopressor", "strategy_perfusion_first"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "vasopressor_when_map_low_partial",
                "all": ["class_vasopressor_or_inopressor", "map_low"],
                "not": ["strategy_perfusion_first"],
                "outcome": "partial",
                "reason": "MAP已低但尚未达到持续/重度低灌注规则，升压方向部分合理。",
            },
            {
                "id": "opioid_or_hypnotic_when_bis_high",
                "all": ["class_opioid_or_hypnotic", "strategy_consider_depth_or_analgesia_increase"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "opioid_or_hypnotic_with_low_map_caution",
                "all": ["class_opioid_or_hypnotic", "strategy_consider_depth_or_analgesia_increase", "map_below_75"],
                "outcome": "partial",
                "reason": "BIS升高支持加深镇静/镇痛，但MAP接近灌注下限，需小步滴定和复评。",
            },
            {
                "id": "decrease_hypnotic_when_reduce_depth",
                "all": ["class_hypnotic", "strategy_reduce_depth", "delta_negative"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "monitoring_in_low_signal_context",
                "all": ["strategy_context_monitoring", "class_monitoring_compatible"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "phenylephrine_with_severe_hypotension_hr_not_low",
                "all": ["drug_phenylephrine", "severe_hypotension", "hr_not_low"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "ephedrine_with_low_hr_hypotension",
                "all": ["drug_ephedrine", "severe_hypotension", "hr_lt_60", "hr_le_100"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "norepinephrine_with_severe_hypotension",
                "all": ["drug_norepinephrine", "severe_hypotension"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "epinephrine_rescue_range",
                "all": ["drug_epinephrine", "map_lt_55"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "atropine_with_critical_brady",
                "all": ["drug_atropine", "severe_hypotension", "hr_lt_45"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
            {
                "id": "vasodilator_or_inodilator_when_map_65_75",
                "all": ["drug_vasodilator_or_inodilator", "map_ge_65", "map_below_75"],
                "outcome": "partial",
                "reason": "MAP虽未低于65，但扩血管/正性肌力药仍需严密复评灌注。",
            },
            {
                "id": "vasodilator_or_inodilator_when_map_ge_75",
                "all": ["drug_vasodilator_or_inodilator", "map_ge_75"],
                "outcome": "aligned",
                "reason": "action_class_matches_miller_priority",
            },
        ],
    }


def _load_clinical_conflict_rules() -> Dict[str, Any]:
    global _CLINICAL_RULES_CACHE  # noqa: PLW0603
    if _CLINICAL_RULES_CACHE is not None:
        return _CLINICAL_RULES_CACHE

    rules = _default_clinical_conflict_rules()
    try:
        if CLINICAL_CONFLICT_RULES_PATH.exists() and yaml is not None:
            loaded = yaml.safe_load(CLINICAL_CONFLICT_RULES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                rules = loaded
    except Exception:
        pass

    _CLINICAL_RULES_CACHE = rules
    return rules


def _rule_matches_facts(rule: Dict[str, Any], facts: Dict[str, bool]) -> bool:
    all_facts = rule.get("all", []) if isinstance(rule.get("all"), list) else []
    not_facts = rule.get("not", []) if isinstance(rule.get("not"), list) else []
    for key in all_facts:
        if not bool(facts.get(str(key), False)):
            return False
    for key in not_facts:
        if bool(facts.get(str(key), False)):
            return False
    return True

DRUG_REFERENCE = {
    "Phenylephrine": {
        "common_scenario": "Hypotension with normal/high HR",
        "bolus_range": "40-100 mcg IV bolus",
        "infusion_range": "0.2-1.0 mcg/kg/min (or titrated by BP response)",
        "contraindication": "Severe bradycardia (HR < 50 bpm)",
        "safety_note": "Pure alpha agonist may worsen reflex bradycardia; avoid in marked bradycardia.",
    },
    "Ephedrine": {
        "common_scenario": "Hypotension with low/normal-low HR",
        "bolus_range": "5-10 mg IV bolus (repeat titration)",
        "contraindication": "Marked tachycardia (HR > 100 bpm)",
        "safety_note": "May further increase myocardial oxygen demand and provoke tachyarrhythmia.",
    },
    "Norepinephrine": {
        "common_scenario": "Refractory vasodilatory hypotension / vasoplegia",
        "infusion_range": "0.02-0.3 mcg/kg/min (titrated)",
        "contraindication": "Uncorrected severe hypovolemia",
        "safety_note": "Prefer volume optimization before aggressive vasoconstriction.",
    },
    "Epinephrine": {
        "common_scenario": "Severe hypotension/shock with low cardiac output",
        "bolus_range": "5-20 mcg IV bolus (case-dependent)",
        "infusion_range": "0.01-0.1 mcg/kg/min (titrated)",
        "contraindication": "Routine non-rescue blood pressure correction",
        "safety_note": "High arrhythmia/lactate risk; reserve for rescue-level scenarios.",
    },
    "Nitroglycerin": {
        "common_scenario": "Myocardial ischemia, severe hypertension, acute pulmonary edema",
        "infusion_range": "5-200 mcg/min (titrated)",
        "contraindication": "MAP < 65 mmHg, RV infarct, severe aortic stenosis",
        "safety_note": "Venodilation can collapse preload when perfusion is already unstable.",
    },
    "Milrinone": {
        "common_scenario": "Post-cardiac surgery RV failure / pulmonary hypertension with low output",
        "loading_range": "often omitted or slow low-dose loading in unstable patients",
        "infusion_range": "0.25-0.75 mcg/kg/min (titrated)",
        "contraindication": "Uncorrected hypotension",
        "safety_note": "Early vasodilation may require concurrent vasopressor support.",
    },
    "Atropine": {
        "common_scenario": "Hemodynamically significant bradycardia (e.g., HR < 45 with hypotension)",
        "bolus_range": "0.5 mg IV, repeat to max 3 mg",
        "contraindication": "Ineffective in denervated transplanted heart; caution in tachycardia",
        "safety_note": "Use for symptomatic bradycardia, not for stable low resting HR alone.",
    },
    "Propofol": {
        "common_scenario": "Anesthesia depth maintenance/deepening",
        "maintenance_range": "4-10 mg/kg/h (adult GA, individualized)",
        "contraindication": "Severe hypotension / shock state",
        "safety_note": "Myocardial depression and vasodilation may further collapse perfusion.",
    },
    "Remifentanil": {
        "common_scenario": "Analgesia during stimulation / sympathetic surge control",
        "infusion_range": "0.05-2.0 mcg/kg/min (titrated to stimulus/hemodynamics)",
        "contraindication": "Unexplained bradycardia with hypotension",
        "safety_note": "Can aggravate bradycardia; reassess perfusion before escalation.",
    },
    "Sevoflurane": {
        "common_scenario": "Volatile hypnotic maintenance",
        "maintenance_range": "ET 1.0-2.5 vol% (adjust by age/MAC and hemodynamics)",
    },
    "Desflurane": {
        "common_scenario": "Volatile hypnotic maintenance",
        "maintenance_range": "ET 3-6 vol% (individualized by age/MAC and hemodynamics)",
    },
    "Isoflurane": {
        "common_scenario": "Volatile hypnotic maintenance",
        "maintenance_range": "ET 0.8-1.5 vol% (individualized by age/MAC and hemodynamics)",
    },
}

ARRDB_TIME_COL_CANDIDATES: List[str] = [
    "time_sec",
    "time_second",
    "time_s",
    "time",
    "sec",
    "seconds",
    "timestamp_sec",
    "timestamp",
]

ARRDB_LABEL_COL_CANDIDATES: List[str] = [
    "label",
    "arrhythmia",
    "arrhythmia_label",
    "rhythm",
    "rhythm_label",
    "annotation",
    "event",
]

ARRDB_NORMAL_LABELS = {
    "",
    "normal",
    "sinus",
    "sinus rhythm",
    "normal sinus rhythm",
    "nsr",
    "n",
}

SYSTEM_PROMPT = """
You are a senior anesthesiologist.
Given de-identified structured intraoperative data, answer in Chinese.
If you need internal reasoning, keep it short inside <think>...</think> (max 3 lines).
After </think>, output EXACTLY ONE QA pair in strict format:
Q: ...
A: 【临床推理】：...
【决策干预（Miller）】：...
【决策干预（VitalDB）】：...
Do not output any bullets, headings, checklists, drafting notes, or instruction echoes.
""".strip()

GOLDEN_ACTION_KEYWORDS: Dict[str, List[str]] = {
    "PHE_RATE": ["去氧肾上腺素", "苯肾上腺素", "phenylephrine"],
    "EPH_VOL": ["麻黄碱", "ephedrine"],
    "EPH_RATE": ["麻黄碱", "ephedrine"],
    "EPI_RATE": ["肾上腺素", "epinephrine"],
    "NTG_VOL": ["硝酸甘油", "nitroglycerin", "glyceryl trinitrate"],
    "NTG_RATE": ["硝酸甘油", "nitroglycerin", "glyceryl trinitrate"],
    "MIL_VOL": ["米力农", "milrinone"],
    "MIL_RATE": ["米力农", "milrinone"],
    "ATRO_VOL": ["阿托品", "atropine"],
    "ATRO_RATE": ["阿托品", "atropine"],
    "PPF20_VOL": ["丙泊酚", "propofol"],
    "REMI_VOL": ["瑞芬太尼", "remifentanil"],
    "REMI_RATE": ["瑞芬太尼", "remifentanil"],
    "RFTN20_VOL": ["瑞芬太尼", "remifentanil"],
    "RFTN50_VOL": ["瑞芬太尼", "remifentanil"],
    "RFTN20_RATE": ["瑞芬太尼", "remifentanil"],
    "RFTN50_RATE": ["瑞芬太尼", "remifentanil"],
    "NOR_RATE": ["去甲肾上腺素", "norepinephrine"],
    "DOPA_RATE": ["多巴胺", "dopamine"],
    "ESMO_RATE": ["艾司洛尔", "esmolol"],
    "NICA_RATE": ["尼卡地平", "nicardipine"],
    "NPS_RATE": ["硝普钠", "nitroprusside"],
    "URA_RATE": ["乌拉地尔", "urapidil"],
    "ROC_VOL": ["罗库溴铵", "rocuronium"],
    "ROC_RATE": ["罗库溴铵", "rocuronium"],
    "SEVO_ET_RATE": ["七氟烷", "sevoflurane"],
    "SEVO_FI_RATE": ["七氟烷", "sevoflurane"],
    "DES_ET_RATE": ["地氟烷", "desflurane"],
    "DES_FI_RATE": ["地氟烷", "desflurane"],
    "ISO_ET_RATE": ["异氟烷", "isoflurane"],
    "ISO_FI_RATE": ["异氟烷", "isoflurane"],
    "MAC_RATE": ["吸入麻醉", "volatile", "mac"],
    "ARR_EVENT": ["心律", "arrhythmia"],
    "UNLABELED_EVENT": [],
}

FEWSHOT_BY_TYPE: Dict[str, str] = {
    "continuous_infusion": (
        "### Example (continuous_infusion)\n"
        "<think>患者胸外科术中，近5分钟 MAP 下行而 BIS 上升，提示麻醉深度与血流动力学存在冲突。"
        "先稳灌注，再小步调整镇静药速率。</think>\n"
        "Q: 胸外科术中患者在维持期出现血压下降趋势，同时麻醉深度指标波动，此时应如何进行更安全的药物调整以兼顾灌注和镇静？\n"
        "A: 【临床推理】：当前关键矛盾是循环稳定性与麻醉深度的平衡。若在低灌注状态下盲目加深镇静，可能进一步加重低血压并影响器官灌注。\n"
        "【决策干预（Miller）】：优先纠正灌注（如滴定升压药），再小步调整镇静/镇痛，并设定短周期复评。\n"
        "【决策干预（VitalDB）】：按记录用药类别执行并复核其与当前血流动力学是否一致。\n"
        "### End Example\n"
    ),
    "bolus_like_event": (
        "### Example (bolus_like_event)\n"
        "<think>患者短时刺激期体征上冲，单次追加药物应以短效、可回退为原则。需避免过度镇静后低血压。</think>\n"
        "Q: 患者在术中刺激期出现体征突变，何时应考虑单次追加给药而非持续大幅上调泵速？\n"
        "A: 【临床推理】：短时、可逆的生理波动更适合短效追加干预；持续上调可能带来过量风险。需要结合血压、心率与麻醉深度的同步变化判断。\n"
        "【决策干预（Miller）】：可优先短效小剂量追加，并在1-3分钟内复评后决定是否继续升级。\n"
        "【决策干预（VitalDB）】：按记录药物类别执行对应的追加/调速策略并保留可回退性。\n"
        "### End Example\n"
    ),
    "arrhythmia_event": (
        "### Example (arrhythmia_event)\n"
        "<think>出现心律事件时，先判断血流动力学稳定性，再决定是否立即药理/电复律路径。麻醉深度与氧合通气也需并行评估。</think>\n"
        "Q: 术中突发心律失常标注事件时，最关键的初始处理优先级是什么？\n"
        "A: 【临床推理】：处理顺序应先看灌注与血压稳定性，再区分可观察与需立即干预的节律。同时排查缺氧、二氧化碳潴留、电解质异常及麻醉深度不匹配。\n"
        "【决策干预（Miller）】：若循环不稳先按围术期急救流程处理；循环稳定时先纠正诱因后再保守滴定节律控制药。\n"
        "【决策干预（VitalDB）】：按记录干预执行，并核对是否与当前循环稳定性等级匹配。\n"
        "### End Example\n"
    ),
    "unlabeled_context_snapshot": (
        "### Example (unlabeled_context_snapshot)\n"
        "<think>无明确事件标签时，依据趋势而非单点，优先识别威胁灌注与氧合的指标。在信息不全时给出保守且可复评的决策。</think>\n"
        "Q: 在缺少明确事件标签的术中快照中，如何基于趋势信息做出安全的药理决策？\n"
        "A: 【临床推理】：应以 MAP/SpO2/HR 的连续趋势为主线，避免仅凭单一瞬时异常下结论。信息缺失时优先采取可逆、可滴定的策略。\n"
        "【决策干预（Miller）】：先小步幅、可逆调整并设置1-3分钟复评窗口，再决定是否升级干预。\n"
        "【决策干预（VitalDB）】：按照记录药物策略执行，同时持续监测MAP/HR/SpO2验证有效性。\n"
        "### End Example\n"
    ),
}


SURGERY_GROUP_RULES: Dict[str, List[str]] = {
    "Thoracic_Surgery": ["thorac", "vats", "lung", "chest", "pulmonary", "mediast"],
    "Neurosurgery": ["neuro", "brain", "crani", "spine", "intracran", "cns"],
    "General_Surgery": ["general", "gastric", "colon", "rect", "hep", "chole", "pancre", "hernia"],
    "Urology": ["uro", "kidney", "renal", "bladder", "prostate"],
    "Gynecology": ["gyn", "hyster", "ovary", "uter", "obstet"],
    "Orthopedics": ["ortho", "joint", "hip", "knee", "fracture", "arthro", "spine"],
    "Cardiac_Surgery": ["cardiac", "cabg", "valve", "aorta", "bypass"],
}


@dataclass
class PipelineConfig:
    clinical_csv: str
    output_dir: str
    group_root: str
    image_root: str
    dataset_jsonl: str
    snapshot_json: str
    llm_jsonl: str
    miller_retrieval_log_jsonl: str
    miller_retrieval_log_csv: str
    miller_retrieval_log_max_chars: int
    signal_interval_sec: float
    med_check_interval_sec: float
    window_sec: int
    min_window_points: int
    anes_dur_min: float
    rate_delta_threshold: float
    vol_delta_threshold: float
    vol_rate_lookback_sec: float
    min_anchor_gap_sec: float
    enable_mbp_unit_fix: bool
    mbp_kpa_threshold: float
    mbp_kpa_to_mmhg_factor: float
    propofol_bolus_rate_threshold_ml_h: float
    propofol_bolus_min_delta_ml: float
    max_cases: int
    max_anchors_per_case: int
    skip_setup_rate_anchors: bool
    setup_rate_before_abs_max: float
    setup_rate_after_threshold: float
    setup_rate_delta_threshold: float
    setup_rate_early_window_sec: float
    skip_medication_filter: bool
    keep_source_duplicate_rows: bool
    anchor_mode: str
    arrdb_annotation_dir: str
    arrdb_time_column: str
    arrdb_label_column: str
    arrdb_keep_normal: bool
    periodic_anchor_step_sec: float
    periodic_anchor_start_sec: float
    department_include: str
    llm_max_workers: int
    llm_progress_every: int
    enable_llm: bool
    llm_model: str
    validate_actual_before_qa: bool
    drop_if_actual_invalid: bool
    drop_if_actual_uncertain: bool
    actual_validation_model: str
    actual_validation_max_tokens: int
    api_key_env: str
    llm_base_url: str
    llm_api_key: str
    enable_miller_rag: bool
    miller_corpus_path: str
    miller_index_path: str
    miller_top_k: int
    miller_chunk_chars: int
    miller_chunk_overlap_chars: int
    miller_max_passage_chars: int
    miller_bis_intent_mode: str
    miller_depth_focus_weight: float
    miller_require_chapter: bool
    miller_allowed_chapters: str
    embedding_backend: str
    embedding_model: str
    embedding_device: str
    embedding_base_url: str
    embedding_api_key_env: str
    embedding_api_key: str
    overwrite_jsonl: bool
    sample_rate: float
    random_seed: int
    export_bucketed_datasets: bool
    train_mix_a_ratio: float
    train_mix_seed: int
    train_mix_max_samples: int
    strict_a_requires_risk_flags: bool
    strict_a_requires_objective_evidence: bool


@dataclass
class MillerRetriever:
    passages: List[Dict[str, Any]]
    embeddings: np.ndarray
    term_freqs: List[Dict[str, int]]
    doc_freqs: Dict[str, int]
    doc_lengths: np.ndarray
    avg_doc_length: float

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.embeddings.size == 0 or not self.passages:
            return []
        qvec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        qnorm = float(np.linalg.norm(qvec))
        if qnorm <= 0:
            return []
        qvec = qvec / qnorm
        scores = self.embeddings @ qvec
        limit = max(1, min(int(top_k), len(self.passages)))
        top_indices = np.argsort(-scores)[:limit]
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.passages[int(idx)])
            item["score"] = float(scores[int(idx)])
            item["rank"] = rank
            hits.append(item)
        return hits

    def bm25_search(self, query_text: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        if not self.passages:
            return []
        qtokens = _tokenize_for_bm25(query_text)
        if not qtokens:
            return []
        scores = np.zeros(len(self.passages), dtype=np.float32)
        n_docs = max(1, len(self.passages))
        avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
        unique_terms = Counter(qtokens)
        for term, qtf in unique_terms.items():
            df = int(self.doc_freqs.get(term, 0))
            if df <= 0:
                continue
            idf = float(np.log1p((n_docs - df + 0.5) / (df + 0.5)))
            for idx, tf_map in enumerate(self.term_freqs):
                tf = int(tf_map.get(term, 0))
                if tf <= 0:
                    continue
                denom = tf + k1 * (1.0 - b + b * (float(self.doc_lengths[idx]) / avgdl))
                scores[idx] += idf * ((tf * (k1 + 1.0)) / max(1e-6, denom)) * float(qtf)

        limit = max(1, min(int(top_k), len(self.passages)))
        top_indices = np.argsort(-scores)[:limit]
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            if float(scores[int(idx)]) <= 0:
                continue
            item = dict(self.passages[int(idx)])
            item["bm25_score"] = float(scores[int(idx)])
            item["rank"] = rank
            hits.append(item)
        return hits


@dataclass
class LocalEmbeddingClient:
    model: Any
    device: str


def is_valid(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def to_caseid(v: Any) -> Optional[int]:
    if not is_valid(v):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def first_valid(row: pd.Series, keys: Sequence[str], default: Any = "Unknown") -> Any:
    for key in keys:
        if key in row and is_valid(row[key]):
            return row[key]
    return default


def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pick_first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _coerce_text(row.get(key))
        if value:
            return value
    return ""


def _infer_chapter_from_text(text: str) -> Tuple[str, str]:
    src = _coerce_text(text)
    if not src:
        return "", ""
    head = " ".join(src.strip().split())[:220]
    patterns = [
        r"^\s*(\d{1,3})\s*[•·\-\–]\s*([A-Za-z][A-Za-z0-9 ,:&()/\-]{3,120})",
        r"(?i)\bchapter\s+(\d{1,3})\s*[:\-–]?\s*([A-Za-z][A-Za-z0-9 ,:&()/\-]{3,120})",
    ]
    for pat in patterns:
        m = re.search(pat, head)
        if not m:
            continue
        num = _coerce_text(m.group(1))
        title = _coerce_text(m.group(2))
        title = re.sub(r"\s+", " ", title).strip(" .;,-")
        if num:
            return num, title
    return "", ""


def _miller_locator_parts(item: Dict[str, Any]) -> Dict[str, Any]:
    chapter = _pick_first_nonempty(item, ["chapter", "chapter_title", "chapter_name", "chapter_id"])
    section = _pick_first_nonempty(item, ["section", "section_title", "section_name"])
    subsection = _pick_first_nonempty(item, ["subsection", "subsection_title", "subsection_name"])
    paragraph = _pick_first_nonempty(item, ["paragraph", "paragraph_id", "para_id", "paragraph_index"])
    page = _pick_first_nonempty(item, ["page", "page_no", "page_index"])
    line_no = _pick_first_nonempty(item, ["line_no"])
    chunk_id = _pick_first_nonempty(item, ["chunk_id"])
    page_chunk_index = _pick_first_nonempty(item, ["page_chunk_index"])
    if not chapter:
        inferred_chapter, inferred_title = _infer_chapter_from_text(item.get("text"))
        if inferred_chapter:
            chapter = inferred_chapter
        if inferred_title and not section:
            section = inferred_title
    if not paragraph:
        if page_chunk_index:
            paragraph = f"p{page or '?'}_chunk{page_chunk_index}"
        elif chunk_id:
            paragraph = str(chunk_id)
    return {
        "chapter": chapter,
        "section": section,
        "subsection": subsection,
        "paragraph": paragraph,
        "page": page,
        "line_no": line_no,
        "chunk_id": chunk_id,
    }


def _format_miller_locator(item: Dict[str, Any], rank: Any = None) -> str:
    parts = _miller_locator_parts(item)
    rank_text = str(rank if rank is not None else item.get("rank", "?")).strip() or "?"
    loc_tokens = []
    if parts["chapter"]:
        loc_tokens.append(f"章节:{parts['chapter']}")
    if parts["section"]:
        loc_tokens.append(f"小节:{parts['section']}")
    if parts["subsection"]:
        loc_tokens.append(f"子节:{parts['subsection']}")
    if parts["paragraph"]:
        loc_tokens.append(f"段落:{parts['paragraph']}")
    if parts["page"]:
        loc_tokens.append(f"页:{parts['page']}")
    if parts["line_no"]:
        loc_tokens.append(f"行:{parts['line_no']}")
    if parts["chunk_id"]:
        loc_tokens.append(f"chunk:{parts['chunk_id']}")
    loc_body = "; ".join(loc_tokens) if loc_tokens else "章节:未知; 段落:未知"
    return f"[M10#{rank_text}|{loc_body}]"


def _tokenize_for_bm25(text: str) -> List[str]:
    src = _coerce_text(text).lower()
    if not src:
        return []
    return re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", src)


def _build_bm25_state(passages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, int]], Dict[str, int], np.ndarray, float]:
    term_freqs: List[Dict[str, int]] = []
    doc_freqs: Dict[str, int] = {}
    doc_lengths: List[int] = []
    for passage in passages:
        tokens = _tokenize_for_bm25(_coerce_text(passage.get("text")))
        tf = Counter(tokens)
        term_freqs.append(dict(tf))
        doc_lengths.append(len(tokens))
        for term in tf.keys():
            doc_freqs[term] = doc_freqs.get(term, 0) + 1
    doc_lengths_arr = np.asarray(doc_lengths, dtype=np.float32)
    avg_doc_length = float(doc_lengths_arr.mean()) if doc_lengths_arr.size > 0 else 0.0
    return term_freqs, doc_freqs, doc_lengths_arr, avg_doc_length


def _chunk_text_blocks(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    raw_blocks = [blk.strip() for blk in re.split(r"\n\s*\n+", text) if blk.strip()]
    if not raw_blocks:
        raw_blocks = [text.strip()] if text.strip() else []
    if not raw_blocks:
        return []
    merged: List[str] = []
    current = ""
    for block in raw_blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if current and len(candidate) > chunk_chars:
            merged.append(current.strip())
            carry = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{carry} {block}".strip() if carry else block
        else:
            current = candidate
    if current:
        merged.append(current.strip())
    return merged


def _load_miller_corpus_chunks(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    corpus_path = os.path.abspath(cfg.miller_corpus_path)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Miller corpus not found: {corpus_path}")
    ext = os.path.splitext(corpus_path)[1].lower()
    chunks: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                text = ""
                for key in ("text", "content", "passage", "chunk", "body"):
                    text = _coerce_text(row.get(key))
                    if text:
                        break
                if not text:
                    continue
                source = _coerce_text(row.get("source")) or _coerce_text(row.get("title")) or os.path.basename(corpus_path)
                chunk: Dict[str, Any] = {
                    "text": text,
                    "source": source,
                    "chunk_id": len(chunks),
                    "line_no": line_no,
                }
                for src_key, dst_key in (
                    ("chapter", "chapter"),
                    ("chapter_title", "chapter"),
                    ("chapter_name", "chapter"),
                    ("section", "section"),
                    ("section_title", "section"),
                    ("section_name", "section"),
                    ("subsection", "subsection"),
                    ("subsection_title", "subsection"),
                    ("subsection_name", "subsection"),
                    ("paragraph", "paragraph"),
                    ("paragraph_id", "paragraph"),
                    ("para_id", "paragraph"),
                    ("paragraph_index", "paragraph"),
                    ("page", "page"),
                    ("page_no", "page"),
                    ("page_index", "page"),
                ):
                    value = _coerce_text(row.get(src_key))
                    if value and not _coerce_text(chunk.get(dst_key)):
                        chunk[dst_key] = value
                chunks.append(chunk)
        return chunks

    if ext == ".pdf":
        if PdfReader is None:
            raise ImportError("Reading PDF Miller corpus requires `pypdf`. Please install it first.")
        reader = PdfReader(corpus_path)
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = _coerce_text(page.extract_text())
            if not page_text:
                continue
            page_chunks = _chunk_text_blocks(
                page_text,
                cfg.miller_chunk_chars,
                cfg.miller_chunk_overlap_chars,
            )
            for local_idx, chunk in enumerate(page_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "source": os.path.basename(corpus_path),
                        "page": page_idx,
                        "chunk_id": len(chunks),
                        "page_chunk_index": local_idx,
                    }
                )
        return chunks

    with open(corpus_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    for idx, chunk in enumerate(
        _chunk_text_blocks(full_text, cfg.miller_chunk_chars, cfg.miller_chunk_overlap_chars)
    ):
        chunks.append(
            {
                "text": chunk,
                "source": os.path.basename(corpus_path),
                "chunk_id": idx,
            }
        )
    return chunks


def _embedding_cache_meta(cfg: PipelineConfig, corpus_path: str) -> Dict[str, Any]:
    stat = os.stat(corpus_path)
    signature = hashlib.sha256(
        (
            f"{os.path.abspath(corpus_path)}|{stat.st_size}|{int(stat.st_mtime)}|"
            f"{cfg.embedding_backend}|{cfg.embedding_model}|{cfg.embedding_device}|"
            f"{cfg.miller_chunk_chars}|{cfg.miller_chunk_overlap_chars}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "signature": signature,
        "corpus_path": os.path.abspath(corpus_path),
        "embedding_backend": cfg.embedding_backend,
        "embedding_model": cfg.embedding_model,
        "embedding_device": cfg.embedding_device,
        "chunk_chars": int(cfg.miller_chunk_chars),
        "chunk_overlap_chars": int(cfg.miller_chunk_overlap_chars),
    }


def _embed_texts(client: Any, model: str, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    if isinstance(client, LocalEmbeddingClient):
        vectors = client.model.encode(
            [str(x) for x in texts],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[start : start + batch_size]]
        resp = client.embeddings.create(model=model, input=batch)
        data = sorted(resp.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in data)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def resolve_embedding_backend(cfg: PipelineConfig) -> str:
    backend = str(cfg.embedding_backend).strip().lower()
    if backend in {"api", "local"}:
        return backend
    model_ref = cfg.embedding_model.strip()
    if model_ref and os.path.exists(model_ref):
        return "local"
    if cfg.embedding_base_url.strip() or cfg.llm_base_url.strip():
        return "api"
    return "local"


def create_embedding_client(cfg: PipelineConfig) -> Any:
    backend = resolve_embedding_backend(cfg)
    if backend == "local":
        if SentenceTransformer is None:
            raise ImportError(
                "Local embedding backend requires `sentence-transformers`. Install requirements first."
            )
        model_ref = cfg.embedding_model.strip()
        if not model_ref:
            raise ValueError("Local embedding backend requires --embedding-model")
        device = cfg.embedding_device.strip() or "cpu"
        model = SentenceTransformer(model_ref, device=device)
        return LocalEmbeddingClient(model=model, device=device)

    if OpenAI is None:
        raise ImportError("openai package is not installed")
    api_key = cfg.embedding_api_key.strip() or cfg.llm_api_key.strip()
    if not api_key:
        env_name = cfg.embedding_api_key_env.strip() or cfg.api_key_env.strip()
        api_key = os.getenv(env_name, "").strip()
    base_url = cfg.embedding_base_url.strip() or cfg.llm_base_url.strip()
    if base_url:
        if not api_key:
            api_key = "local"
        return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    if not api_key:
        env_name = cfg.embedding_api_key_env.strip() or cfg.api_key_env.strip()
        raise EnvironmentError(f"Missing embedding API key in --embedding-api-key or env {env_name}")
    return OpenAI(api_key=api_key)


def _make_miller_retriever(passages: List[Dict[str, Any]], embeddings: np.ndarray) -> MillerRetriever:
    term_freqs, doc_freqs, doc_lengths, avg_doc_length = _build_bm25_state(passages)
    return MillerRetriever(
        passages=passages,
        embeddings=embeddings,
        term_freqs=term_freqs,
        doc_freqs=doc_freqs,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
    )


def build_miller_retriever(client: Optional[Any], cfg: PipelineConfig) -> MillerRetriever:
    if not cfg.enable_miller_rag:
        return _make_miller_retriever(passages=[], embeddings=np.zeros((0, 0), dtype=np.float32))
    if not cfg.miller_corpus_path.strip():
        raise ValueError("--enable-miller-rag requires --miller-corpus-path")

    corpus_path = os.path.abspath(cfg.miller_corpus_path)
    cache_path = cfg.miller_index_path.strip()
    expected_meta = _embedding_cache_meta(cfg, corpus_path)

    if cache_path and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            meta = json.loads(str(cached["meta_json"].item()))
            if meta == expected_meta:
                passage_json = cached["passage_json"]
                passages = [json.loads(str(x)) for x in passage_json.tolist()]
                embeddings = _normalize_embeddings(np.asarray(cached["embeddings"], dtype=np.float32))
                return _make_miller_retriever(passages=passages, embeddings=embeddings)
        except Exception:
            pass

    if client is None:
        raise RuntimeError(
            "Miller index cache is missing or stale, and embedding client is unavailable. "
            "Provide a valid --miller-index-path cache or an embedding backend/model that can run now."
        )

    passages = _load_miller_corpus_chunks(cfg)
    if not passages:
        raise ValueError(f"No valid Miller corpus passages found in {corpus_path}")
    embeddings = _embed_texts(client, cfg.embedding_model, [p["text"] for p in passages])
    embeddings = _normalize_embeddings(embeddings)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            passage_json=np.asarray([json.dumps(p, ensure_ascii=False) for p in passages], dtype=object),
            meta_json=np.asarray(json.dumps(expected_meta, ensure_ascii=False), dtype=object),
        )
    return _make_miller_retriever(passages=passages, embeddings=embeddings)


def _translate_miller_hint(text: str) -> str:
    src = _coerce_text(text)
    if not src:
        return ""

    replacements = [
        (
            r"重度低血压（MAP < ([\d.]+) mmHg，持续约([\d.]+)s，>= ([\d.]+)s）",
            r"severe hypotension (MAP < \1 mmHg for about \2 s, >= \3 s)",
        ),
        (
            r"低血压（MAP < ([\d.]+) mmHg，持续约([\d.]+)s，>= ([\d.]+)s）",
            r"hypotension (MAP < \1 mmHg for about \2 s, >= \3 s)",
        ),
        (
            r"MAP低于([\d.]+)但持续不足([\d.]+)s（早期预警）",
            r"MAP below \1 mmHg but lasting less than \2 s (early warning)",
        ),
        (
            r"心动过速（HR > ([\d.]+) bpm，持续约([\d.]+)s）",
            r"tachycardia (HR > \1 bpm for about \2 s)",
        ),
        (
            r"心动过缓（HR < ([\d.]+) bpm，持续约([\d.]+)s）",
            r"bradycardia (HR < \1 bpm for about \2 s)",
        ),
        (
            r"重度低氧血症（SpO2 < ([\d.]+)%，持续约([\d.]+)s）",
            r"severe hypoxemia (SpO2 < \1% for about \2 s)",
        ),
        (
            r"血氧下降（SpO2 < ([\d.]+)%，持续约([\d.]+)s）",
            r"oxygen desaturation (SpO2 < \1% for about \2 s)",
        ),
        (
            r"BIS持续偏高（>([\d.]+)，持续约([\d.]+)s，需结合刺激与EMG）",
            r"persistently elevated BIS (>\1 for about \2 s; interpret with stimulation and EMG)",
        ),
        (
            r"BIS持续偏低（<([\d.]+)，持续约([\d.]+)s，需结合低灌注排查）",
            r"persistently low BIS (<\1 for about \2 s; evaluate possible hypoperfusion)",
        ),
        (
            r"BIS短时偏离（持续不足([\d.]+)s，不单独作为给药依据）",
            r"brief BIS deviation lasting less than \1 s; not a standalone drug trigger",
        ),
        (
            r"MAP较基线下降明显（([\d.]+)%）",
            r"marked MAP decrease from baseline (\1%)",
        ),
        (
            r"MAP绝对阈值触发：作为器官灌注底线优先处理",
            r"absolute MAP threshold triggered; prioritize organ perfusion floor",
        ),
        (
            r"MAP相对下降触发：用于个体化风险分层",
            r"relative MAP drop triggered; use for individualized risk stratification",
        ),
        (
            r"BIS 数据缺失（优先依据MAP/HR/SpO2趋势和手术刺激评估）",
            r"BIS unavailable; infer anesthetic depth from MAP/HR/SpO2 trends and surgical stimulation",
        ),
        (
            r"MAP低\+HR低：优先考虑麻醉相关抑制或传导问题，避免单纯加深麻醉。",
            r"low MAP with low HR suggests anesthetic depression or conduction suppression; avoid simply deepening anesthesia",
        ),
        (
            r"MAP低\+HR高：需警惕低容量/失血或应激反应，不应仅按BIS加药。",
            r"low MAP with high HR suggests hypovolemia, bleeding, or stress response; do not escalate drugs based on BIS alone",
        ),
        (
            r"MAP低\+正在加深麻醉：符合药理性低血压风险，应先守住灌注底线。",
            r"low MAP while anesthesia is being deepened suggests drug-induced hypotension risk; protect perfusion first",
        ),
        (
            r"MAP低\+升压药背景：提示可能为难治性低血压，需要复核病因与容量状态。",
            r"low MAP despite vasopressor background suggests refractory hypotension; reassess cause and volume status",
        ),
        (
            r"BIS高合并低血压时，不能机械加深麻醉，应先稳定循环。",
            r"high BIS with hypotension should not trigger automatic deepening; stabilize circulation first",
        ),
        (
            r"BIS高\+HR/MAP上冲更像镇痛不足或镇静镇痛双不足，需要联合调整。",
            r"high BIS with HR/MAP surge suggests inadequate analgesia or combined hypnotic-analgesic insufficiency",
        ),
        (
            r"BIS低\+MAP低提示可能过深麻醉并低灌注，宜减浅麻醉并支持循环。",
            r"low BIS with low MAP suggests excessive anesthetic depth and hypoperfusion; lighten anesthesia and support circulation",
        ),
    ]
    out = src
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out)

    sex_map = {"M": "male", "F": "female"}
    if out in sex_map:
        return sex_map[out]
    return out


def _join_english_hints(items: Any, limit: int = 4) -> str:
    if not isinstance(items, list):
        return ""
    translated: List[str] = []
    for item in items[:limit]:
        hint = _translate_miller_hint(str(item))
        if hint:
            translated.append(hint)
    return "; ".join(translated)


def _append_intent(intents: List[str], text: str) -> None:
    item = _coerce_text(text)
    if item and item not in intents:
        intents.append(item)


def build_miller_intent_tags(snapshot: Dict[str, Any], cfg: Optional[Any] = None) -> List[str]:
    patient = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
    assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    recent = assess.get("recent_state_mean", {}) if isinstance(assess, dict) else {}
    baseline = assess.get("baseline_comparison", {}) if isinstance(assess, dict) else {}
    flags = assess.get("risk_flags", []) if isinstance(assess, dict) else []
    contextual = assess.get("contextual_interpretation", []) if isinstance(assess, dict) else []
    context = snapshot.get("preop_context", [])
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}

    map_now = _safe_float(recent.get("MAP_mmhg"))
    hr_now = _safe_float(recent.get("HR_bpm"))
    spo2_now = _safe_float(recent.get("SpO2_pct"))
    bis_now = _safe_float(recent.get("BIS"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))
    surgery_group = _coerce_text(patient.get("surgery_group"))
    surgery = _coerce_text(snapshot.get("surgery_type"))
    med_key = _coerce_text(anchor.get("medication_key"))

    bis_mode = str(getattr(cfg, "miller_bis_intent_mode", "paired_only") or "paired_only").strip().lower()
    if bis_mode not in {"full", "paired_only", "off"}:
        bis_mode = "paired_only"
    allow_isolated_bis = bis_mode == "full"
    allow_paired_bis = bis_mode in {"full", "paired_only"}

    intents: List[str] = []
    if surgery_group:
        _append_intent(intents, f"{surgery_group.replace('_', ' ').lower()} anesthesia")
    if surgery:
        _append_intent(intents, surgery.lower())
    if isinstance(context, list):
        for item in context[:2]:
            cleaned = _coerce_text(item)
            if cleaned:
                _append_intent(intents, cleaned.lower())

    if map_now is not None and map_now < 55.0:
        _append_intent(intents, "intraoperative severe hypotension")
        _append_intent(intents, "perfusion-first management")
    elif map_now is not None and map_now < 65.0:
        _append_intent(intents, "intraoperative hypotension")
        _append_intent(intents, "perfusion-first management")
    if map_drop_pct is not None and map_drop_pct >= 20.0:
        _append_intent(intents, "relative MAP decrease from baseline")
    if spo2_now is not None and spo2_now < 90.0:
        _append_intent(intents, "intraoperative hypoxemia")
        _append_intent(intents, "oxygenation-first management")
    elif spo2_now is not None and spo2_now < 94.0:
        _append_intent(intents, "oxygen desaturation during anesthesia")
    if hr_now is not None and hr_now > 100.0:
        _append_intent(intents, "intraoperative tachycardia")
    elif hr_now is not None and hr_now < 50.0:
        _append_intent(intents, "intraoperative bradycardia")
    if allow_isolated_bis:
        if bis_now is not None and bis_now > 60.0:
            _append_intent(intents, "high BIS during general anesthesia")
        elif bis_now is not None and bis_now < 40.0:
            _append_intent(intents, "low BIS during general anesthesia")

    if allow_paired_bis and map_now is not None and map_now < 65.0 and bis_now is not None and bis_now > 60.0:
        _append_intent(intents, "high BIS with hypotension")
        _append_intent(intents, "do not deepen anesthesia before stabilizing perfusion")
    if allow_paired_bis and map_now is not None and map_now < 65.0 and bis_now is not None and bis_now < 40.0:
        _append_intent(intents, "excessive anesthetic depth with hypoperfusion")
        _append_intent(intents, "reduce anesthetic depth and support circulation")
    if allow_paired_bis and bis_now is not None and bis_now > 60.0 and hr_now is not None and hr_now > 100.0:
        _append_intent(intents, "inadequate analgesia versus inadequate anesthetic depth")
    if med_key in {"REMI_VOL", "REMI_RATE", "RFTN20_VOL", "RFTN50_VOL", "RFTN20_RATE", "RFTN50_RATE"}:
        _append_intent(intents, "opioid titration during general anesthesia")
    if med_key == "PPF20_VOL":
        _append_intent(intents, "propofol adjustment during general anesthesia")
    if med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE", "MAC_RATE"}:
        _append_intent(intents, "volatile anesthetic adjustment")
    if med_key in {"PHE_RATE", "EPH_VOL", "EPH_RATE", "NOR_RATE", "EPI_RATE"}:
        _append_intent(intents, "vasopressor choice during intraoperative hypotension")

    def _allow_bis_hint(hint: str) -> bool:
        low = str(hint or "").lower()
        if "bis" not in low:
            return True
        if bis_mode == "off":
            return False
        if bis_mode == "full":
            return True
        paired_terms = ("map", "hypotension", "perfusion", "hr", "tachycardia", "bradycardia", "spo2", "oxygen")
        return any(term in low for term in paired_terms)

    if isinstance(flags, list):
        for flag in flags[:3]:
            translated = _translate_miller_hint(str(flag)).lower()
            if _allow_bis_hint(translated):
                _append_intent(intents, translated)
    if isinstance(contextual, list):
        for item in contextual[:2]:
            translated = _translate_miller_hint(str(item)).lower()
            if _allow_bis_hint(translated):
                _append_intent(intents, translated)

    return intents[:8]


def rewrite_miller_query(snapshot: Dict[str, Any], cfg: Optional[Any] = None) -> Tuple[List[str], str]:
    intents = build_miller_intent_tags(snapshot, cfg=cfg)
    if not intents:
        return [], "intraoperative anesthesia management; anesthetic depth; hemodynamic stability"
    return intents, "; ".join(intents)


def _clinical_focus_score(text: str, intent_tags: Sequence[str], cfg: Optional[Any] = None) -> float:
    low = _coerce_text(text).lower()
    if not low:
        return 0.0
    score = 0.0
    depth_weight = float(getattr(cfg, "miller_depth_focus_weight", 0.10))
    depth_weight = max(0.0, min(0.5, depth_weight))
    focus_groups: Dict[str, Tuple[Tuple[str, ...], float]] = {
        "hemodynamics": (("hypotension", "blood pressure", "arterial pressure", "map", "perfusion", "vasopressor"), 0.25),
        "depth": (("bis", "depth of anesthesia", "anesthetic depth", "volatile", "propofol", "hypnosis"), depth_weight),
        "stimulus": (("stimulation", "surgical stimulation", "analgesia", "opioid", "remifentanil", "nociception"), 0.25),
        "oxygenation": (("oxygenation", "hypoxemia", "desaturation", "ventilation", "one-lung"), 0.25),
        "thoracic": (("thoracic", "lung", "one-lung ventilation", "lobectomy"), 0.25),
    }
    for terms, weight in focus_groups.values():
        if any(term in low for term in terms):
            score += float(weight)

    for tag in intent_tags:
        tag_tokens = [tok for tok in _tokenize_for_bm25(str(tag)) if len(tok) >= 4]
        if not tag_tokens:
            continue
        overlap = sum(1 for tok in tag_tokens if tok in low)
        score += min(0.25, 0.05 * overlap)

    generic_penalty_terms = (
        "contents",
        "index",
        "copyright",
        "preface",
        "acknowledgments",
        "preoperative evaluation",
        "history and physical examination",
    )
    if any(term in low for term in generic_penalty_terms):
        score -= 0.6
    return score


def _parse_allowed_chapters(raw: str) -> set[str]:
    values: set[str] = set()
    for token in str(raw or "").split(","):
        t = str(token).strip()
        if not t:
            continue
        values.add(t.lower())
        m = re.search(r"\d{1,3}", t)
        if m:
            values.add(m.group(0))
    return values


def _chapter_matches(chapter_text: str, allowed: set[str]) -> bool:
    if not chapter_text:
        return False
    chap = str(chapter_text).strip().lower()
    if not allowed:
        return True
    if chap in allowed:
        return True
    m = re.search(r"\d{1,3}", chap)
    if m and m.group(0) in allowed:
        return True
    return False


def build_miller_query(snapshot: Dict[str, Any], cfg: Optional[Any] = None) -> str:
    patient = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
    assess = snapshot.get("clinical_assessment", {}) if isinstance(snapshot.get("clinical_assessment"), dict) else {}
    recent = (
        assess.get("recent_state_mean", {})
        if isinstance(assess, dict)
        else {}
    )
    flags = (
        assess.get("risk_flags", [])
        if isinstance(assess, dict)
        else []
    )
    contextual = (
        assess.get("contextual_interpretation", [])
        if isinstance(assess, dict)
        else []
    )
    context = snapshot.get("preop_context", [])
    surgery = _coerce_text(snapshot.get("surgery_type")) or "unknown surgery"
    stage = _coerce_text(snapshot.get("intraop_stage"))
    age = _coerce_text(patient.get("age")) or "unknown age"
    sex = _translate_miller_hint(_coerce_text(patient.get("sex")) or "unknown sex")
    asa = _coerce_text(patient.get("asa")) or "unknown ASA"
    department = _coerce_text(patient.get("department"))
    surgery_group = _coerce_text(patient.get("surgery_group"))
    risk_text = _join_english_hints(flags, limit=4)
    interp_text = _join_english_hints(contextual, limit=3)
    ctx_text = "; ".join(str(x) for x in context[:3]) if isinstance(context, list) else ""
    map_now = recent.get("MAP_mmhg")
    hr_now = recent.get("HR_bpm")
    spo2_now = recent.get("SpO2_pct")
    bis_now = recent.get("BIS")
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = _coerce_text(anchor.get("medication_key"))
    intervention_type = _coerce_text(snapshot.get("interpreted_intervention_type"))
    intents, rewritten = rewrite_miller_query(snapshot, cfg=cfg)
    return (
        "Perioperative anesthesia evidence retrieval query: "
        f"{age}-year-old {sex}, ASA {asa}, department {department}, surgery group {surgery_group}, "
        f"undergoing {surgery}. Current stage: {stage}. "
        f"Preoperative context: {ctx_text}. "
        f"Recent mean physiologic state: MAP {map_now} mmHg, HR {hr_now} bpm, SpO2 {spo2_now}%, BIS {bis_now}. "
        f"Risk flags: {risk_text}. "
        f"Clinical interpretation: {interp_text}. "
        f"Intent tags: {'; '.join(intents)}. "
        f"Rewritten retrieval focus: {rewritten}. "
        f"Observed action anchor: medication key {med_key}, intervention type {intervention_type}. "
        "Retrieve the most relevant Miller anesthesia evidence on anesthetic depth adjustment, analgesic titration, "
        "hemodynamic safety thresholds, perfusion-first priority, oxygenation-first priority, BIS interpretation, "
        "and medication choice for this physiologic scenario."
    )


def retrieve_miller_context(
    snapshot: Dict[str, Any],
    retriever: MillerRetriever,
    embedding_client: Optional[Any],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    query_raw = build_miller_query(snapshot, cfg=cfg)
    intent_tags, query_rewritten = rewrite_miller_query(snapshot, cfg=cfg)
    bm25_hits = retriever.bm25_search(query_rewritten, max(cfg.miller_top_k * 3, 8))
    if embedding_client is None:
        hits = []
        for rank, item in enumerate(bm25_hits[: cfg.miller_top_k], start=1):
            ranked = dict(item)
            ranked["rank"] = rank
            ranked["text"] = _coerce_text(ranked.get("text"))[: cfg.miller_max_passage_chars]
            ranked["retrieval_methods"] = ["bm25"]
            hits.append(ranked)
        return {
            "query": query_rewritten,
            "query_raw": query_raw,
            "query_rewritten": query_rewritten,
            "intent_tags": intent_tags,
            "bm25_results": hits,
            "dense_results": [],
            "results": hits,
            "retrieval_mode": "bm25_only",
        }

    query_vec = _embed_texts(embedding_client, cfg.embedding_model, [query_rewritten])
    if query_vec.shape[0] == 0:
        return {
            "query": query_rewritten,
            "query_raw": query_raw,
            "query_rewritten": query_rewritten,
            "intent_tags": intent_tags,
            "bm25_results": bm25_hits,
            "dense_results": [],
            "results": [],
        }
    dense_hits = retriever.search(query_vec[0], max(cfg.miller_top_k * 3, 8))

    fusion: Dict[str, Dict[str, Any]] = {}
    rrf_k = 60.0
    bm25_weight = 0.6
    dense_weight = 0.4

    for rank, item in enumerate(bm25_hits, start=1):
        key = f"{item.get('source')}::{item.get('chunk_id')}"
        fusion[key] = {
            **item,
            "bm25_rank": rank,
            "bm25_score": float(item.get("bm25_score", 0.0)),
            "dense_rank": None,
            "dense_score": 0.0,
            "fusion_score": bm25_weight / (rrf_k + rank),
            "retrieval_methods": ["bm25"],
        }

    for rank, item in enumerate(dense_hits, start=1):
        key = f"{item.get('source')}::{item.get('chunk_id')}"
        if key not in fusion:
            fusion[key] = {
                **item,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "dense_rank": rank,
                "dense_score": float(item.get("score", 0.0)),
                "fusion_score": dense_weight / (rrf_k + rank),
                "retrieval_methods": ["dense"],
            }
        else:
            fusion[key]["dense_rank"] = rank
            fusion[key]["dense_score"] = float(item.get("score", 0.0))
            fusion[key]["fusion_score"] += dense_weight / (rrf_k + rank)
            methods = list(fusion[key].get("retrieval_methods", []))
            if "dense" not in methods:
                methods.append("dense")
            fusion[key]["retrieval_methods"] = methods

    for item in fusion.values():
        focus_score = _clinical_focus_score(_coerce_text(item.get("text")), intent_tags, cfg=cfg)
        item["clinical_focus_score"] = float(focus_score)
        item["fusion_score"] = float(item.get("fusion_score", 0.0)) + (0.15 * focus_score)

    ranked_all = sorted(fusion.values(), key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)
    require_chapter = bool(getattr(cfg, "miller_require_chapter", False))
    allowed_chapters = _parse_allowed_chapters(getattr(cfg, "miller_allowed_chapters", ""))
    if require_chapter or allowed_chapters:
        filtered: List[Dict[str, Any]] = []
        for item in ranked_all:
            parts = _miller_locator_parts(item)
            chapter = str(parts.get("chapter") or "").strip()
            if require_chapter and not chapter:
                continue
            if allowed_chapters and (not _chapter_matches(chapter, allowed_chapters)):
                continue
            filtered.append(item)
        # Strict by default: if chapter-constrained result exists, only keep constrained hits.
        # If none exists, fallback to original ranked list to avoid empty retrieval.
        ranked_all = filtered if filtered else ranked_all

    hits = ranked_all[: cfg.miller_top_k]
    for rank, item in enumerate(hits, start=1):
        item["text"] = _coerce_text(item.get("text"))[: cfg.miller_max_passage_chars]
        item["rank"] = rank
    for bucket in (bm25_hits[: cfg.miller_top_k], dense_hits[: cfg.miller_top_k]):
        for item in bucket:
            item["text"] = _coerce_text(item.get("text"))[: cfg.miller_max_passage_chars]
    return {
        "query": query_rewritten,
        "query_raw": query_raw,
        "query_rewritten": query_rewritten,
        "intent_tags": intent_tags,
        "bm25_results": bm25_hits[: cfg.miller_top_k],
        "dense_results": dense_hits[: cfg.miller_top_k],
        "results": hits,
    }


def infer_surgery_group(department: str, opname: str) -> str:
    text = f"{department} {opname}".lower()
    for group, kws in SURGERY_GROUP_RULES.items():
        if any(kw in text for kw in kws):
            return group
    dept_clean = "".join(ch if ch.isalnum() else "_" for ch in department.strip())
    dept_clean = "_".join([x for x in dept_clean.split("_") if x])
    if dept_clean:
        return f"Dept_{dept_clean}"
    return "Other"


def resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sanitize_text_for_json(text: str) -> str:
    # Remove illegal control chars that can break JSONL parsing in some readers.
    return "".join(ch for ch in text if (ch in "\n\r\t" or ord(ch) >= 32))


def _sanitize_obj_for_json(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize_text_for_json(obj)
    if isinstance(obj, list):
        return [_sanitize_obj_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_obj_for_json(v) for k, v in obj.items()}
    return obj


def _safe_json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(_sanitize_obj_for_json(obj), ensure_ascii=False)


def _build_miller_retrieval_log_record(
    rec: Dict[str, Any],
    retrieval: Dict[str, Any],
    max_chars: int,
) -> Dict[str, Any]:
    snapshot = rec.get("snapshot", {}) if isinstance(rec.get("snapshot"), dict) else {}
    output: Dict[str, Any] = {
        "caseid": rec.get("caseid"),
        "operation_time_sec": snapshot.get("operation_time_sec"),
        "query": retrieval.get("query"),
        "query_raw": retrieval.get("query_raw"),
        "query_rewritten": retrieval.get("query_rewritten"),
        "intent_tags": retrieval.get("intent_tags"),
        "results": [],
    }
    for item in retrieval.get("results", []) if isinstance(retrieval.get("results"), list) else []:
        if not isinstance(item, dict):
            continue
        text = _coerce_text(item.get("text"))
        locator_parts = _miller_locator_parts(item)
        output["results"].append(
            {
                "rank": item.get("rank"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "chapter": locator_parts.get("chapter"),
                "section": locator_parts.get("section"),
                "subsection": locator_parts.get("subsection"),
                "paragraph": locator_parts.get("paragraph"),
                "page": locator_parts.get("page"),
                "line_no": locator_parts.get("line_no"),
                "locator": _format_miller_locator(item, rank=item.get("rank")),
                "fusion_score": item.get("fusion_score"),
                "bm25_rank": item.get("bm25_rank"),
                "dense_rank": item.get("dense_rank"),
                "retrieval_methods": item.get("retrieval_methods"),
                "clinical_focus_score": item.get("clinical_focus_score"),
                "text": text[: max(100, int(max_chars))],
            }
        )
    return output


def _iter_miller_retrieval_csv_rows(log_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    query = log_record.get("query")
    query_raw = log_record.get("query_raw")
    query_rewritten = log_record.get("query_rewritten")
    caseid = log_record.get("caseid")
    op_time = log_record.get("operation_time_sec")
    intent_tags = log_record.get("intent_tags")
    intent_text = ", ".join(str(x) for x in intent_tags) if isinstance(intent_tags, list) else str(intent_tags or "")
    for item in log_record.get("results", []) if isinstance(log_record.get("results"), list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "caseid": caseid,
                "operation_time_sec": op_time,
                "query": query,
                "query_raw": query_raw,
                "query_rewritten": query_rewritten,
                "intent_tags": intent_text,
                "rank": item.get("rank"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "chapter": item.get("chapter"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "paragraph": item.get("paragraph"),
                "page": item.get("page"),
                "line_no": item.get("line_no"),
                "locator": item.get("locator"),
                "fusion_score": item.get("fusion_score"),
                "bm25_rank": item.get("bm25_rank"),
                "dense_rank": item.get("dense_rank"),
                "retrieval_methods": ", ".join(item.get("retrieval_methods", []))
                if isinstance(item.get("retrieval_methods"), list)
                else str(item.get("retrieval_methods") or ""),
                "clinical_focus_score": item.get("clinical_focus_score"),
                "text": item.get("text"),
            }
        )
    return rows


def _physio_filter_series(series: pd.Series, key: Optional[str]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if key == "MBP":
        # MAP physiologic bound in mmHg; also removes negative/zero artifacts.
        return s.where((s >= 20.0) & (s <= 220.0))
    if key == "HR":
        return s.where((s >= 20.0) & (s <= 220.0))
    if key == "SPO2":
        return s.where((s >= 50.0) & (s <= 100.0))
    if key == "BIS":
        return s.where((s >= 1.0) & (s <= 100.0))
    return s


def resolve_vital_column(df: pd.DataFrame, vital_key: str) -> Optional[str]:
    cands = VITAL_TRACK_CANDIDATES.get(vital_key, [])
    best_col: Optional[str] = None
    best_n = -1
    for col in cands:
        if col not in df.columns:
            continue
        s = _physio_filter_series(pd.to_numeric(df[col], errors="coerce"), key=vital_key).dropna()
        n = int(len(s))
        if n > best_n:
            best_n = n
            best_col = col
    if best_col is not None and best_n > 0:
        return best_col
    return resolve_column(df, cands)


def medication_track_candidates() -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {k: list(v) for k, v in MEDICATION_TRACK_CANDIDATES.items()}
    for key, cands in ADDITIONAL_MEDICATION_TRACK_CANDIDATES.items():
        if key in merged:
            continue
        merged[key] = list(cands)
    return merged


def all_track_candidates() -> List[str]:
    tracks: List[str] = []
    for cands in medication_track_candidates().values():
        tracks.extend(cands)
    for cands in VITAL_TRACK_CANDIDATES.values():
        tracks.extend(cands)
    return list(dict.fromkeys(tracks))


def load_clinical_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"clinical CSV not found: {path}")
    df = pd.read_csv(path)
    if "caseid" not in df.columns:
        raise ValueError("clinical CSV must contain column: caseid")
    df = df.copy()
    df["caseid"] = df["caseid"].apply(to_caseid)
    df = df.dropna(subset=["caseid"])
    df["caseid"] = df["caseid"].astype(int)
    return df


def fetch_medication_frame(caseid: int, interval_sec: float) -> Optional[pd.DataFrame]:
    med_tracks = []
    for cands in medication_track_candidates().values():
        med_tracks.extend(cands)
    med_tracks = list(dict.fromkeys(med_tracks))
    try:
        vf = vitaldb.VitalFile(caseid, track_names=med_tracks)
        med_df = vf.to_pandas(med_tracks, interval_sec)
    except Exception:
        return None
    if med_df is None or med_df.empty:
        return None
    return med_df


def _rate_delta_threshold_for_med_key(med_key: str, cfg: PipelineConfig) -> float:
    base = float(cfg.rate_delta_threshold)
    if med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE"}:
        return min(base, 0.15)
    if med_key == "MAC_RATE":
        return min(base, 0.05)
    return base


def case_has_medication(caseid: int, cfg: PipelineConfig) -> bool:
    med_df = fetch_medication_frame(caseid, cfg.med_check_interval_sec)
    if med_df is None:
        return False
    for med_key, cands in medication_track_candidates().items():
        col = resolve_column(med_df, cands)
        if col is None:
            continue
        s = pd.to_numeric(med_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        diff = s.diff().fillna(0)
        if med_key.endswith("_RATE"):
            th = _rate_delta_threshold_for_med_key(med_key, cfg)
            if (s.abs() > 0).any() or (diff.abs() >= th * 0.5).any():
                return True
        else:
            if (diff >= cfg.vol_delta_threshold * 0.5).any():
                return True
    return False


def stage1_group_and_filter(cfg: PipelineConfig) -> pd.DataFrame:
    print(">>> Stage 1: load clinical data, classify surgery groups, and filter invalid cases")
    df = load_clinical_table(cfg.clinical_csv)

    if "ane_dur" in df.columns:
        dur = pd.to_numeric(df["ane_dur"], errors="coerce")
        # Keep rows with unknown duration (NaN) for multi-source datasets where
        # certain sources may not provide anesthesia duration.
        df = df[dur.isna() | (dur >= cfg.anes_dur_min)].copy()

    if cfg.max_cases > 0:
        df = df.head(cfg.max_cases).copy()

    if "department" in df.columns:
        dept = df["department"].fillna("Unknown")
    else:
        dept = pd.Series(["Unknown"] * len(df), index=df.index)
    if "opname" in df.columns:
        opname = df["opname"].fillna("Unknown surgery")
    else:
        opname = pd.Series(["Unknown surgery"] * len(df), index=df.index)

    if cfg.department_include.strip():
        keys = [k.strip().lower() for k in cfg.department_include.split(",") if k.strip()]
        if keys:
            dep_text = dept.astype(str).str.lower()
            keep_mask = dep_text.apply(lambda x: any(k in x for k in keys))
            before_n = len(df)
            df = df[keep_mask].copy()
            dept = dept[keep_mask]
            opname = opname[keep_mask]
            print(f"  - department filter ({cfg.department_include}): {len(df)}/{before_n} kept")

    df["surgery_group"] = [infer_surgery_group(str(d), str(o)) for d, o in zip(dept, opname)]
    if cfg.keep_source_duplicate_rows and "source_dataset" in df.columns:
        df = df.drop_duplicates(subset=["caseid", "source_dataset"], keep="first").reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["caseid"], keep="first").reset_index(drop=True)

    if cfg.anchor_mode in ("arrdb", "hybrid", "periodic"):
        valid_df = df.copy()
    elif cfg.skip_medication_filter:
        valid_df = df.copy()
    else:
        valid_mask: List[bool] = []
        total = len(df)
        for i, caseid in enumerate(df["caseid"].tolist(), start=1):
            if i % 20 == 0 or i == total:
                print(f"  - medication filter progress: {i}/{total}")
            valid_mask.append(case_has_medication(caseid, cfg))
        valid_df = df[pd.Series(valid_mask, index=df.index)].copy()

    os.makedirs(cfg.group_root, exist_ok=True)
    for group, gdf in valid_df.groupby("surgery_group"):
        out_dir = os.path.join(cfg.group_root, group)
        os.makedirs(out_dir, exist_ok=True)
        gdf[["caseid"]].to_csv(os.path.join(out_dir, "caseids.csv"), index=False)
        gdf.to_csv(os.path.join(out_dir, "clinical_subset.csv"), index=False)

    print(f"Stage 1 done: {len(valid_df)} valid cases")
    return valid_df


def fetch_case_frame(caseid: int, interval_sec: float, cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    tracks = all_track_candidates()
    try:
        vf = vitaldb.VitalFile(caseid, track_names=tracks)
        df = vf.to_pandas(tracks, interval_sec)
    except Exception as e:
        print(f"  - case {caseid} load failed: {e}")
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df["Time"] = np.arange(len(df), dtype=float) * interval_sec
    _normalize_mbp_unit_if_needed(df, caseid=caseid, cfg=cfg)
    return df


def _normalize_mbp_unit_if_needed(df: pd.DataFrame, caseid: Optional[int], cfg: Optional[PipelineConfig]) -> None:
    if cfg is not None and not cfg.enable_mbp_unit_fix:
        return
    mbp_col = resolve_vital_column(df, "MBP")
    if mbp_col is None:
        return
    s = pd.to_numeric(df[mbp_col], errors="coerce")
    # Only use physiologically plausible positive values for unit inference.
    valid = s[(s > 0) & (s < 300)].dropna()
    if valid.empty:
        return

    threshold = cfg.mbp_kpa_threshold if cfg is not None else 20.0
    factor = cfg.mbp_kpa_to_mmhg_factor if cfg is not None else 7.50062

    # If most valid MBP values are below threshold, data is likely logged in kPa.
    low_ratio = float((valid < threshold).mean())
    median_v = float(valid.median())
    q90_v = float(valid.quantile(0.90))
    likely_kpa = (low_ratio >= 0.8) or (median_v < threshold and q90_v < threshold * 1.5)
    if likely_kpa:
        df[mbp_col] = s * factor
        df["__mbp_unit_converted__"] = 1


def _compute_smoothed_rate_for_vol_anchor(
    value_series: pd.Series,
    time_series: pd.Series,
    anchor_idx: int,
    lookback_sec: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    sub = pd.DataFrame({"t": time_series.iloc[: anchor_idx + 1], "v": value_series.iloc[: anchor_idx + 1]}).dropna()
    if len(sub) < 2:
        return None, None, None, None, None, None

    t_now = float(sub.iloc[-1]["t"])
    v_now = float(sub.iloc[-1]["v"])
    target_t = t_now - lookback_sec

    hist = sub[sub["t"] <= target_t]
    if not hist.empty:
        ref = hist.iloc[-1]
    else:
        ref = sub.iloc[0]

    t_ref = float(ref["t"])
    v_ref = float(ref["v"])
    dt = t_now - t_ref
    if dt <= 0:
        return None, None, t_ref, t_now, v_ref, v_now
    dv = v_now - v_ref
    rate = (dv / dt) * 3600.0
    return rate, dt, t_ref, t_now, v_ref, v_now


def _resolve_column_case_insensitive(df: pd.DataFrame, candidates: Sequence[str], explicit: str = "") -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    col_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _parse_time_to_sec(v: Any) -> Optional[float]:
    if not is_valid(v):
        return None
    try:
        return float(v)
    except Exception:
        pass

    s = str(v).strip()
    if not s:
        return None

    # Support HH:MM:SS(.ms) and MM:SS(.ms)
    if ":" in s:
        parts = s.split(":")
        if len(parts) in (2, 3):
            try:
                nums = [float(x) for x in parts]
                if len(nums) == 2:
                    return nums[0] * 60.0 + nums[1]
                return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
            except Exception:
                pass

    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _load_arrdb_annotation(caseid: int, cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    ann_file = os.path.join(cfg.arrdb_annotation_dir, f"Annotation_file_{caseid}.csv")
    if not os.path.exists(ann_file):
        return None
    try:
        df = pd.read_csv(ann_file)
    except Exception as e:
        print(f"  - case {caseid} arrdb annotation load failed: {e}")
        return None
    if df is None or df.empty:
        return None
    return df


def find_arrdb_anchors(caseid: int, df_case: pd.DataFrame, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    ann_df = _load_arrdb_annotation(caseid, cfg)
    if ann_df is None:
        return []

    time_col = _resolve_column_case_insensitive(
        ann_df,
        ARRDB_TIME_COL_CANDIDATES,
        explicit=cfg.arrdb_time_column.strip(),
    )
    if time_col is None:
        # Try first numeric-like column as a fallback.
        for c in ann_df.columns:
            test = pd.to_numeric(ann_df[c], errors="coerce")
            if test.notna().sum() >= max(1, int(len(ann_df) * 0.5)):
                time_col = str(c)
                break
    if time_col is None:
        print(f"  - case {caseid} arrdb annotation skipped: time column not found")
        return []

    label_col = _resolve_column_case_insensitive(
        ann_df,
        ARRDB_LABEL_COL_CANDIDATES,
        explicit=cfg.arrdb_label_column.strip(),
    )
    if label_col is None:
        # Pick first non-time object-like column.
        for c in ann_df.columns:
            if str(c) == str(time_col):
                continue
            if ann_df[c].dtype == object:
                label_col = str(c)
                break

    max_t = None
    if "Time" in df_case.columns:
        tvals = pd.to_numeric(df_case["Time"], errors="coerce").dropna()
        if not tvals.empty:
            max_t = float(tvals.max())

    raw_events: List[Dict[str, Any]] = []
    prev_label = None
    for ridx, row in ann_df.iterrows():
        t_sec = _parse_time_to_sec(row.get(time_col))
        if t_sec is None or t_sec < 0:
            continue
        if max_t is not None and t_sec > max_t:
            continue

        label = ""
        if label_col is not None and is_valid(row.get(label_col)):
            label = str(row.get(label_col)).strip()
        label_norm = re.sub(r"\s+", " ", label.lower()) if label else ""
        if (not cfg.arrdb_keep_normal) and label_norm in ARRDB_NORMAL_LABELS:
            prev_label = label or prev_label
            continue

        if not label:
            label = "arrhythmia_event"

        raw_events.append(
            {
                "time_sec": float(t_sec),
                "medication_key": "ARR_EVENT",
                "track": f"ARRDB/{label_col or 'annotation'}",
                "delta": 0.0,
                "before": prev_label,
                "after": label,
                "prev_time_sec": None,
                "dt_sec": None,
                "inferred_rate_ml_per_h": None,
                "smoothed_rate_ml_per_h": None,
                "smoothed_dt_sec": None,
                "smoothed_ref_time_sec": None,
                "smoothed_ref_volume_ml": None,
                "smoothed_current_volume_ml": None,
                "smoothed_delta_volume_ml": None,
                "anchor_source": "arrdb",
                "arrhythmia_label": label,
                "annotation_row_id": int(ridx),
            }
        )
        prev_label = label

    raw_events.sort(key=lambda x: x["time_sec"])
    if not raw_events:
        return []

    deduped: List[Dict[str, Any]] = []
    for event in raw_events:
        if not deduped:
            deduped.append(event)
            continue
        last = deduped[-1]
        if (
            abs(float(event["time_sec"]) - float(last["time_sec"])) < cfg.min_anchor_gap_sec
            and str(event.get("arrhythmia_label", "")) == str(last.get("arrhythmia_label", ""))
        ):
            continue
        deduped.append(event)
    return deduped


def find_periodic_anchors(df: pd.DataFrame, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    if "Time" not in df.columns:
        return []
    tvals = pd.to_numeric(df["Time"], errors="coerce").dropna()
    if tvals.empty:
        return []
    max_t = float(tvals.max())
    step = max(1.0, float(cfg.periodic_anchor_step_sec))
    start_t = max(float(cfg.periodic_anchor_start_sec), float(cfg.window_sec))
    if start_t > max_t:
        return []

    anchors: List[Dict[str, Any]] = []
    t = start_t
    while t <= max_t:
        anchors.append(
            {
                "time_sec": float(t),
                "medication_key": "UNLABELED_EVENT",
                "track": "TIME/PERIODIC",
                "delta": 0.0,
                "before": None,
                "after": None,
                "prev_time_sec": None,
                "dt_sec": None,
                "inferred_rate_ml_per_h": None,
                "smoothed_rate_ml_per_h": None,
                "smoothed_dt_sec": None,
                "smoothed_ref_time_sec": None,
                "smoothed_ref_volume_ml": None,
                "smoothed_current_volume_ml": None,
                "smoothed_delta_volume_ml": None,
                "anchor_source": "periodic",
            }
        )
        t += step
    return anchors


def is_probable_setup_rate_anchor(anchor: Dict[str, Any], cfg: PipelineConfig) -> bool:
    if not cfg.skip_setup_rate_anchors:
        return False
    med_key = str(anchor.get("medication_key", ""))
    if not med_key.endswith("_RATE"):
        return False

    before = anchor.get("before")
    after = anchor.get("after")
    delta = anchor.get("delta")
    time_sec = anchor.get("time_sec")
    if before is None or after is None or delta is None:
        return False

    try:
        before_f = float(before)
        after_f = float(after)
        delta_f = float(delta)
        t_f = float(time_sec) if time_sec is not None else 0.0
    except (TypeError, ValueError):
        return False

    if abs(before_f) > float(cfg.setup_rate_before_abs_max):
        return False
    if abs(delta_f) < float(cfg.setup_rate_delta_threshold):
        return False
    if after_f < float(cfg.setup_rate_after_threshold):
        return False
    if float(cfg.setup_rate_early_window_sec) > 0 and t_f > float(cfg.setup_rate_early_window_sec):
        return False
    return True


def find_anchors(df: pd.DataFrame, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    time_series = pd.to_numeric(df["Time"], errors="coerce") if "Time" in df.columns else None
    prev_time_series = time_series.shift(1) if time_series is not None else None
    dt_series = (time_series - prev_time_series) if time_series is not None else None

    for med_key, cands in medication_track_candidates().items():
        col = resolve_column(df, cands)
        if col is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            continue
        diff = s.diff()
        prev = s.shift(1)
        if med_key.endswith("_RATE"):
            th = _rate_delta_threshold_for_med_key(med_key, cfg)
            idx = diff[(diff.abs() >= th) & diff.notna()].index
        else:
            idx = diff[(diff >= cfg.vol_delta_threshold) & diff.notna()].index
        for i in idx:
            t = float(df.at[i, "Time"])
            d = float(diff.at[i]) if pd.notna(diff.at[i]) else 0.0
            before = float(prev.at[i]) if pd.notna(prev.at[i]) else None
            after = float(s.at[i]) if pd.notna(s.at[i]) else None
            prev_t = float(prev_time_series.at[i]) if prev_time_series is not None and pd.notna(prev_time_series.at[i]) else None
            dt_sec = float(dt_series.at[i]) if dt_series is not None and pd.notna(dt_series.at[i]) else None
            inferred_rate_ml_per_h = None
            if med_key.endswith("_VOL") and dt_sec is not None and dt_sec > 0:
                inferred_rate_ml_per_h = (d / dt_sec) * 3600.0

            smoothed_rate_ml_per_h = None
            smoothed_dt_sec = None
            smoothed_ref_time_sec = None
            smoothed_ref_volume_ml = None
            smoothed_current_volume_ml = None
            smoothed_delta_volume_ml = None
            if med_key.endswith("_VOL") and time_series is not None:
                (
                    smoothed_rate_ml_per_h,
                    smoothed_dt_sec,
                    smoothed_ref_time_sec,
                    _,
                    smoothed_ref_volume_ml,
                    smoothed_current_volume_ml,
                ) = _compute_smoothed_rate_for_vol_anchor(
                    value_series=s,
                    time_series=time_series,
                    anchor_idx=int(i),
                    lookback_sec=cfg.vol_rate_lookback_sec,
                )
                if (
                    smoothed_current_volume_ml is not None
                    and smoothed_ref_volume_ml is not None
                ):
                    smoothed_delta_volume_ml = smoothed_current_volume_ml - smoothed_ref_volume_ml

            # Hard-kill common TCI/init pseudo-anchors, e.g. 0 -> 400 setup jumps.
            if before is not None and abs(before) <= 1e-6 and d >= 100.0:
                continue

            # Ignore tiny background volume drift that is usually not a real decision event.
            if med_key.endswith("_VOL") and smoothed_delta_volume_ml is not None:
                if smoothed_delta_volume_ml < 0.5:
                    continue

            event = {
                "time_sec": t,
                "medication_key": med_key,
                "track": col,
                "delta": d,
                "before": before,
                "after": after,
                "prev_time_sec": prev_t,
                "dt_sec": dt_sec,
                "inferred_rate_ml_per_h": inferred_rate_ml_per_h,
                "smoothed_rate_ml_per_h": smoothed_rate_ml_per_h,
                "smoothed_dt_sec": smoothed_dt_sec,
                "smoothed_ref_time_sec": smoothed_ref_time_sec,
                "smoothed_ref_volume_ml": smoothed_ref_volume_ml,
                "smoothed_current_volume_ml": smoothed_current_volume_ml,
                "smoothed_delta_volume_ml": smoothed_delta_volume_ml,
                "anchor_source": "medication",
            }
            if is_probable_setup_rate_anchor(event, cfg):
                continue
            anchors.append(event)

    anchors.sort(key=lambda x: x["time_sec"])
    if not anchors:
        return anchors

    # De-duplicate nearby anchors to avoid dense duplicates from same intervention
    deduped: List[Dict[str, Any]] = []
    last_t = None
    for event in anchors:
        t = event["time_sec"]
        if last_t is None or abs(t - last_t) >= cfg.min_anchor_gap_sec:
            deduped.append(event)
            last_t = t
    return deduped


def trend_label(slope: float) -> str:
    if slope <= -0.2:
        return "rapidly decreased"
    if slope <= -0.05:
        return "decreased"
    if slope >= 0.2:
        return "rapidly increased"
    if slope >= 0.05:
        return "increased"
    return "stable"


def summarize_series(series: pd.Series, vital_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    s = _physio_filter_series(series, key=vital_key).dropna()
    if len(s) < 10:
        return None
    n = len(s)
    k = max(5, min(20, n // 10))
    start = float(s.iloc[:k].mean())
    end = float(s.iloc[-k:].mean())
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, s.values.astype(float), 1)[0]) if n >= 2 else 0.0
    return {
        "start": start,
        "end": end,
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
        "slope": slope,
    }


def build_trend_text(vital_key: str, summary: Optional[Dict[str, float]]) -> str:
    if summary is None:
        return "insufficient valid data"
    unit = VITAL_UNIT.get(vital_key, "")
    trend = trend_label(summary["slope"])
    return (
        f"from {summary['start']:.1f}{unit} {trend} to {summary['end']:.1f}{unit}; "
        f"mean {summary['mean']:.1f}{unit}, range [{summary['min']:.1f}, {summary['max']:.1f}]"
    )


def _last_window_mean(series: pd.Series, n_points: int = 60, vital_key: Optional[str] = None) -> Optional[float]:
    s = _physio_filter_series(series, key=vital_key).dropna()
    if s.empty:
        return None
    n = min(n_points, len(s))
    return float(s.iloc[-n:].median())


def _infer_series_step_seconds(tvals: np.ndarray) -> float:
    if len(tvals) < 2:
        return 1.0
    diffs = np.diff(tvals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _tail_condition_duration_sec(
    tvals: np.ndarray,
    mask_vals: np.ndarray,
) -> float:
    if len(tvals) == 0 or len(mask_vals) == 0:
        return 0.0
    if len(tvals) != len(mask_vals):
        return 0.0
    if not bool(mask_vals[-1]):
        return 0.0

    j = len(mask_vals) - 1
    while j >= 0 and bool(mask_vals[j]):
        j -= 1
    start_idx = j + 1
    step_sec = _infer_series_step_seconds(tvals)
    if start_idx == 0:
        return float(max(step_sec, tvals[-1] - tvals[0] + step_sec))
    return float(max(step_sec, tvals[-1] - tvals[start_idx] + step_sec))


def _tail_persistence_by_vital(
    df_window: pd.DataFrame,
    vital_key: str,
    predicate: Any,
) -> float:
    if "Time" not in df_window.columns:
        return 0.0
    col = resolve_vital_column(df_window, vital_key)
    if col is None:
        return 0.0
    t = pd.to_numeric(df_window["Time"], errors="coerce")
    s = _physio_filter_series(pd.to_numeric(df_window[col], errors="coerce"), key=vital_key)
    valid = t.notna() & s.notna()
    if not valid.any():
        return 0.0
    tvals = t[valid].to_numpy(dtype=float)
    sval = s[valid]
    try:
        mask_vals = np.asarray(predicate(sval), dtype=bool)
    except Exception:
        return 0.0
    return _tail_condition_duration_sec(tvals=tvals, mask_vals=mask_vals)


def _safe_get_series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    col = resolve_vital_column(df, key)
    if col is None:
        return None
    return df[col]


def _median_in_time_window(
    df: pd.DataFrame,
    vital_key: str,
    start_t: float,
    end_t: float,
) -> Optional[float]:
    if "Time" not in df.columns:
        return None
    col = resolve_vital_column(df, vital_key)
    if col is None:
        return None
    t = pd.to_numeric(df["Time"], errors="coerce")
    mask = (t >= float(start_t)) & (t <= float(end_t))
    if not mask.any():
        return None
    s = _physio_filter_series(pd.to_numeric(df.loc[mask, col], errors="coerce"), key=vital_key).dropna()
    if s.empty:
        return None
    return float(s.median())


def build_baseline_comparison(
    df_case: pd.DataFrame,
    df_window: pd.DataFrame,
    anchor_time_sec: float,
) -> Dict[str, Optional[float]]:
    # Baseline from 20~10 min before anchor (when available).
    base_end = max(0.0, float(anchor_time_sec) - 600.0)
    base_start = max(0.0, float(anchor_time_sec) - 1200.0)

    mbp_baseline = _median_in_time_window(df_case, "MBP", base_start, base_end)
    hr_baseline = _median_in_time_window(df_case, "HR", base_start, base_end)

    mbp_current = None
    hr_current = None
    mbp_s = _safe_get_series(df_window, "MBP")
    hr_s = _safe_get_series(df_window, "HR")
    if mbp_s is not None:
        mbp_current = _last_window_mean(mbp_s, vital_key="MBP")
    if hr_s is not None:
        hr_current = _last_window_mean(hr_s, vital_key="HR")

    map_drop_pct = None
    hr_change_pct = None
    if mbp_baseline is not None and mbp_current is not None and mbp_baseline > 0:
        map_drop_pct = float((mbp_baseline - mbp_current) / mbp_baseline * 100.0)
    if hr_baseline is not None and hr_current is not None and hr_baseline > 0:
        hr_change_pct = float((hr_current - hr_baseline) / hr_baseline * 100.0)

    return {
        "baseline_window_start_sec": float(base_start),
        "baseline_window_end_sec": float(base_end),
        "MAP_baseline_mmhg": mbp_baseline,
        "MAP_current_mmhg": mbp_current,
        "MAP_drop_from_baseline_pct": map_drop_pct,
        "HR_baseline_bpm": hr_baseline,
        "HR_current_bpm": hr_current,
        "HR_change_from_baseline_pct": hr_change_pct,
    }


def build_clinical_assessment(
    df_window: pd.DataFrame,
    anchor: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    hr_s = _safe_get_series(df_window, "HR")
    mbp_s = _safe_get_series(df_window, "MBP")
    spo2_s = _safe_get_series(df_window, "SPO2")
    bis_s = _safe_get_series(df_window, "BIS")

    hr_last = _last_window_mean(hr_s, vital_key="HR") if hr_s is not None else None
    mbp_last = _last_window_mean(mbp_s, vital_key="MBP") if mbp_s is not None else None
    spo2_last = _last_window_mean(spo2_s, vital_key="SPO2") if spo2_s is not None else None
    bis_last = _last_window_mean(bis_s, vital_key="BIS") if bis_s is not None else None

    decision_windows = {
        "critical_window_sec": float(ANES_THRESHOLDS["critical_window_sec"]),
        "hemodynamic_window_sec": float(ANES_THRESHOLDS["hemodynamic_window_sec"]),
        "slow_trend_window_sec": float(ANES_THRESHOLDS["slow_trend_window_sec"]),
    }
    map_severe_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="MBP",
        predicate=lambda s: s < ANES_THRESHOLDS["map_severe_hypotension_mmhg"],
    )
    map_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="MBP",
        predicate=lambda s: s < ANES_THRESHOLDS["map_hypotension_mmhg"],
    )
    hr_tachy_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="HR",
        predicate=lambda s: s > ANES_THRESHOLDS["hr_tachycardia_bpm"],
    )
    hr_brady_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="HR",
        predicate=lambda s: s < ANES_THRESHOLDS["hr_bradycardia_bpm"],
    )
    spo2_severe_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SPO2",
        predicate=lambda s: s < ANES_THRESHOLDS["spo2_severe_low_pct"],
    )
    spo2_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SPO2",
        predicate=lambda s: s < ANES_THRESHOLDS["spo2_low_pct"],
    )
    bis_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="BIS",
        predicate=lambda s: s > ANES_THRESHOLDS["bis_light"],
    )
    bis_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="BIS",
        predicate=lambda s: s < ANES_THRESHOLDS["bis_deep"],
    )

    flags: List[str] = []
    contextual_interpretation: List[str] = []
    map_absolute_triggered = False
    map_relative_triggered = False

    if mbp_last is not None:
        if map_severe_persist_sec >= decision_windows["critical_window_sec"]:
            flags.append(
                f"重度低血压（MAP < 55 mmHg，持续约{map_severe_persist_sec:.0f}s，>= {decision_windows['critical_window_sec']:.0f}s）"
            )
            map_absolute_triggered = True
        elif map_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"低血压（MAP < 65 mmHg，持续约{map_low_persist_sec:.0f}s，>= {decision_windows['hemodynamic_window_sec']:.0f}s）"
            )
            map_absolute_triggered = True
        elif mbp_last < ANES_THRESHOLDS["map_hypotension_mmhg"]:
            flags.append(
                f"MAP低于65但持续不足{decision_windows['hemodynamic_window_sec']:.0f}s（早期预警）"
            )

    if hr_last is not None:
        if hr_tachy_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"心动过速（HR > 100 bpm，持续约{hr_tachy_persist_sec:.0f}s）"
            )
        elif hr_brady_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"心动过缓（HR < 50 bpm，持续约{hr_brady_persist_sec:.0f}s）"
            )

    if spo2_last is not None:
        if spo2_severe_persist_sec >= decision_windows["critical_window_sec"]:
            flags.append(
                f"重度低氧血症（SpO2 < 90%，持续约{spo2_severe_persist_sec:.0f}s）"
            )
        elif spo2_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"血氧下降（SpO2 < 94%，持续约{spo2_low_persist_sec:.0f}s）"
            )

    if bis_last is not None:
        if bis_high_persist_sec >= decision_windows["slow_trend_window_sec"]:
            flags.append(
                f"BIS持续偏高（>60，持续约{bis_high_persist_sec:.0f}s，需结合刺激与EMG）"
            )
        elif bis_low_persist_sec >= decision_windows["slow_trend_window_sec"]:
            flags.append(
                f"BIS持续偏低（<40，持续约{bis_low_persist_sec:.0f}s，需结合低灌注排查）"
            )
        elif bis_last > ANES_THRESHOLDS["bis_light"] or bis_last < ANES_THRESHOLDS["bis_deep"]:
            flags.append(
                f"BIS短时偏离（持续不足{decision_windows['slow_trend_window_sec']:.0f}s，不单独作为给药依据）"
            )
    else:
        flags.append("BIS 数据缺失（优先依据MAP/HR/SpO2趋势和手术刺激评估）")

    map_drop_pct = None
    if baseline_comparison:
        map_drop_pct = baseline_comparison.get("MAP_drop_from_baseline_pct")
        if map_drop_pct is not None and float(map_drop_pct) >= ANES_THRESHOLDS["map_relative_drop_pct"]:
            map_relative_triggered = True
            flags.append(f"MAP较基线下降明显（{float(map_drop_pct):.1f}%）")

    if map_absolute_triggered:
        flags.append("MAP绝对阈值触发：作为器官灌注底线优先处理")
    if map_relative_triggered:
        flags.append("MAP相对下降触发：用于个体化风险分层")

    med_key = str(anchor.get("medication_key", ""))
    anchor_source = str(anchor.get("anchor_source", "medication"))
    delta = _safe_float(anchor.get("delta"))
    intervention_hint = "未触发特定干预启发式。"
    if med_key == "PHE_RATE":
        intervention_hint = "去氧肾上腺素通常用于血管扩张相关低血压且心率不低的场景；若明显心动过缓需谨慎。"
    elif med_key in {"EPH_VOL", "EPH_RATE"}:
        intervention_hint = "麻黄碱更适合低血压合并低心率；若已明显心动过速应避免继续加量。"
    elif med_key == "NOR_RATE":
        intervention_hint = "去甲肾上腺素用于难治性血管扩张性低血压；疑似低容量时应先扩容后升压。"
    elif med_key == "EPI_RATE":
        intervention_hint = "肾上腺素主要用于抢救级循环衰竭场景；不宜作为常规轻中度低血压首选。"
    elif med_key in {"NTG_VOL", "NTG_RATE"}:
        intervention_hint = "硝酸甘油用于缺血/高血压/肺水肿场景；MAP偏低时可显著恶化循环，需谨慎。"
    elif med_key in {"MIL_VOL", "MIL_RATE"}:
        intervention_hint = "米力农可改善低心排与肺高压，但有扩血管效应；低血压未纠正前避免升级。"
    elif med_key in {"ATRO_VOL", "ATRO_RATE"}:
        intervention_hint = "阿托品用于有血流动力学意义的严重心动过缓；非症状性慢心率不应机械使用。"
    elif med_key == "PPF20_VOL":
        intervention_hint = "丙泊酚调整需先看灌注：当MAP已低时应先稳循环，再评估是否继续加深麻醉。"
    elif med_key in {"REMI_VOL", "REMI_RATE", "RFTN20_VOL", "RFTN50_VOL", "RFTN20_RATE", "RFTN50_RATE"}:
        intervention_hint = "阿片类调整常用于手术刺激上冲控制；需同步监测呼吸抑制、心动过缓与低血压。"
    elif med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE", "MAC_RATE"}:
        intervention_hint = "吸入麻醉浓度调整应与MAP/HR/SpO2共同判断，避免仅凭BIS单指标机械加深或减浅。"
    elif anchor_source == "arrdb" or med_key == "ARR_EVENT":
        arr_label = str(anchor.get("arrhythmia_label", anchor.get("after", ""))).strip()
        if arr_label:
            flags.append(f"心律事件标注：{arr_label}")
            severe_kw = ("vf", "vt", "ventricular", "asystole", "torsade", "af with rvr")
            if any(k in arr_label.lower() for k in severe_kw):
                flags.append("严重心律失常风险（arrdb标注）")
        intervention_hint = "该锚点来自 arrdb 心律标注事件，建议结合血流动力学与麻醉深度判断是否需要节律/循环干预。"
    elif anchor_source == "periodic" or med_key == "UNLABELED_EVENT":
        intervention_hint = "该锚点来自无标记时间采样，用于让模型基于体征上下文学习临床推理。"

    if mbp_last is not None and mbp_last < ANES_THRESHOLDS["map_hypotension_mmhg"]:
        if hr_last is not None and hr_last < ANES_THRESHOLDS["hr_bradycardia_bpm"]:
            contextual_interpretation.append("MAP低+HR低：优先考虑麻醉相关抑制或传导问题，避免单纯加深麻醉。")
        if hr_last is not None and hr_last > ANES_THRESHOLDS["hr_tachycardia_bpm"]:
            contextual_interpretation.append("MAP低+HR高：需警惕低容量/失血或应激反应，不应仅按BIS加药。")
        if med_key in ANESTHETIC_DEPTH_MED_KEYS and delta is not None and delta > 0:
            contextual_interpretation.append("MAP低+正在加深麻醉：符合药理性低血压风险，应先守住灌注底线。")
        if med_key in VASOACTIVE_MED_KEYS:
            contextual_interpretation.append("MAP低+升压药背景：提示可能为难治性低血压，需要复核病因与容量状态。")

    if bis_last is not None:
        if (
            bis_last > ANES_THRESHOLDS["bis_light"]
            and mbp_last is not None
            and mbp_last < ANES_THRESHOLDS["map_hypotension_mmhg"]
        ):
            contextual_interpretation.append("BIS高合并低血压时，不能机械加深麻醉，应先稳定循环。")
        elif (
            bis_last > ANES_THRESHOLDS["bis_light"]
            and hr_last is not None
            and hr_last > ANES_THRESHOLDS["hr_tachycardia_bpm"]
        ):
            contextual_interpretation.append("BIS高+HR/MAP上冲更像镇痛不足或镇静镇痛双不足，需要联合调整。")
        if (
            bis_last < ANES_THRESHOLDS["bis_deep"]
            and mbp_last is not None
            and mbp_last < ANES_THRESHOLDS["map_hypotension_mmhg"]
        ):
            contextual_interpretation.append("BIS低+MAP低提示可能过深麻醉并低灌注，宜减浅麻醉并支持循环。")

    severity = "low"
    if any(("重度" in f) or ("严重" in f) for f in flags):
        severity = "high"
    elif flags:
        severity = "moderate"

    return {
        "recent_state_mean": {
            "MAP_mmhg": mbp_last,
            "HR_bpm": hr_last,
            "SpO2_pct": spo2_last,
            "BIS": bis_last,
        },
        "baseline_comparison": baseline_comparison if baseline_comparison is not None else {},
        "risk_flags": flags,
        "contextual_interpretation": contextual_interpretation,
        "risk_level": severity,
        "map_policy": {
            "absolute_primary": True,
            "relative_layered": True,
            "absolute_triggered": map_absolute_triggered,
            "relative_triggered": map_relative_triggered,
        },
        "persistence_seconds": {
            "map_lt_55": map_severe_persist_sec,
            "map_lt_65": map_low_persist_sec,
            "hr_gt_100": hr_tachy_persist_sec,
            "hr_lt_50": hr_brady_persist_sec,
            "spo2_lt_90": spo2_severe_persist_sec,
            "spo2_lt_94": spo2_low_persist_sec,
            "bis_gt_60": bis_high_persist_sec,
            "bis_lt_40": bis_low_persist_sec,
        },
        "decision_windows_sec": decision_windows,
        "intervention_consideration": intervention_hint,
        "missing_data_guidance": (
            "If BIS data is missing, infer anesthesia depth from autonomic signs "
            "(HR/MBP trends) and surgical stimulation context."
        ),
        "drug_reference": DRUG_REFERENCE,
    }

def generate_window_plot(
    df_window: pd.DataFrame,
    caseid: int,
    anchor_time: float,
    image_root: str,
    suffix: str,
) -> Optional[str]:
    available: List[Tuple[str, str, pd.Series, pd.Series]] = []
    for vital_key in VITAL_TRACK_CANDIDATES.keys():
        col = resolve_vital_column(df_window, vital_key)
        if col is None:
            continue
        raw = pd.to_numeric(df_window[col], errors="coerce")
        filtered = _physio_filter_series(raw, key=vital_key)
        valid = filtered.dropna()
        if valid.empty:
            continue
        available.append((vital_key, col, raw, filtered))

    if not available:
        return None

    os.makedirs(image_root, exist_ok=True)
    out_path = os.path.join(image_root, f"case_{caseid}_T{int(anchor_time)}_{suffix}.png")
    fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    x = df_window["Time"].to_numpy()
    for ax, (vital_key, col, raw, filtered) in zip(axes, available):
        y = filtered.to_numpy()
        valid_mask = np.isfinite(y)
        valid_n = int(valid_mask.sum())
        total_n = int(len(y))
        valid_ratio = (valid_n / total_n) if total_n > 0 else 0.0
        label = f"{VITAL_DISPLAY.get(vital_key, vital_key)} (valid {valid_n}/{total_n}, {valid_ratio:.0%})"

        if valid_n >= 2:
            # Use lightly interpolated display curve so sparse missing points still show continuity.
            y_disp = pd.Series(y).interpolate(limit_direction="both").to_numpy()
            ax.plot(x, y_disp, linewidth=1.2, label=label)
        elif valid_n == 1:
            idx = int(np.where(valid_mask)[0][0])
            ax.scatter([x[idx]], [y[idx]], s=12, label=label + " single-point")
        else:
            continue

        if vital_key == "MBP":
            ax.axhline(65, linestyle="--", color="gray", linewidth=1.0, alpha=0.6)
        ax.axvline(anchor_time, linestyle="-.", color="purple", linewidth=1.5, label="Intervention")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel(VITAL_DISPLAY.get(vital_key, vital_key), fontsize=8)

    axes[-1].set_xlabel("Time (sec)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def collect_preop_context(row: pd.Series) -> List[str]:
    keys = ["dx", "opdiag", "opdiag1", "preop_dx", "comorbidity", "allergy", "lab"]
    values: List[str] = []
    for k in keys:
        if k in row and is_valid(row[k]):
            values.append(str(row[k]).strip())
    if not values:
        values.append("No obvious preop comorbidity/lab abnormality in source table")
    return values[:5]


def infer_intervention_type(anchor: Dict[str, Any], cfg: PipelineConfig) -> str:
    if str(anchor.get("anchor_source", "")) == "arrdb" or str(anchor.get("medication_key", "")) == "ARR_EVENT":
        return "arrhythmia_event"
    if str(anchor.get("anchor_source", "")) == "periodic" or str(anchor.get("medication_key", "")) == "UNLABELED_EVENT":
        return "unlabeled_context_snapshot"

    med_key = str(anchor.get("medication_key", ""))
    if not med_key.endswith("_VOL"):
        return "rate_adjustment"

    smoothed_rate = anchor.get("smoothed_rate_ml_per_h")
    smoothed_delta_ml = anchor.get("smoothed_delta_volume_ml")
    if med_key == "PPF20_VOL":
        if (
            smoothed_rate is not None
            and smoothed_delta_ml is not None
            and float(smoothed_rate) >= cfg.propofol_bolus_rate_threshold_ml_h
            and float(smoothed_delta_ml) >= cfg.propofol_bolus_min_delta_ml
        ):
            return "bolus_like_event"
    return "continuous_infusion"


def describe_intervention(anchor: Dict[str, Any], cfg: PipelineConfig) -> str:
    if str(anchor.get("anchor_source", "")) == "arrdb" or str(anchor.get("medication_key", "")) == "ARR_EVENT":
        label = str(anchor.get("arrhythmia_label", anchor.get("after", "arrhythmia_event"))).strip()
        return f"心律事件标注：{label}"
    if str(anchor.get("anchor_source", "")) == "periodic" or str(anchor.get("medication_key", "")) == "UNLABELED_EVENT":
        return "无标记时间采样锚点（用于上下文推理训练）"

    med_key = anchor["medication_key"]
    label = MEDICATION_DISPLAY.get(med_key, med_key)
    delta = float(anchor["delta"])
    before = anchor.get("before")
    after = anchor.get("after")
    dt_sec = anchor.get("dt_sec")
    inferred_rate_ml_per_h = anchor.get("inferred_rate_ml_per_h")
    smoothed_rate_ml_per_h = anchor.get("smoothed_rate_ml_per_h")
    smoothed_dt_sec = anchor.get("smoothed_dt_sec")
    smoothed_delta_volume_ml = anchor.get("smoothed_delta_volume_ml")

    if med_key.endswith("_VOL"):
        # 【补丁 2：丙泊酚推注与泵注的精准中文描述】
        if med_key == "PPF20_VOL" and smoothed_rate_ml_per_h is not None and smoothed_delta_volume_ml is not None:
            if float(smoothed_rate_ml_per_h) > cfg.propofol_bolus_rate_threshold_ml_h:
                return f"单次追加推注丙泊酚约 {smoothed_delta_volume_ml:.1f} mL"
            else:
                return f"丙泊酚静脉维持泵注平滑速率约 {smoothed_rate_ml_per_h:.2f} mL/h"

        # 回退给其他体积药物的记录
        smooth_text = ""
        if smoothed_rate_ml_per_h is not None and smoothed_dt_sec is not None:
            smooth_text = f"；{smoothed_dt_sec:.1f}s平滑窗口估算速率 {smoothed_rate_ml_per_h:.2f} mL/h"
        
        if before is None or after is None:
            if inferred_rate_ml_per_h is not None and dt_sec is not None:
                return f"{label}：累计量变化 {delta:+.3f} mL（{dt_sec:.1f}s内，瞬时估算速率 {inferred_rate_ml_per_h:.2f} mL/h{smooth_text}）"
            return f"{label}：累计量变化 {delta:+.3f} mL"
            
        if inferred_rate_ml_per_h is not None and dt_sec is not None:
            return (
                f"{label}：累计量 {before:.3f} -> {after:.3f} mL"
                f"（变化 {delta:+.3f} mL，{dt_sec:.1f}s内瞬时估算速率 {inferred_rate_ml_per_h:.2f} mL/h{smooth_text}）"
            )
            
        return f"{label}：累计量 {before:.3f} -> {after:.3f} mL（变化 {delta:+.3f} mL{smooth_text}）"

    if med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE"}:
        unit = "vol%"
        if before is None or after is None:
            return f"{label}：浓度变化 {delta:+.3f} {unit}"
        return f"{label}：{before:.3f} -> {after:.3f} {unit}（变化 {delta:+.3f} {unit}）"

    if med_key == "MAC_RATE":
        if before is None or after is None:
            return f"{label}：变化 {delta:+.3f}"
        return f"{label}：{before:.3f} -> {after:.3f}（变化 {delta:+.3f}）"

    if before is None or after is None:
        return f"{label}：速率变化 {delta:+.3f}"
    return f"{label}：{before:.3f} -> {after:.3f}（变化 {delta:+.3f}）"

def build_snapshot(
    row: pd.Series,
    surgery_group: str,
    anchor: Dict[str, Any],
    df_case: pd.DataFrame,
    df_window: pd.DataFrame,
    image_path: Optional[str],
    window_sec: int,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    trends: Dict[str, str] = {}
    stat_block: Dict[str, Dict[str, float]] = {}
    for vital_key, cands in VITAL_TRACK_CANDIDATES.items():
        col = resolve_vital_column(df_window, vital_key)
        summary = summarize_series(df_window[col], vital_key=vital_key) if col is not None else None
        trends[vital_key] = build_trend_text(vital_key, summary)
        if summary is not None:
            stat_block[vital_key] = summary

    age = first_valid(row, ["age"], "Unknown")
    sex = first_valid(row, ["sex", "gender"], "Unknown")
    bmi = first_valid(row, ["bmi"], "Unknown")
    asa = first_valid(row, ["asa"], "Unknown")
    department = first_valid(row, ["department"], "Unknown")
    opname = first_valid(row, ["opname"], "Unknown surgery")
    baseline_comparison = build_baseline_comparison(
        df_case=df_case,
        df_window=df_window,
        anchor_time_sec=float(anchor["time_sec"]),
    )
    clinical_assessment = build_clinical_assessment(
        df_window=df_window,
        anchor=anchor,
        baseline_comparison=baseline_comparison,
    )
    intervention_type = infer_intervention_type(anchor, cfg)
    tmp_snapshot_for_eval = {
        "clinical_assessment": clinical_assessment,
        "actual_intervention": describe_intervention(anchor, cfg),
        "anchor_detail": {
            "medication_key": anchor.get("medication_key"),
            "delta": anchor.get("delta"),
        },
    }
    miller_alignment = evaluate_vitaldb_vs_miller(tmp_snapshot_for_eval)
    actual_intervention_text = str(tmp_snapshot_for_eval["actual_intervention"])

    return {
        "patient_background": {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "asa": asa,
            "department": department,
            "surgery_group": surgery_group,
        },
        "preop_context": collect_preop_context(row),
        "surgery_type": opname,
        "intraop_stage": f"Intraoperative (relative timestamp: {int(anchor['time_sec'])} sec)",
        f"vital_trend_last_{int(window_sec / 60)}min": trends,
        "vital_stats": stat_block,
        "baseline_comparison": baseline_comparison,
        "clinical_assessment": clinical_assessment,
        "miller_alignment": miller_alignment,
        "actual_intervention": actual_intervention_text,
        "interpreted_intervention_type": intervention_type,
        "anchor_detail": {
            "anchor_source": anchor.get("anchor_source", "medication"),
            "track": anchor["track"],
            "medication_key": anchor["medication_key"],
            "arrhythmia_label": anchor.get("arrhythmia_label"),
            "annotation_row_id": anchor.get("annotation_row_id"),
            "time_sec": int(anchor["time_sec"]),
            "prev_time_sec": anchor.get("prev_time_sec"),
            "delta_time_sec": anchor.get("dt_sec"),
            "delta": float(anchor["delta"]),
            "inferred_rate_ml_per_h": anchor.get("inferred_rate_ml_per_h"),
            "smoothed_rate_ml_per_h": anchor.get("smoothed_rate_ml_per_h"),
            "smoothed_delta_time_sec": anchor.get("smoothed_dt_sec"),
            "smoothed_ref_time_sec": anchor.get("smoothed_ref_time_sec"),
            "smoothed_ref_volume_ml": anchor.get("smoothed_ref_volume_ml"),
            "smoothed_current_volume_ml": anchor.get("smoothed_current_volume_ml"),
            "smoothed_delta_volume_ml": anchor.get("smoothed_delta_volume_ml"),
            "intervention_type": intervention_type,
            "before": anchor.get("before"),
            "after": anchor.get("after"),
        },
        "unit_corrections": {
            "mbp_kpa_to_mmhg_applied": bool(pd.to_numeric(df_window.get("__mbp_unit_converted__", pd.Series([0])), errors="coerce").fillna(0).max() > 0)
        },
        "waveform_image_path": image_path if image_path else "",
    }

def stage2_extract_snapshots(cases_df: pd.DataFrame, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    print(f">>> Stage 2: detect anchors (mode={cfg.anchor_mode}) and build decision snapshots")
    records: List[Dict[str, Any]] = []
    total = len(cases_df)

    for i, (_, row) in enumerate(cases_df.iterrows(), start=1):
        caseid = int(row["caseid"])
        if i % 10 == 0 or i == total:
            print(f"  - stage2 progress: {i}/{total}")

        df_case = fetch_case_frame(caseid, cfg.signal_interval_sec, cfg)
        if df_case is None:
            continue

        if cfg.anchor_mode == "arrdb":
            anchors = find_arrdb_anchors(caseid=caseid, df_case=df_case, cfg=cfg)
        elif cfg.anchor_mode == "periodic":
            anchors = find_periodic_anchors(df_case=df_case, cfg=cfg)
        elif cfg.anchor_mode == "hybrid":
            anchors = find_anchors(df_case, cfg) + find_arrdb_anchors(caseid=caseid, df_case=df_case, cfg=cfg)
            anchors = sorted(anchors, key=lambda x: float(x.get("time_sec", 0.0)))
        else:
            anchors = find_anchors(df_case, cfg)
        if not anchors:
            continue

        for j, anchor in enumerate(anchors[: cfg.max_anchors_per_case], start=1):
            t = float(anchor["time_sec"])
            start_t = max(0.0, t - cfg.window_sec)
            df_window = df_case[(df_case["Time"] >= start_t) & (df_case["Time"] <= t)].copy()
            if len(df_window) < cfg.min_window_points:
                continue

            image_path = generate_window_plot(
                df_window=df_window,
                caseid=caseid,
                anchor_time=t,
                image_root=cfg.image_root,
                suffix=f"a{j}",
            )
            # 👇 这里补上 cfg=cfg
            snapshot = build_snapshot(
                row=row,
                surgery_group=str(row.get("surgery_group", "Other")),
                anchor=anchor,
                df_case=df_case,
                df_window=df_window,
                image_path=image_path,
                window_sec=cfg.window_sec,
                cfg=cfg  
            )
            records.append(
                {
                    "caseid": caseid,
                    "surgery_group": str(row.get("surgery_group", "Other")),
                    "snapshot": snapshot,
                    "llm_output": None,
                }
            )

    os.makedirs(os.path.dirname(cfg.snapshot_json), exist_ok=True)
    with open(cfg.snapshot_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Stage 2 done: {len(records)} snapshots")
    return records

def _fewshot_text_for_snapshot(snapshot: Dict[str, Any]) -> str:
    itype = str(snapshot.get("interpreted_intervention_type", "")).strip()
    if not itype:
        itype = str(snapshot.get("anchor_detail", {}).get("intervention_type", "")).strip()
    if itype in FEWSHOT_BY_TYPE:
        return FEWSHOT_BY_TYPE[itype]
    return FEWSHOT_BY_TYPE["continuous_infusion"]


def _golden_action_hint(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    anchor = snapshot.get("anchor_detail", {})
    med_key = str(anchor.get("medication_key", "")).strip()
    actual = str(snapshot.get("actual_intervention", "")).strip()
    kws = GOLDEN_ACTION_KEYWORDS.get(med_key, [])
    return {
        "medication_key": med_key,
        "actual_intervention": actual,
        "keywords": kws,
    }


def _question_focus_instruction(snapshot: Dict[str, Any]) -> str:
    itype = str(snapshot.get("interpreted_intervention_type", "")).strip()
    if not itype:
        itype = str(snapshot.get("anchor_detail", {}).get("intervention_type", "")).strip()
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).strip()
    delta = _safe_float(anchor.get("delta"))
    smoothed_delta_ml = _safe_float(anchor.get("smoothed_delta_volume_ml"))

    if itype == "rate_adjustment":
        if med_key.endswith("_RATE") and delta is not None and abs(delta) < 1.0:
            return (
                "Question focus: this is a mild maintenance adjustment sample. "
                "Q should ask for status evaluation plus maintenance strategy."
            )
        if med_key.endswith("_VOL") and smoothed_delta_ml is not None and smoothed_delta_ml < 1.0:
            return (
                "Question focus: this is a mild maintenance adjustment sample. "
                "Q should ask for status evaluation plus maintenance strategy."
            )
    if itype in {"continuous_infusion", "unlabeled_context_snapshot"}:
        return (
            "Question focus: this is a maintenance/state-assessment sample. "
            "Q should ask for status evaluation plus maintenance strategy, not emergency rescue."
        )
    return (
        "Question focus: this is an active-decision sample. "
        "Q should prioritize MAP/HR/SpO2 risk first, then discuss anesthesia-depth adjustment."
    )


def _format_miller_evidence(retrieval: Optional[Dict[str, Any]]) -> str:
    if not retrieval or not retrieval.get("results"):
        return ""
    query_raw = _coerce_text(retrieval.get("query_raw"))
    query_rewritten = _coerce_text(retrieval.get("query_rewritten") or retrieval.get("query"))
    intents = retrieval.get("intent_tags", [])
    intent_text = "; ".join(str(x) for x in intents) if isinstance(intents, list) else ""
    blocks = [
        f"Miller retrieval raw query:\n{query_raw}\n",
        f"Miller retrieval rewritten query:\n{query_rewritten}\n",
        f"Miller retrieval intent tags:\n{intent_text}\n",
        "Evidence locator format to cite in output: [M10#rank|章节:...; 小节:...; 段落:...; 页:...; chunk:...]",
        "Retrieved evidence excerpts from Miller's Anesthesia, 10th edition (hybrid top-k):",
    ]
    for item in retrieval["results"]:
        source = _coerce_text(item.get("source")) or "unknown_source"
        score = float(item.get("fusion_score", item.get("score", 0.0)))
        chunk_id = item.get("chunk_id")
        methods = ",".join(item.get("retrieval_methods", [])) if isinstance(item.get("retrieval_methods"), list) else ""
        text = _coerce_text(item.get("text"))
        locator = _format_miller_locator(item, rank=item.get("rank", "?"))
        blocks.append(
            f"[Evidence #{item.get('rank', '?')}] {locator} source={source} chunk={chunk_id} methods={methods} score={score:.4f}\n{text}"
        )
    return "\n".join(blocks) + "\n\n"


def build_user_prompt(snapshot: Dict[str, Any], retrieval: Optional[Dict[str, Any]] = None) -> str:
    snap_text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    fewshot = _fewshot_text_for_snapshot(snapshot)
    golden = _golden_action_hint(snapshot)
    med_key = golden["medication_key"]
    actual = golden["actual_intervention"]
    kws = golden["keywords"]
    kw_text = ", ".join(kws) if kws else "N/A"
    q_focus = _question_focus_instruction(snapshot)
    evidence_block = _format_miller_evidence(retrieval)
    has_evidence = bool(retrieval and isinstance(retrieval.get("results"), list) and retrieval.get("results"))
    evidence_rule = (
        "- 【决策干预（Miller）】必须至少包含1个证据定位标签，格式如 [M10#1|章节:...; 段落:...].\n"
        if has_evidence
        else "- 若无检索证据，仍保持三段格式，并在【决策干预（Miller）】中写明“证据定位不足”。\n"
    )
    return (
        "Below is a real OR monitoring snapshot in structured JSON:\n"
        f"{snap_text}\n\n"
        f"{fewshot}\n"
        f"{evidence_block}"
        "Supervision policy for this dataset:\n"
        f"- logged_action (golden): {actual}\n"
        f"- medication_key: {med_key}\n"
        f"- expected drug keywords: {kw_text}\n"
        "You MUST align 【决策干预（VitalDB）】 with logged_action (same drug class/category). "
        "Do not output a contradictory drug.\n"
        f"{q_focus}\n"
        "Clinical priority policy:\n"
        "- MAP absolute threshold is the perfusion floor; relative MAP drop is layered risk stratification.\n"
        "- BIS must be interpreted with MAP/HR/SpO2 and surgical stimulation; do not use BIS as a standalone trigger.\n"
        "- 【决策干预（Miller）】 MUST be grounded primarily in the retrieved excerpts from Miller's Anesthesia, 10th edition, when such excerpts are provided.\n"
        "- Do not present generic anesthesia knowledge as if it were a Miller 10th edition recommendation unless it is supported by the retrieved excerpts.\n"
        "- If retrieved Miller evidence is incomplete or ambiguous, explicitly stay conservative and fall back to objective physiologic signals in the snapshot.\n"
        f"{evidence_rule}"
        "You MUST output EXACTLY ONE QA pair in Chinese with this strict format:\n\n"
        "Q: <描述病人背景、术中阶段、体征趋势的流畅段落，最后提问最合理的药理干预>\n"
        "A: 【临床推理】：<精炼总结核心病理生理机制>\n"
        "【决策干预（Miller）】：<基于第十版米勒麻醉学检索证据的建议，必须写出证据定位如[M10#1|章节:...; 段落:...]>\n"
        "【决策干预（VitalDB）】：<与logged_action一致的实际策略，不得与golden冲突>\n\n"
        "The final QA block MUST be exactly these 4 lines (one line per label), no extra lines before or after.\n"
        "Use labels exactly as: Q:, A:, 【临床推理】, 【决策干预（Miller）】, 【决策干预（VitalDB）】.\n"
        "Do not change brackets, punctuation, or label names.\n"
        "If BIS data is missing, evaluate anesthesia depth indirectly using autonomic signs "
        "(HR, MBP trends) and surgical stimulation context.\n"
        "Use 'clinical_assessment.risk_flags', 'contextual_interpretation', "
        "'baseline_comparison', and 'drug_reference' to improve realism.\n"
        "Do not output any text outside the final QA pair.\n"
        "Forbidden: instruction echo, Analyze/Strategy/Constraint Check, bullet list, self-correction text."
    )
    
def _clean_raw_output(text: str) -> str:
    out = text.strip()
    out = re.sub(r"^```(?:json|markdown|text)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```$", "", out)
    out = out.replace("\r\n", "\n").strip()
    return out

def _extract_qa_block(text: str) -> str:
    out = _clean_raw_output(text)
    # Remove any CoT enclosed in think tags before extracting QA.
    out = re.sub(r"(?is)<think>.*?</think>", "", out).strip()
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()
    # Keep only the last Q: segment to avoid draft Q/A that may appear earlier.
    q_matches = list(re.finditer(r"(?im)^Q\s*[:：]", out))
    if q_matches:
        out = out[q_matches[-1].start() :].strip()
    # Strictly extract final Q/A block.
    match = re.search(
        r"(Q\s*[:：].*?A\s*[:：].*?【决策干预.*?(?=\n\n|\n\*|\n<|<|```|$))",
        out,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return out.strip()

def _is_strict_qa(text: str) -> bool:
    out = _extract_qa_block(text)
    low = out.lower()
    if LEAK_TOKEN_RE.search(low):
        return False
    banned = ["let's think", "<think>", "</think>", "**content requirements**", "**strategy**"]
    if any(b in low for b in banned):
        return False
    if re.search(r"(?im)^\s*(\*|-|\d+\.)\s+", out):
        return False
    if not re.search(r"(?im)^\s*Q\s*[:：]", out):
        return False
    if not re.search(r"(?im)^\s*A\s*[:：]", out):
        return False
    has_reason = ("【临床推理】" in out) or ("[Clinical Reasoning]" in out)
    has_decision_dual = ("【决策干预（Miller）】" in out) and ("【决策干预（VitalDB）】" in out)
    if not has_reason:
        return False
    if not has_decision_dual:
        return False
    lines = [line.strip() for line in out.splitlines() if line.strip()]
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
    if not has_locator:
        return False
    return True


def _decision_section_vitaldb(text: str) -> str:
    out = _extract_qa_block(text)
    m = re.search(r"【决策干预（VitalDB）】[:：]?\s*(.*?)(?=\n【|$)", out, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"【决策干预】[:：]?\s*(.*)$", out, re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return out


def _decision_section(text: str) -> str:
    out = _extract_qa_block(text)
    m = re.search(r"【决策干预（Miller）】[:：]?\s*(.*?)(?=\n【|$)", out, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return _decision_section_vitaldb(out)


def _is_action_aligned(text: str, snapshot: Dict[str, Any]) -> bool:
    hint = _golden_action_hint(snapshot)
    kws = [str(k).strip() for k in hint.get("keywords", []) if str(k).strip()]
    if not kws:
        return True
    dec = _decision_section_vitaldb(text).lower()
    return any(k.lower() in dec for k in kws)


def _repair_qa_output(client: Any, model: str, raw_text: str, snapshot: Dict[str, Any]) -> Optional[str]:
    hint = _golden_action_hint(snapshot)
    med_key = hint.get("medication_key", "")
    actual = hint.get("actual_intervention", "")
    kws = hint.get("keywords", [])
    kw_text = ", ".join(kws) if kws else "N/A"
    repair_sys = (
        "You are a strict medical QA formatter. "
        "Return only final QA in Chinese. "
        "No thinking process, no bullets, no markdown, no instruction echo, no extra preface/suffix."
    )
    repair_user = (
        "Rewrite to strict format. You MUST output EXACTLY this 4-line template:\n"
        "Q: <一句问题>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：<1-3句，且必须包含证据定位标签如[M10#1|章节:...; 段落:...]>\n"
        "【决策干预（VitalDB）】：<1-2句>\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        "Do not output anything outside the 4-line QA block.\n"
        "Miller line must include at least one M10 locator token.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        "【决策干预（VitalDB）】必须与golden logged_action同药物类别，不得矛盾。\n"
        "Source text:\n"
        f"{raw_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=LLM_MAX_TOKENS_DEFAULT,
            messages=[
                {"role": "system", "content": repair_sys},
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
    return content.strip() if isinstance(content, str) else str(content).strip()
def create_openai_client(cfg: PipelineConfig) -> Any:
    if OpenAI is None:
        raise ImportError("openai package is not installed")
    if cfg.llm_api_key.strip():
        api_key = cfg.llm_api_key.strip()
    else:
        api_key = os.getenv(cfg.api_key_env, "").strip()

    # For local OpenAI-compatible servers (vLLM / Ollama), any non-empty key is often acceptable.
    if cfg.llm_base_url.strip():
        if not api_key:
            api_key = "local"
        return OpenAI(api_key=api_key, base_url=cfg.llm_base_url.strip().rstrip("/"))

    # OpenAI cloud mode (default): key is required.
    if not api_key:
        raise ValueError(
            f"LLM key is empty. Set --llm-api-key or env var {cfg.api_key_env}, "
            "or provide --llm-base-url for local model serving."
        )
    return OpenAI(api_key=api_key)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    out = _clean_raw_output(text)
    if out.startswith("{") and out.endswith("}"):
        try:
            obj = json.loads(out)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", out)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _rule_validate_actual_intervention(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    actual = str(snapshot.get("actual_intervention", "")).strip()
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).strip()
    if not actual:
        return {"verdict": "invalid", "confidence": 0.95, "reason": "empty_actual_intervention"}

    kws = [str(k).strip().lower() for k in GOLDEN_ACTION_KEYWORDS.get(med_key, []) if str(k).strip()]
    if kws and not any(k in actual.lower() for k in kws):
        return {"verdict": "invalid", "confidence": 0.9, "reason": "keyword_mismatch_with_medication_key"}

    before = anchor.get("before")
    after = anchor.get("after")
    delta = anchor.get("delta")
    try:
        if before is not None and after is not None and delta is not None:
            b = float(before)
            a = float(after)
            d = float(delta)
            diff = abs((a - b) - d)
            tol = max(0.05, 0.02 * max(abs(d), 1.0))
            if diff > tol:
                return {"verdict": "invalid", "confidence": 0.88, "reason": "numeric_delta_inconsistent_with_anchor"}
    except Exception:
        pass

    return {"verdict": "valid", "confidence": 0.8, "reason": "rule_consistent_with_anchor"}


def validate_actual_intervention(client: Any, model: str, snapshot: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
    rule_meta = _rule_validate_actual_intervention(snapshot)
    payload = {
        "actual_intervention": snapshot.get("actual_intervention"),
        "interpreted_intervention_type": snapshot.get("interpreted_intervention_type"),
        "anchor_detail": snapshot.get("anchor_detail"),
        "clinical_assessment": snapshot.get("clinical_assessment"),
    }
    sys_prompt = (
        "你是麻醉数据质控审核员。"
        "请判断 logged_action(即actual_intervention) 是否与给定锚点数值一致且可作为训练标签。"
        "只输出JSON，不要输出任何解释性前后缀。"
    )
    user_prompt = (
        "请输出如下JSON格式：\n"
        "{\"verdict\":\"valid|invalid|uncertain\",\"confidence\":0.0,\"reason\":\"...\"}\n\n"
        "判定标准：\n"
        "- valid: 与anchor_detail/类型一致，语义可解释。\n"
        "- invalid: 与锚点字段明显矛盾、单位/方向明显错误、或不可作为可靠标签。\n"
        "- uncertain: 仅在关键字段缺失、无法判断时使用。\n"
        "若未发现明确矛盾，默认给出 valid。\n\n"
        f"输入:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        return {
            "verdict": rule_meta["verdict"],
            "confidence": max(float(rule_meta.get("confidence", 0.0)), 0.6),
            "reason": f"fallback_rule_after_validation_request_failed: {e}; {rule_meta.get('reason','')}",
        }
    if not resp.choices:
        return {
            "verdict": rule_meta["verdict"],
            "confidence": max(float(rule_meta.get("confidence", 0.0)), 0.6),
            "reason": f"fallback_rule_after_validation_empty_choices; {rule_meta.get('reason','')}",
        }
    content = resp.choices[0].message.content
    raw = content.strip() if isinstance(content, str) else str(content or "").strip()

    obj = _extract_first_json_object(raw)
    if obj is None:
        low = raw.lower()
        if "invalid" in low or "无效" in raw or "不准确" in raw:
            return {"verdict": "invalid", "confidence": 0.35, "reason": "validation_non_json_output_detected_invalid"}
        if "valid" in low or "有效" in raw or "准确" in raw:
            return {"verdict": "valid", "confidence": 0.35, "reason": "validation_non_json_output_detected_valid"}
        return {
            "verdict": rule_meta["verdict"],
            "confidence": max(float(rule_meta.get("confidence", 0.0)), 0.6),
            "reason": f"fallback_rule_after_validation_non_json_output; {rule_meta.get('reason','')}",
        }

    verdict = str(obj.get("verdict", "uncertain")).strip().lower()
    if verdict not in {"valid", "invalid", "uncertain"}:
        verdict = "uncertain"
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    reason = str(obj.get("reason", "")).strip()
    # If model is uncertain or gives placeholder reason, fall back to deterministic consistency rule.
    if verdict == "uncertain" or reason in {"", "...", "…", "unknown"}:
        return {
            "verdict": rule_meta["verdict"],
            "confidence": max(confidence, float(rule_meta.get("confidence", 0.0))),
            "reason": f"fallback_rule_after_model_uncertain; {rule_meta.get('reason','')}",
        }
    # Keep model invalid decisions for safety; otherwise prefer model output.
    return {"verdict": verdict, "confidence": confidence, "reason": reason}


def _skip_after_actual_validation(meta: Dict[str, Any], cfg: PipelineConfig) -> bool:
    verdict = str(meta.get("verdict", "uncertain")).strip().lower()
    if verdict == "invalid" and cfg.drop_if_actual_invalid:
        return True
    if verdict == "uncertain" and cfg.drop_if_actual_uncertain:
        return True
    return False


def generate_single_qa(
    client: Any,
    model: str,
    snapshot: Dict[str, Any],
    retrieval: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    user_prompt = build_user_prompt(snapshot, retrieval=retrieval)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=LLM_MAX_TOKENS_DEFAULT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        print(f"  - LLM request failed: {e}")
        return None
    if not resp.choices:
        return None
    content = resp.choices[0].message.content
    if content is None:
        return None
    raw = content.strip() if isinstance(content, str) else str(content).strip()
    cleaned = _extract_qa_block(raw)
    if _is_strict_qa(cleaned) and _is_action_aligned(cleaned, snapshot):
        return cleaned

    repaired = _repair_qa_output(client, model, raw, snapshot)
    if repaired:
        repaired_cleaned = _extract_qa_block(repaired)
        if _is_strict_qa(repaired_cleaned) and _is_action_aligned(repaired_cleaned, snapshot):
            return repaired_cleaned

    # Fail fast: do not keep polluted draft text in final dataset.
    return None


def stage3_generate_qa(records: List[Dict[str, Any]], cfg: PipelineConfig) -> None:
    if not cfg.enable_llm:
        print(">>> Stage 3: skipped (LLM disabled)")
        return
    total = len(records)
    progress_every = max(1, int(cfg.llm_progress_every))
    workers = max(1, int(cfg.llm_max_workers))
    print(f">>> Stage 3: generate QA by LLM (workers={workers})")
    val_model = cfg.actual_validation_model.strip() if cfg.actual_validation_model.strip() else cfg.llm_model
    val_checked = 0
    val_kept = 0
    val_skipped = 0
    retriever = _make_miller_retriever(passages=[], embeddings=np.zeros((0, 0), dtype=np.float32))
    shared_embed_client: Any = None

    if cfg.enable_miller_rag:
        print(">>> Stage 3a: build Miller embedding retriever")
        cache_loaded = False
        try:
            retriever = build_miller_retriever(None, cfg)
            cache_loaded = True
        except RuntimeError:
            cache_loaded = False

        if not cache_loaded:
            shared_embed_client = create_embedding_client(cfg)
            retriever = build_miller_retriever(shared_embed_client, cfg)
        else:
            try:
                shared_embed_client = create_embedding_client(cfg)
            except Exception as e:  # noqa: BLE001
                shared_embed_client = None
                print(f"  - embedding client unavailable, fallback to BM25-only retrieval: {e}")
        print(f"  - Miller retriever ready: {len(retriever.passages)} chunks")

    os.makedirs(os.path.dirname(cfg.llm_jsonl), exist_ok=True)
    retrieval_log_path = cfg.miller_retrieval_log_jsonl.strip()
    retrieval_csv_path = cfg.miller_retrieval_log_csv.strip()
    if cfg.enable_miller_rag and retrieval_log_path:
        os.makedirs(os.path.dirname(retrieval_log_path), exist_ok=True)
        print(f"  - Miller retrieval log: {retrieval_log_path}")
    if cfg.enable_miller_rag and retrieval_csv_path:
        os.makedirs(os.path.dirname(retrieval_csv_path), exist_ok=True)
        print(f"  - Miller retrieval csv: {retrieval_csv_path}")

    if workers <= 1:
        client = create_openai_client(cfg)
        embed_client = shared_embed_client if cfg.enable_miller_rag else None
        retrieval_ctx = open(retrieval_log_path, "w", encoding="utf-8") if (cfg.enable_miller_rag and retrieval_log_path) else nullcontext(None)
        retrieval_csv_ctx = (
            open(retrieval_csv_path, "w", encoding="utf-8", newline="")
            if (cfg.enable_miller_rag and retrieval_csv_path)
            else nullcontext(None)
        )
        with open(cfg.llm_jsonl, "w", encoding="utf-8") as f, retrieval_ctx as retrieval_f, retrieval_csv_ctx as retrieval_csv_f:
            csv_writer = None
            csv_header_written = False
            for i, rec in enumerate(records, start=1):
                if i % progress_every == 0 or i == total:
                    print(f"  - LLM progress: {i}/{total}")
                if cfg.validate_actual_before_qa:
                    meta = validate_actual_intervention(
                        client=client,
                        model=val_model,
                        snapshot=rec["snapshot"],
                        max_tokens=cfg.actual_validation_max_tokens,
                    )
                    rec["actual_validation"] = meta
                    val_checked += 1
                    if _skip_after_actual_validation(meta, cfg):
                        rec["llm_output"] = None
                        val_skipped += 1
                        f.write(_safe_json_dumps(rec) + "\n")
                        continue
                    val_kept += 1

                retrieval = None
                if cfg.enable_miller_rag:
                    retrieval = retrieve_miller_context(rec["snapshot"], retriever, embed_client, cfg)
                    rec["miller_retrieval"] = retrieval
                    if retrieval_f is not None:
                        retrieval_log_rec = _build_miller_retrieval_log_record(
                            rec=rec,
                            retrieval=retrieval,
                            max_chars=cfg.miller_retrieval_log_max_chars,
                        )
                        retrieval_f.write(_safe_json_dumps(retrieval_log_rec) + "\n")
                        if retrieval_csv_f is not None:
                            csv_rows = _iter_miller_retrieval_csv_rows(retrieval_log_rec)
                            if csv_rows:
                                if csv_writer is None:
                                    csv_writer = csv.DictWriter(retrieval_csv_f, fieldnames=list(csv_rows[0].keys()))
                                if not csv_header_written:
                                    csv_writer.writeheader()
                                    csv_header_written = True
                                for row in csv_rows:
                                    csv_writer.writerow(row)

                rec["llm_output"] = generate_single_qa(
                    client,
                    cfg.llm_model,
                    rec["snapshot"],
                    retrieval=retrieval,
                )
                f.write(_safe_json_dumps(rec) + "\n")
        if cfg.validate_actual_before_qa:
            print(f"  - actual validation: checked={val_checked}, kept={val_kept}, skipped={val_skipped}")
        return

    thread_local = threading.local()

    def _get_thread_client() -> Any:
        if not hasattr(thread_local, "client"):
            thread_local.client = create_openai_client(cfg)
        return thread_local.client

    def _get_thread_embed_client() -> Any:
        if not hasattr(thread_local, "embed_client"):
            if shared_embed_client is None:
                thread_local.embed_client = None
            else:
                try:
                    thread_local.embed_client = create_embedding_client(cfg)
                except Exception as e:  # noqa: BLE001
                    print(f"  - embedding client unavailable in worker, fallback BM25-only: {e}")
                    thread_local.embed_client = None
        return thread_local.embed_client

    def _worker(
        idx: int,
        snap: Dict[str, Any],
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            client = _get_thread_client()
            meta: Optional[Dict[str, Any]] = None
            if cfg.validate_actual_before_qa:
                meta = validate_actual_intervention(
                    client=client,
                    model=val_model,
                    snapshot=snap,
                    max_tokens=cfg.actual_validation_max_tokens,
                )
                if _skip_after_actual_validation(meta, cfg):
                    return idx, None, meta, None
            retrieval = None
            if cfg.enable_miller_rag:
                embed_client = _get_thread_embed_client()
                retrieval = retrieve_miller_context(snap, retriever, embed_client, cfg)
            qa = generate_single_qa(client, cfg.llm_model, snap, retrieval=retrieval)
            return idx, qa, meta, retrieval
        except Exception as e:  # noqa: BLE001
            print(f"  - LLM worker failed at idx={idx}: {e}")
            return (
                idx,
                None,
                {"verdict": "uncertain", "confidence": 0.0, "reason": f"worker_exception: {e}"},
                None,
            )

    retrieval_ctx = open(retrieval_log_path, "w", encoding="utf-8") if (cfg.enable_miller_rag and retrieval_log_path) else nullcontext(None)
    retrieval_csv_ctx = (
        open(retrieval_csv_path, "w", encoding="utf-8", newline="")
        if (cfg.enable_miller_rag and retrieval_csv_path)
        else nullcontext(None)
    )
    with ThreadPoolExecutor(max_workers=workers) as ex, open(cfg.llm_jsonl, "w", encoding="utf-8") as f, retrieval_ctx as retrieval_f, retrieval_csv_ctx as retrieval_csv_f:
        csv_writer = None
        csv_header_written = False
        futures = [ex.submit(_worker, idx, rec["snapshot"]) for idx, rec in enumerate(records)]
        done = 0
        for fut in as_completed(futures):
            idx, qa, meta, retrieval = fut.result()
            records[idx]["llm_output"] = qa
            if retrieval is not None:
                records[idx]["miller_retrieval"] = retrieval
                if retrieval_f is not None:
                    retrieval_log_rec = _build_miller_retrieval_log_record(
                        rec=records[idx],
                        retrieval=retrieval,
                        max_chars=cfg.miller_retrieval_log_max_chars,
                    )
                    retrieval_f.write(_safe_json_dumps(retrieval_log_rec) + "\n")
                    if retrieval_csv_f is not None:
                        csv_rows = _iter_miller_retrieval_csv_rows(retrieval_log_rec)
                        if csv_rows:
                            if csv_writer is None:
                                csv_writer = csv.DictWriter(retrieval_csv_f, fieldnames=list(csv_rows[0].keys()))
                            if not csv_header_written:
                                csv_writer.writeheader()
                                csv_header_written = True
                            for row in csv_rows:
                                csv_writer.writerow(row)
            if cfg.validate_actual_before_qa:
                records[idx]["actual_validation"] = meta
                val_checked += 1
                if _skip_after_actual_validation(meta or {}, cfg):
                    val_skipped += 1
                else:
                    val_kept += 1
            f.write(_safe_json_dumps(records[idx]) + "\n")
            done += 1
            if done % progress_every == 0 or done == total:
                print(f"  - LLM concurrent progress: {done}/{total}")
    if cfg.validate_actual_before_qa:
        print(f"  - actual validation: checked={val_checked}, kept={val_kept}, skipped={val_skipped}")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _has_nonempty_risk_flags(snapshot: Dict[str, Any]) -> bool:
    assess = snapshot.get("clinical_assessment", {})
    if not isinstance(assess, dict):
        return False
    flags = assess.get("risk_flags", [])
    return isinstance(flags, list) and any(str(x).strip() for x in flags)


def _has_objective_instability(snapshot: Dict[str, Any]) -> bool:
    assess = snapshot.get("clinical_assessment", {})
    if not isinstance(assess, dict):
        return False
    recent = assess.get("recent_state_mean", {})
    baseline = assess.get("baseline_comparison", {})
    if not isinstance(recent, dict):
        recent = {}
    if not isinstance(baseline, dict):
        baseline = {}

    map_now = _safe_float(recent.get("MAP_mmhg"))
    hr_now = _safe_float(recent.get("HR_bpm"))
    spo2_now = _safe_float(recent.get("SpO2_pct"))
    bis_now = _safe_float(recent.get("BIS"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))

    if map_now is not None and map_now < 65.0:
        return True
    if map_drop_pct is not None and map_drop_pct >= 15.0:
        return True
    if hr_now is not None and (hr_now < 50.0 or hr_now > 100.0):
        return True
    if spo2_now is not None and spo2_now < 94.0:
        return True
    # BIS is intentionally de-emphasized: only counts when coupled with hemodynamic/oxygenation abnormality.
    if (
        bis_now is not None
        and (bis_now < 40.0 or bis_now > 60.0)
        and (
            (map_now is not None and map_now < 65.0)
            or (hr_now is not None and (hr_now < 50.0 or hr_now > 100.0))
            or (spo2_now is not None and spo2_now < 94.0)
        )
    ):
        return True
    return False


def _infer_action_class_from_snapshot(snapshot: Dict[str, Any]) -> str:
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).strip()
    if med_key in MED_CLASS_BY_KEY:
        return MED_CLASS_BY_KEY[med_key]

    text = str(snapshot.get("actual_intervention", "")).lower()
    if any(k in text for k in ("去甲", "去氧", "肾上腺素", "norepinephrine", "phenylephrine", "epinephrine")):
        return "vasopressor"
    if any(k in text for k in ("麻黄碱", "ephedrine")):
        return "vasopressor"
    if any(k in text for k in ("丙泊酚", "propofol")):
        return "hypnotic_iv"
    if any(k in text for k in ("瑞芬太尼", "remifentanil")):
        return "opioid_analgesic"
    if any(k in text for k in ("七氟烷", "地氟烷", "异氟烷", "sevoflurane", "desflurane", "isoflurane")):
        return "hypnotic_volatile"
    if any(k in text for k in ("硝酸甘油", "nitroglycerin", "glyceryl trinitrate", "tng")):
        return "vasodilator"
    if any(k in text for k in ("米力农", "milrinone")):
        return "inodilator"
    if any(k in text for k in ("阿托品", "atropine")):
        return "chronotropic"
    if any(k in text for k in ("尼卡地平", "硝普钠", "艾司洛尔", "乌拉地尔", "nicardipine", "nitroprusside", "esmolol")):
        return "anti_sympathetic"
    return "unknown"


def _infer_action_drug_from_snapshot(snapshot: Dict[str, Any]) -> str:
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).strip()
    if med_key in ACTION_DRUG_BY_MED_KEY:
        return ACTION_DRUG_BY_MED_KEY[med_key]

    text = str(snapshot.get("actual_intervention", "")).lower()
    if any(k in text for k in ("去氧", "苯肾上腺素", "phenylephrine")):
        return "phenylephrine"
    if any(k in text for k in ("麻黄碱", "ephedrine")):
        return "ephedrine"
    if any(k in text for k in ("去甲", "norepinephrine")):
        return "norepinephrine"
    if any(k in text for k in ("肾上腺素", "epinephrine")):
        return "epinephrine"
    if any(k in text for k in ("硝酸甘油", "nitroglycerin", "glyceryl trinitrate", "tng")):
        return "nitroglycerin"
    if any(k in text for k in ("米力农", "milrinone")):
        return "milrinone"
    if any(k in text for k in ("阿托品", "atropine")):
        return "atropine"
    if any(k in text for k in ("丙泊酚", "propofol")):
        return "propofol"
    if any(k in text for k in ("瑞芬太尼", "remifentanil")):
        return "remifentanil"
    return "unknown"


def _is_action_escalation(snapshot: Dict[str, Any], delta: Optional[float]) -> bool:
    if delta is not None:
        if delta > 0:
            return True
        if delta < 0:
            return False
    text = str(snapshot.get("actual_intervention", "")).lower()
    decrease_keywords = ("减少", "减量", "下调", "降低", "停用", "decrease", "down-titrate", "stop", "wean")
    increase_keywords = ("增加", "加量", "追加", "上调", "推注", "滴注", "泵注", "increase", "up-titrate", "bolus")
    if any(k in text for k in decrease_keywords):
        return False
    return any(k in text for k in increase_keywords)


def evaluate_vitaldb_vs_miller(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    assess = snapshot.get("clinical_assessment", {})
    if not isinstance(assess, dict):
        assess = {}
    recent = assess.get("recent_state_mean", {})
    baseline = assess.get("baseline_comparison", {})
    persistence = assess.get("persistence_seconds", {})

    if not isinstance(recent, dict):
        recent = {}
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(persistence, dict):
        persistence = {}

    map_now = _safe_float(recent.get("MAP_mmhg"))
    hr_now = _safe_float(recent.get("HR_bpm"))
    spo2_now = _safe_float(recent.get("SpO2_pct"))
    bis_now = _safe_float(recent.get("BIS"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))
    map_lt_65_persist = _safe_float(persistence.get("map_lt_65")) or 0.0
    map_lt_55_persist = _safe_float(persistence.get("map_lt_55")) or 0.0
    spo2_lt_90_persist = _safe_float(persistence.get("spo2_lt_90")) or 0.0

    critical_sec = float(MILLER_POLICY_THRESHOLDS["critical_window_sec"])
    hemo_sec = float(MILLER_POLICY_THRESHOLDS["hemodynamic_window_sec"])

    strategies: List[str] = []
    reasons: List[str] = []

    def _add_strategy(tag: str, reason: str) -> None:
        if tag not in strategies:
            strategies.append(tag)
            reasons.append(reason)

    if spo2_now is not None and (
        spo2_now < ANES_THRESHOLDS["spo2_severe_low_pct"] or spo2_lt_90_persist >= critical_sec
    ):
        _add_strategy("oxygenation_first", "低氧优先纠正氧合/通气，不应先机械加深麻醉。")

    severe_hypotension = False
    if map_now is not None:
        if map_now < ANES_THRESHOLDS["map_severe_hypotension_mmhg"] or map_lt_55_persist >= critical_sec:
            severe_hypotension = True
        if map_now < ANES_THRESHOLDS["map_hypotension_mmhg"] and map_lt_65_persist >= hemo_sec:
            severe_hypotension = True
    if map_drop_pct is not None and map_drop_pct >= ANES_THRESHOLDS["map_relative_drop_pct"]:
        severe_hypotension = True
    map_low = (map_now is not None) and (map_now < ANES_THRESHOLDS["map_hypotension_mmhg"])

    if severe_hypotension:
        _add_strategy("perfusion_first", "MAP绝对阈值/相对下降触发，应先稳灌注。")
        if hr_now is not None and hr_now < ANES_THRESHOLDS["hr_bradycardia_bpm"]:
            _add_strategy("treat_brady_or_reduce_depth", "低血压伴心动过缓，应排查过深麻醉/传导抑制。")
        if hr_now is not None and hr_now > ANES_THRESHOLDS["hr_tachycardia_bpm"]:
            _add_strategy("consider_volume_or_stimulus", "低血压伴心动过速，需评估容量不足/失血/刺激反应。")
        if hr_now is not None and hr_now >= 60.0:
            _add_strategy("prefer_phenylephrine_when_hr_not_low", "低血压且HR不低时可优先考虑去氧肾上腺素。")
        if hr_now is not None and hr_now < 60.0:
            _add_strategy("prefer_ephedrine_when_hypotension_with_low_hr", "低血压合并低心率时，麻黄碱常优于纯α激动剂。")
        if hr_now is not None and hr_now < 45.0:
            _add_strategy("consider_atropine_for_hemodynamic_bradycardia", "HR<45且灌注受损时可考虑阿托品纠正迷走性慢心率。")
        if (map_now is not None and map_now < ANES_THRESHOLDS["map_severe_hypotension_mmhg"]) or map_lt_55_persist >= critical_sec:
            _add_strategy("consider_norepinephrine_for_refractory_hypotension", "重度/持续低血压可考虑去甲肾上腺素，但需同步评估容量。")
        if hr_now is not None and hr_now > 110.0:
            _add_strategy("rule_out_hypovolemia_before_high_dose_norepinephrine", "疑似低容量时应先扩容，避免直接强化去甲导致微循环风险。")

    if map_low:
        _add_strategy("avoid_vasodilator_when_map_low", "MAP<65时应避免先行扩血管药（硝酸甘油/米力农）升级。")

    if (
        bis_now is not None
        and bis_now > ANES_THRESHOLDS["bis_light"]
        and (map_now is None or map_now >= ANES_THRESHOLDS["map_hypotension_mmhg"])
        and (spo2_now is None or spo2_now >= ANES_THRESHOLDS["spo2_low_pct"])
    ):
        _add_strategy("consider_depth_or_analgesia_increase", "BIS升高仅在灌注和氧合可接受时考虑加深镇静/镇痛。")

    if bis_now is not None and bis_now < ANES_THRESHOLDS["bis_deep"] and severe_hypotension:
        _add_strategy("reduce_depth", "BIS偏低合并低灌注时，优先减浅麻醉并支持循环。")

    if not strategies:
        _add_strategy("context_monitoring", "未触发强干预信号，以连续监测和小步可逆调整为主。")

    actual_class = _infer_action_class_from_snapshot(snapshot)
    actual_drug = _infer_action_drug_from_snapshot(snapshot)
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    delta = _safe_float(anchor.get("delta"))
    is_escalation = _is_action_escalation(snapshot, delta)
    clinical_rules = _load_clinical_conflict_rules()
    classes_worsen_perfusion = set(
        str(x).strip()
        for x in clinical_rules.get("classes_worsen_perfusion", [])
        if str(x).strip()
    )
    map_below_75 = (map_now is not None) and (map_now < 75.0)
    map_ge_65 = (map_now is not None) and (map_now >= 65.0)
    map_ge_75 = (map_now is not None) and (map_now >= 75.0)

    facts: Dict[str, bool] = {
        "strategy_oxygenation_first": "oxygenation_first" in strategies,
        "strategy_perfusion_first": "perfusion_first" in strategies,
        "strategy_reduce_depth": "reduce_depth" in strategies,
        "strategy_consider_depth_or_analgesia_increase": "consider_depth_or_analgesia_increase" in strategies,
        "strategy_context_monitoring": "context_monitoring" in strategies,
        "action_escalation": bool(is_escalation),
        "class_worsen_perfusion": actual_class in classes_worsen_perfusion,
        "class_hypnotic_or_vasodilator_inodilator": actual_class in {"hypnotic_iv", "hypnotic_volatile", "vasodilator", "inodilator"},
        "class_hypnotic": actual_class in {"hypnotic_iv", "hypnotic_volatile"},
        "class_vasopressor_or_inopressor": actual_class in {"vasopressor", "inopressor"},
        "class_opioid_or_hypnotic": actual_class in {"opioid_analgesic", "hypnotic_iv", "hypnotic_volatile"},
        "class_monitoring_compatible": actual_class in {"unknown", "neuromuscular", "arrhythmia"},
        "drug_phenylephrine": actual_drug == "phenylephrine",
        "drug_ephedrine": actual_drug == "ephedrine",
        "drug_norepinephrine": actual_drug == "norepinephrine",
        "drug_epinephrine": actual_drug == "epinephrine",
        "drug_nitroglycerin": actual_drug == "nitroglycerin",
        "drug_milrinone": actual_drug == "milrinone",
        "drug_atropine": actual_drug == "atropine",
        "drug_propofol": actual_drug == "propofol",
        "drug_remifentanil": actual_drug == "remifentanil",
        "drug_vasodilator_or_inodilator": actual_drug in {"nitroglycerin", "milrinone"},
        "map_low": bool(map_low),
        "map_below_75": bool(map_below_75),
        "map_lt_55": (map_now is not None) and (map_now < 55.0),
        "map_not_lt_55": (map_now is None) or (map_now >= 55.0),
        "map_ge_65": bool(map_ge_65),
        "map_ge_75": bool(map_ge_75),
        "map_drop_ge_relative": (map_drop_pct is not None) and (map_drop_pct >= ANES_THRESHOLDS["map_relative_drop_pct"]),
        "bis_high": (bis_now is not None) and (bis_now > ANES_THRESHOLDS["bis_light"]),
        "hr_lt_50": (hr_now is not None) and (hr_now < 50.0),
        "hr_lt_60": (hr_now is not None) and (hr_now < 60.0),
        "hr_lt_45": (hr_now is not None) and (hr_now < 45.0),
        "hr_gt_100": (hr_now is not None) and (hr_now > 100.0),
        "hr_gt_110": (hr_now is not None) and (hr_now > 110.0),
        "hr_le_100": (hr_now is not None) and (hr_now <= 100.0),
        "hr_not_low": (hr_now is None) or (hr_now >= 60.0),
        "delta_negative": (delta is not None) and (delta < 0),
        "severe_hypotension": bool(severe_hypotension),
    }

    conflicts: List[str] = []
    high_risk_conflict = False
    for rule in clinical_rules.get("conflict_rules", []):
        if not isinstance(rule, dict):
            continue
        if _rule_matches_facts(rule, facts):
            reason = str(rule.get("reason", "")).strip()
            if reason:
                conflicts.append(reason)
            if bool(rule.get("high_risk", False)):
                high_risk_conflict = True

    aligned = False
    aligned_reason = "action_class_matches_miller_priority"
    partial_reasons: List[str] = []
    for rule in clinical_rules.get("alignment_rules", []):
        if not isinstance(rule, dict):
            continue
        if not _rule_matches_facts(rule, facts):
            continue
        outcome = str(rule.get("outcome", "")).strip().lower()
        reason = str(rule.get("reason", "")).strip()
        if outcome == "aligned":
            aligned = True
            if reason:
                aligned_reason = reason
        elif outcome in {"partial", "partially_aligned"}:
            if reason:
                partial_reasons.append(reason)

    verdict = "uncertain"
    reason = "insufficient_discriminative_signal"
    if conflicts:
        verdict = "misaligned"
        reason = conflicts[0]
    elif partial_reasons:
        verdict = "partially_aligned"
        reason = partial_reasons[0]
    elif aligned:
        verdict = "aligned"
        reason = aligned_reason

    return {
        "verdict": verdict,
        "reason": reason,
        "high_risk_conflict": high_risk_conflict,
        "miller_recommended_strategies": strategies,
        "miller_rationale": reasons,
        "vitaldb_action_class": actual_class,
        "vitaldb_action_drug": actual_drug,
        "vitaldb_action_text": str(snapshot.get("actual_intervention", "")),
    }


def classify_training_bucket(rec: Dict[str, Any], cfg: PipelineConfig) -> Tuple[str, str]:
    snap = rec.get("snapshot", {}) if isinstance(rec.get("snapshot"), dict) else {}
    anchor = snap.get("anchor_detail", {}) if isinstance(snap.get("anchor_detail"), dict) else {}
    itype = str(snap.get("interpreted_intervention_type", "") or anchor.get("intervention_type", "")).strip()
    med_key = str(anchor.get("medication_key", "")).strip()

    before = _safe_float(anchor.get("before"))
    after = _safe_float(anchor.get("after"))
    delta = _safe_float(anchor.get("delta"))
    smoothed_delta_ml = _safe_float(anchor.get("smoothed_delta_volume_ml"))

    # C bucket: known noisy/unusable anchors
    # Highest-priority veto: zero-start rate jumps are treated as setup/init artifacts.
    if itype == "rate_adjustment" and med_key.endswith("_RATE"):
        if before is not None and abs(before) <= 1e-6:
            return "C", "zero_start_rate_adjustment"

    if med_key.endswith("_RATE"):
        if (
            before is not None
            and after is not None
            and delta is not None
            and abs(before) <= float(cfg.setup_rate_before_abs_max)
            and abs(delta) >= float(cfg.setup_rate_delta_threshold)
            and after >= float(cfg.setup_rate_after_threshold)
        ):
            return "C", "setup_like_rate_jump"
    if med_key.endswith("_VOL") and smoothed_delta_ml is not None and smoothed_delta_ml < 0.5:
        return "C", "tiny_vol_background_drift"

    val_meta = rec.get("actual_validation", {})
    if isinstance(val_meta, dict):
        verdict = str(val_meta.get("verdict", "")).strip().lower()
        if verdict == "invalid":
            return "C", "actual_validation_invalid"

    # A bucket candidate: high-value active decisions
    a_candidate = False
    a_reason = ""
    if itype == "bolus_like_event":
        a_candidate = True
        a_reason = "bolus_like_event"
    if itype == "rate_adjustment":
        if med_key.endswith("_RATE") and delta is not None and abs(delta) >= max(1.0, float(cfg.rate_delta_threshold) * 2.0):
            a_candidate = True
            a_reason = "large_rate_adjustment"
        if med_key.endswith("_VOL") and smoothed_delta_ml is not None and smoothed_delta_ml >= max(1.0, float(cfg.propofol_bolus_min_delta_ml)):
            a_candidate = True
            a_reason = "large_volume_adjustment"

    if a_candidate:
        if cfg.strict_a_requires_risk_flags and (not _has_nonempty_risk_flags(snap)):
            return "B", f"downgraded_from_{a_reason}_missing_risk_flags"
        if cfg.strict_a_requires_objective_evidence and (not _has_objective_instability(snap)):
            return "B", f"downgraded_from_{a_reason}_missing_objective_instability"
        return "A", a_reason

    # B bucket: maintenance/context monitoring samples
    return "B", "maintenance_or_context"


def _record_has_trainable_qa(rec: Dict[str, Any]) -> bool:
    qa = rec.get("llm_output")
    if not (isinstance(qa, str) and qa.strip()):
        return False
    cleaned = _extract_qa_block(qa)
    if not _is_strict_qa(cleaned):
        return False
    snap = rec.get("snapshot")
    if isinstance(snap, dict) and not _is_action_aligned(cleaned, snap):
        return False
    return True


def _write_jsonl_records(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(_safe_json_dumps(rec) + "\n")


def _build_ab_mix(
    bucket_a: List[Dict[str, Any]],
    bucket_b: List[Dict[str, Any]],
    ratio_a: float,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not bucket_a and not bucket_b:
        return []
    r = max(0.0, min(1.0, float(ratio_a)))
    rng = random.Random(seed)
    a_pool = bucket_a[:]
    b_pool = bucket_b[:]
    rng.shuffle(a_pool)
    rng.shuffle(b_pool)

    if max_samples <= 0:
        # Use as many as possible under ratio while respecting pool limits.
        if r <= 0.0:
            take_b = len(b_pool)
            take_a = 0
        elif r >= 1.0:
            take_a = len(a_pool)
            take_b = 0
        else:
            max_n_by_a = int(len(a_pool) / r) if r > 0 else 0
            max_n_by_b = int(len(b_pool) / (1.0 - r)) if r < 1 else 0
            n = min(max_n_by_a, max_n_by_b)
            take_a = int(round(n * r))
            take_b = n - take_a
        mixed = a_pool[:take_a] + b_pool[:take_b]
        rng.shuffle(mixed)
        return mixed

    n_total = max(0, int(max_samples))
    target_a = int(round(n_total * r))
    target_b = n_total - target_a
    take_a = min(target_a, len(a_pool))
    take_b = min(target_b, len(b_pool))
    leftover = n_total - (take_a + take_b)
    if leftover > 0:
        extra_a = min(leftover, len(a_pool) - take_a)
        take_a += extra_a
        leftover -= extra_a
    if leftover > 0:
        extra_b = min(leftover, len(b_pool) - take_b)
        take_b += extra_b

    mixed = a_pool[:take_a] + b_pool[:take_b]
    rng.shuffle(mixed)
    return mixed


def export_bucketed_training_sets(
    records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    base_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    base_path = base_jsonl_path if base_jsonl_path else cfg.dataset_jsonl
    root, ext = os.path.splitext(base_path)
    ext = ext if ext else ".jsonl"

    a_rows: List[Dict[str, Any]] = []
    b_rows: List[Dict[str, Any]] = []
    c_rows: List[Dict[str, Any]] = []
    for rec in records:
        bucket, reason = classify_training_bucket(rec, cfg)
        rec["training_bucket"] = bucket
        rec["training_bucket_reason"] = reason
        if bucket == "A":
            a_rows.append(rec)
        elif bucket == "B":
            b_rows.append(rec)
        else:
            c_rows.append(rec)

    # For training files, keep only records with valid final QA text.
    a_train = [r for r in a_rows if _record_has_trainable_qa(r)]
    b_train = [r for r in b_rows if _record_has_trainable_qa(r)]
    c_train = [r for r in c_rows if _record_has_trainable_qa(r)]

    a_path = f"{root}.bucket_A{ext}"
    b_path = f"{root}.bucket_B{ext}"
    c_path = f"{root}.bucket_C{ext}"
    _write_jsonl_records(a_path, a_train)
    _write_jsonl_records(b_path, b_train)
    _write_jsonl_records(c_path, c_train)

    mix_rows = _build_ab_mix(
        bucket_a=a_train,
        bucket_b=b_train,
        ratio_a=cfg.train_mix_a_ratio,
        max_samples=cfg.train_mix_max_samples,
        seed=cfg.train_mix_seed,
    )
    a_pct = int(round(cfg.train_mix_a_ratio * 100))
    b_pct = 100 - a_pct
    mix_path = f"{root}.train_mix_A{a_pct}_B{b_pct}{ext}"
    _write_jsonl_records(mix_path, mix_rows)

    summary = {
        "bucket_a_total": len(a_rows),
        "bucket_b_total": len(b_rows),
        "bucket_c_total": len(c_rows),
        "bucket_a_trainable": len(a_train),
        "bucket_b_trainable": len(b_train),
        "bucket_c_trainable": len(c_train),
        "mixed_trainable": len(mix_rows),
        "bucket_a_path": a_path,
        "bucket_b_path": b_path,
        "bucket_c_path": c_path,
        "mix_path": mix_path,
    }
    return summary


def build_vitaldb_accuracy_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    verdict_counter: Dict[str, int] = {}
    reason_counter: Dict[str, int] = {}
    med_counter: Dict[str, int] = {}
    misaligned_med_counter: Dict[str, int] = {}
    high_risk_conflicts = 0

    for rec in records:
        snap = rec.get("snapshot", {}) if isinstance(rec.get("snapshot"), dict) else {}
        if not snap:
            continue
        alignment = snap.get("miller_alignment")
        if not isinstance(alignment, dict):
            alignment = evaluate_vitaldb_vs_miller(snap)
            snap["miller_alignment"] = alignment

        anchor = snap.get("anchor_detail", {}) if isinstance(snap.get("anchor_detail"), dict) else {}
        med_key = str(anchor.get("medication_key", "UNKNOWN")).strip() or "UNKNOWN"
        verdict = str(alignment.get("verdict", "uncertain")).strip().lower() or "uncertain"
        reason = str(alignment.get("reason", "unknown")).strip() or "unknown"
        high_risk = bool(alignment.get("high_risk_conflict", False))

        total += 1
        verdict_counter[verdict] = verdict_counter.get(verdict, 0) + 1
        reason_counter[reason] = reason_counter.get(reason, 0) + 1
        med_counter[med_key] = med_counter.get(med_key, 0) + 1
        if verdict in {"misaligned", "potentially_inaccurate"}:
            misaligned_med_counter[med_key] = misaligned_med_counter.get(med_key, 0) + 1
        if high_risk:
            high_risk_conflicts += 1

    misaligned_n = verdict_counter.get("misaligned", 0) + verdict_counter.get("potentially_inaccurate", 0)
    aligned_n = verdict_counter.get("aligned", 0)
    partial_n = verdict_counter.get("partially_aligned", 0)
    uncertain_n = verdict_counter.get("uncertain", 0)
    misaligned_ratio = float(misaligned_n / total) if total > 0 else 0.0

    top_reasons = sorted(reason_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_misaligned_meds = sorted(misaligned_med_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_evaluated": total,
        "aligned": aligned_n,
        "partially_aligned": partial_n,
        "misaligned": misaligned_n,
        "potentially_inaccurate": misaligned_n,
        "uncertain": uncertain_n,
        "misaligned_ratio": misaligned_ratio,
        "potentially_inaccurate_ratio": misaligned_ratio,
        "high_risk_conflicts": high_risk_conflicts,
        "verdict_counts": verdict_counter,
        "top_reasons": top_reasons,
        "medication_key_counts": med_counter,
        "top_misaligned_medication_keys": top_misaligned_meds,
        "top_inaccurate_medication_keys": top_misaligned_meds,
    }


def stage4_save_dataset(records: List[Dict[str, Any]], cfg: PipelineConfig) -> None:
    print(">>> Stage 4: save merged dataset to JSONL")
    os.makedirs(os.path.dirname(cfg.dataset_jsonl), exist_ok=True)
    mode = "w" if cfg.overwrite_jsonl else "a"
    with open(cfg.dataset_jsonl, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(_safe_json_dumps(rec) + "\n")
    print(f"Stage 4 done: wrote {len(records)} records -> {cfg.dataset_jsonl}")

    report = build_vitaldb_accuracy_report(records)
    report_path = os.path.join(os.path.dirname(cfg.dataset_jsonl), "vitaldb_miller_alignment_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    print(">>> Stage 4.3: VitalDB vs Miller alignment report")
    print(
        "  - verdicts: "
        f"aligned={report['aligned']} "
        f"partially_aligned={report['partially_aligned']} "
        f"misaligned={report['misaligned']} "
        f"uncertain={report['uncertain']}"
    )
    print(
        "  - misaligned_ratio="
        f"{report['misaligned_ratio']:.2%} "
        f"(high_risk_conflicts={report['high_risk_conflicts']})"
    )
    print(f"  - report file: {report_path}")

    if cfg.export_bucketed_datasets:
        summary = export_bucketed_training_sets(records, cfg)
        print(">>> Stage 4.6: bucketed training export")
        print(
            "  - buckets total: "
            f"A={summary['bucket_a_total']} B={summary['bucket_b_total']} C={summary['bucket_c_total']}"
        )
        print(
            "  - trainable: "
            f"A={summary['bucket_a_trainable']} B={summary['bucket_b_trainable']} C={summary['bucket_c_trainable']} "
            f"mixed={summary['mixed_trainable']}"
        )
        print(f"  - A file: {summary['bucket_a_path']}")
        print(f"  - B file: {summary['bucket_b_path']}")
        print(f"  - C file: {summary['bucket_c_path']}")
        print(f"  - MIX file: {summary['mix_path']}")


def clean_jsonl_file(
    input_jsonl: str,
    field: str = "llm_output",
    drop_invalid: bool = False,
    output_jsonl: Optional[str] = None,
    enforce_action_alignment: bool = True,
) -> str:
    input_path = input_jsonl
    if output_jsonl:
        output_path = output_jsonl
    else:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}.cleaned{ext if ext else '.jsonl'}"

    total = 0
    changed = 0
    strict_ok = 0
    dropped = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            original = rec.get(field)
            if isinstance(original, str) and original.strip():
                cleaned = _extract_qa_block(original)
                if cleaned != original:
                    changed += 1
                valid = _is_strict_qa(cleaned)
                if valid and enforce_action_alignment and isinstance(rec.get("snapshot"), dict):
                    valid = _is_action_aligned(cleaned, rec.get("snapshot", {}))
                rec[field] = cleaned
                if valid:
                    strict_ok += 1
                elif drop_invalid:
                    dropped += 1
                    continue
            elif drop_invalid:
                dropped += 1
                continue

            fout.write(_safe_json_dumps(rec) + "\n")

    print(">>> Stage 4.5: auto-clean report")
    print(f"  - input:   {input_path}")
    print(f"  - output:  {output_path}")
    print(f"  - total:   {total}")
    print(f"  - changed: {changed}")
    print(f"  - strict:  {strict_ok}")
    print(f"  - dropped: {dropped}")
    return output_path


def stage5_sample_review(cfg: PipelineConfig) -> None:
    print(">>> Stage 5: random sample review")
    if not os.path.exists(cfg.dataset_jsonl):
        print("  - dataset JSONL not found, skip")
        return
    with open(cfg.dataset_jsonl, "r", encoding="utf-8") as f:
        lines = [x for x in f.readlines() if x.strip()]
    if not lines:
        print("  - dataset is empty, skip")
        return
    n = max(1, int(len(lines) * cfg.sample_rate))
    n = min(n, len(lines))
    random.seed(cfg.random_seed)
    picks = random.sample(lines, n)
    print(f"  - total {len(lines)} records, sample {n}")
    for i, line in enumerate(picks, start=1):
        rec = json.loads(line)
        snapshot = rec.get("snapshot", {})
        print(f"\n  [sample {i}] caseid={rec.get('caseid')} group={rec.get('surgery_group')}")
        print(f"  intervention={snapshot.get('actual_intervention')}")
        llm_out = rec.get("llm_output")
        if is_valid(llm_out):
            print(f"  llm_preview={str(llm_out)[:260]}")
        else:
            print("  llm_preview=[empty]")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Extract anesthesia intervention anchors from VitalDB and build QA training dataset"
    )
    parser.add_argument("--clinical-csv", default="clinical_information.csv")
    parser.add_argument("--output-dir", default="Anes_Dataset")
    parser.add_argument("--signal-interval-sec", type=float, default=1.0)
    parser.add_argument("--med-check-interval-sec", type=float, default=3.0)
    parser.add_argument("--window-sec", type=int, default=300, help="5 minutes = 300, 10 minutes = 600")
    parser.add_argument("--min-window-points", type=int, default=60)
    parser.add_argument("--anes-dur-min", type=float, default=30.0)
    parser.add_argument("--rate-delta-threshold", type=float, default=0.5)
    parser.add_argument("--vol-delta-threshold", type=float, default=0.03)
    parser.add_argument("--vol-rate-lookback-sec", type=float, default=60.0)
    parser.add_argument("--min-anchor-gap-sec", type=float, default=30.0)
    parser.add_argument("--disable-mbp-unit-fix", action="store_true")
    parser.add_argument("--mbp-kpa-threshold", type=float, default=20.0)
    parser.add_argument("--mbp-kpa-to-mmhg-factor", type=float, default=7.50062)
    parser.add_argument("--propofol-bolus-rate-threshold-ml-h", type=float, default=50.0)
    parser.add_argument("--propofol-bolus-min-delta-ml", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all")
    parser.add_argument("--max-anchors-per-case", type=int, default=3)
    parser.add_argument(
        "--skip-setup-rate-anchors",
        action="store_true",
        help="Skip likely pump setup/init rate anchors (e.g., 0->400 early jumps).",
    )
    parser.add_argument("--setup-rate-before-abs-max", type=float, default=1.0)
    parser.add_argument("--setup-rate-after-threshold", type=float, default=300.0)
    parser.add_argument("--setup-rate-delta-threshold", type=float, default=100.0)
    parser.add_argument("--setup-rate-early-window-sec", type=float, default=1800.0)
    parser.add_argument("--skip-medication-filter", action="store_true")
    parser.add_argument(
        "--department-include",
        default="",
        help="Comma-separated department keywords to keep (case-insensitive), e.g. 'Thoracic surgery'.",
    )
    parser.add_argument(
        "--keep-source-duplicate-rows",
        action="store_true",
        help="When source_dataset exists, keep both rows for same caseid across different sources.",
    )
    parser.add_argument(
        "--anchor-mode",
        default="medication",
        choices=["medication", "arrdb", "hybrid", "periodic"],
        help="Anchor source mode: medication deltas, arrdb labels, hybrid, or unlabeled periodic sampling.",
    )
    parser.add_argument(
        "--arrdb-annotation-dir",
        default="downloaded_results/vitaldb-arrhythmia-1.0.0/Annotation_Files",
        help="Directory that contains Annotation_file_<caseid>.csv for arrdb anchor mode.",
    )
    parser.add_argument("--arrdb-time-column", default="", help="Optional explicit arrdb time column name.")
    parser.add_argument("--arrdb-label-column", default="", help="Optional explicit arrdb label column name.")
    parser.add_argument("--arrdb-keep-normal", action="store_true", help="Keep normal/sinus labels in arrdb mode.")
    parser.add_argument(
        "--periodic-anchor-step-sec",
        type=float,
        default=300.0,
        help="For periodic mode: create one anchor every N seconds.",
    )
    parser.add_argument(
        "--periodic-anchor-start-sec",
        type=float,
        default=300.0,
        help="For periodic mode: first anchor time in seconds.",
    )
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--validate-actual-before-qa",
        action="store_true",
        help="Use LLM to validate actual_intervention first, then generate QA.",
    )
    parser.add_argument(
        "--drop-if-actual-invalid",
        action="store_true",
        help="If actual validation verdict is invalid, skip QA generation for that sample.",
    )
    parser.add_argument(
        "--drop-if-actual-uncertain",
        action="store_true",
        help="If actual validation verdict is uncertain, skip QA generation for that sample.",
    )
    parser.add_argument(
        "--actual-validation-model",
        default="",
        help="Optional model for actual validation. Empty means use --llm-model.",
    )
    parser.add_argument(
        "--actual-validation-max-tokens",
        type=int,
        default=256,
        help="Max tokens for actual-validation LLM call.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--llm-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--llm-api-key",
        default="local",
        help="Explicit key/token. For local servers can be any non-empty string",
    )
    parser.add_argument(
        "--enable-miller-rag",
        action="store_true",
        help="Enable embedding retrieval over a local Miller corpus before QA generation.",
    )
    parser.add_argument(
        "--miller-corpus-path",
        default="",
        help="Path to a licensed Miller corpus in .txt/.md or .jsonl format.",
    )
    parser.add_argument(
        "--miller-index-path",
        default="",
        help="Optional .npz cache path for Miller embeddings.",
    )
    parser.add_argument("--miller-top-k", type=int, default=3, help="Top-k Miller passages to inject into prompt.")
    parser.add_argument(
        "--miller-chunk-chars",
        type=int,
        default=1200,
        help="Chunk size in characters when corpus is plain text/markdown.",
    )
    parser.add_argument(
        "--miller-chunk-overlap-chars",
        type=int,
        default=200,
        help="Chunk overlap in characters when corpus is plain text/markdown.",
    )
    parser.add_argument(
        "--miller-max-passage-chars",
        type=int,
        default=800,
        help="Maximum characters kept for each retrieved passage in prompt injection.",
    )
    parser.add_argument(
        "--miller-bis-intent-mode",
        default="paired_only",
        choices=["full", "paired_only", "off"],
        help="How strongly BIS drives Miller retrieval intents: full / paired_only / off.",
    )
    parser.add_argument(
        "--miller-depth-focus-weight",
        type=float,
        default=0.10,
        help="Weight of depth/BIS terms in clinical_focus_score rerank (default lowered from 0.25).",
    )
    parser.add_argument(
        "--miller-require-chapter",
        action="store_true",
        help="Require retrieved passages to have a chapter locator when possible.",
    )
    parser.add_argument(
        "--miller-allowed-chapters",
        default="",
        help="Optional comma-separated chapter constraints, e.g. '21,35'.",
    )
    parser.add_argument(
        "--embedding-backend",
        default="auto",
        choices=["auto", "api", "local"],
        help="Embedding backend for Miller retrieval: local sentence-transformers or OpenAI-compatible API.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model used for Miller retrieval.",
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Device for local embedding backend, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default="",
        help="Optional embedding endpoint; empty means reuse --llm-base-url.",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default="OPENAI_API_KEY",
        help="Env var name for embedding API key when --embedding-api-key is empty.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default="",
        help="Explicit embedding API key/token. Empty means reuse --llm-api-key or env.",
    )
    parser.add_argument("--llm-max-workers", type=int, default=1, help="Parallel LLM workers.")
    parser.add_argument("--llm-progress-every", type=int, default=10, help="Print LLM progress every N records.")
    parser.add_argument(
        "--miller-retrieval-log-jsonl",
        default="",
        help="Optional JSONL path to record Miller retrieval query and Top-k evidence per sample.",
    )
    parser.add_argument(
        "--miller-retrieval-log-max-chars",
        type=int,
        default=1200,
        help="Max chars kept for each logged Miller evidence snippet.",
    )
    parser.add_argument(
        "--miller-retrieval-log-csv",
        default="",
        help="Optional CSV path to record Miller retrieval results (one row per retrieved chunk).",
    )
    parser.add_argument("--overwrite-jsonl", action="store_true")
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--export-bucketed-datasets",
        action="store_true",
        help="Export A/B/C bucketed datasets and an A/B mixed training JSONL.",
    )
    parser.add_argument(
        "--train-mix-a-ratio",
        type=float,
        default=0.8,
        help="Target A-class ratio in mixed training export (0.0-1.0).",
    )
    parser.add_argument(
        "--train-mix-seed",
        type=int,
        default=42,
        help="Random seed for A/B mixed training sampling.",
    )
    parser.add_argument(
        "--train-mix-max-samples",
        type=int,
        default=0,
        help="Max records in mixed training export; 0 means use all available A/B records.",
    )
    parser.add_argument(
        "--strict-a-requires-risk-flags",
        action="store_true",
        help="Only keep A-bucket samples when risk_flags are non-empty; otherwise downgrade to B.",
    )
    parser.add_argument(
        "--strict-a-requires-objective-evidence",
        action="store_true",
        help="Only keep A-bucket samples when objective instability exists; otherwise downgrade to B.",
    )
    args = parser.parse_args()

    group_root = os.path.join(args.output_dir, "Data")
    image_root = os.path.join(args.output_dir, "images")
    dataset_root = os.path.join(args.output_dir, "datasets")
    retrieval_log_jsonl = (
        args.miller_retrieval_log_jsonl.strip()
        if str(args.miller_retrieval_log_jsonl).strip()
        else os.path.join(dataset_root, "miller_retrieval_records.jsonl")
    )
    retrieval_log_csv = (
        args.miller_retrieval_log_csv.strip()
        if str(args.miller_retrieval_log_csv).strip()
        else os.path.join(dataset_root, "miller_retrieval_records.csv")
    )

    return PipelineConfig(
        clinical_csv=args.clinical_csv,
        output_dir=args.output_dir,
        group_root=group_root,
        image_root=image_root,
        dataset_jsonl=os.path.join(dataset_root, "anes_qa_dataset.jsonl"),
        snapshot_json=os.path.join(dataset_root, "snapshots.json"),
        llm_jsonl=os.path.join(dataset_root, "llm_outputs.jsonl"),
        miller_retrieval_log_jsonl=retrieval_log_jsonl,
        miller_retrieval_log_csv=retrieval_log_csv,
        miller_retrieval_log_max_chars=max(200, int(args.miller_retrieval_log_max_chars)),
        signal_interval_sec=args.signal_interval_sec,
        med_check_interval_sec=args.med_check_interval_sec,
        window_sec=args.window_sec,
        min_window_points=args.min_window_points,
        anes_dur_min=args.anes_dur_min,
        rate_delta_threshold=args.rate_delta_threshold,
        vol_delta_threshold=args.vol_delta_threshold,
        vol_rate_lookback_sec=args.vol_rate_lookback_sec,
        min_anchor_gap_sec=args.min_anchor_gap_sec,
        enable_mbp_unit_fix=(not args.disable_mbp_unit_fix),
        mbp_kpa_threshold=args.mbp_kpa_threshold,
        mbp_kpa_to_mmhg_factor=args.mbp_kpa_to_mmhg_factor,
        propofol_bolus_rate_threshold_ml_h=args.propofol_bolus_rate_threshold_ml_h,
        propofol_bolus_min_delta_ml=args.propofol_bolus_min_delta_ml,
        max_cases=args.max_cases,
        max_anchors_per_case=args.max_anchors_per_case,
        skip_setup_rate_anchors=args.skip_setup_rate_anchors,
        setup_rate_before_abs_max=args.setup_rate_before_abs_max,
        setup_rate_after_threshold=args.setup_rate_after_threshold,
        setup_rate_delta_threshold=args.setup_rate_delta_threshold,
        setup_rate_early_window_sec=args.setup_rate_early_window_sec,
        skip_medication_filter=args.skip_medication_filter,
        keep_source_duplicate_rows=args.keep_source_duplicate_rows,
        anchor_mode=args.anchor_mode,
        arrdb_annotation_dir=args.arrdb_annotation_dir,
        arrdb_time_column=args.arrdb_time_column,
        arrdb_label_column=args.arrdb_label_column,
        arrdb_keep_normal=args.arrdb_keep_normal,
        periodic_anchor_step_sec=args.periodic_anchor_step_sec,
        periodic_anchor_start_sec=args.periodic_anchor_start_sec,
        department_include=args.department_include,
        llm_max_workers=args.llm_max_workers,
        llm_progress_every=args.llm_progress_every,
        enable_llm=args.enable_llm,
        llm_model=args.llm_model,
        validate_actual_before_qa=args.validate_actual_before_qa,
        drop_if_actual_invalid=args.drop_if_actual_invalid,
        drop_if_actual_uncertain=args.drop_if_actual_uncertain,
        actual_validation_model=args.actual_validation_model,
        actual_validation_max_tokens=args.actual_validation_max_tokens,
        api_key_env=args.api_key_env,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        enable_miller_rag=args.enable_miller_rag,
        miller_corpus_path=args.miller_corpus_path,
        miller_index_path=args.miller_index_path,
        miller_top_k=max(1, min(5, int(args.miller_top_k))),
        miller_chunk_chars=max(300, int(args.miller_chunk_chars)),
        miller_chunk_overlap_chars=max(0, min(int(args.miller_chunk_overlap_chars), max(299, int(args.miller_chunk_chars) - 1))),
        miller_max_passage_chars=max(200, int(args.miller_max_passage_chars)),
        miller_bis_intent_mode=str(args.miller_bis_intent_mode).strip().lower(),
        miller_depth_focus_weight=max(0.0, min(0.5, float(args.miller_depth_focus_weight))),
        miller_require_chapter=bool(args.miller_require_chapter),
        miller_allowed_chapters=str(args.miller_allowed_chapters or "").strip(),
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key_env=args.embedding_api_key_env,
        embedding_api_key=args.embedding_api_key,
        overwrite_jsonl=args.overwrite_jsonl,
        sample_rate=args.sample_rate,
        random_seed=args.random_seed,
        export_bucketed_datasets=args.export_bucketed_datasets,
        train_mix_a_ratio=max(0.0, min(1.0, float(args.train_mix_a_ratio))),
        train_mix_seed=args.train_mix_seed,
        train_mix_max_samples=max(0, int(args.train_mix_max_samples)),
        strict_a_requires_risk_flags=args.strict_a_requires_risk_flags,
        strict_a_requires_objective_evidence=args.strict_a_requires_objective_evidence,
    )


def reset_image_root(image_root: str) -> None:
    if os.path.isdir(image_root):
        shutil.rmtree(image_root, ignore_errors=True)
    os.makedirs(image_root, exist_ok=True)


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.group_root, exist_ok=True)
    reset_image_root(cfg.image_root)
    os.makedirs(os.path.dirname(cfg.dataset_jsonl), exist_ok=True)

    print("=== Pipeline start ===")
    cases_df = stage1_group_and_filter(cfg)
    if cases_df.empty:
        print("No valid case after stage 1, exit.")
        return

    records = stage2_extract_snapshots(cases_df, cfg)
    if not records:
        print("No intervention anchor extracted, exit.")
        return

    stage3_generate_qa(records, cfg)
    stage4_save_dataset(records, cfg)
    stage5_sample_review(cfg)
    print("=== Pipeline finished ===")


if __name__ == "__main__":
    main()

