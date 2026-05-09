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
    "PHE_VOL": ["Orchestra/PHE_VOL", "Orchestra/PHENYLEPHRINE_VOL"],
    "EPI_RATE": [
        "Orchestra/EPI_RATE",
    ],
    "EPI_VOL": ["Orchestra/EPI_VOL", "Orchestra/EPINEPHRINE_VOL"],
    "NOR_RATE": ["Orchestra/NOR_RATE"],
    "NOR_VOL": ["Orchestra/NOR_VOL", "Orchestra/NOREPI_VOL", "Orchestra/NOREPINEPHRINE_VOL"],
    "EPH_VOL": ["Orchestra/EPH_VOL", "Orchestra/EPHE_VOL", "Orchestra/EPHEDRINE_VOL"],
    "EPH_RATE": ["Orchestra/EPH_RATE", "Orchestra/EPHE_RATE", "Orchestra/EPHEDRINE_RATE"],
    "DOPA_RATE": ["Orchestra/DOPA_RATE"],
    "DOPA_VOL": ["Orchestra/DOPA_VOL", "Orchestra/DOPAMINE_VOL"],
    "ESMO_RATE": ["Orchestra/ESMO_RATE"],
    "ESMO_VOL": ["Orchestra/ESMO_VOL", "Orchestra/ESMOLOL_VOL"],
    "NICA_RATE": ["Orchestra/NICA_RATE"],
    "NICA_VOL": ["Orchestra/NICA_VOL", "Orchestra/NICARDIPINE_VOL"],
    "NPS_RATE": ["Orchestra/NPS_RATE"],
    "NPS_VOL": ["Orchestra/NPS_VOL", "Orchestra/NITROPRUSSIDE_VOL"],
    "NTG_VOL": ["Orchestra/NTG_VOL", "Orchestra/TNG_VOL", "Orchestra/NITRO_VOL"],
    "NTG_RATE": ["Orchestra/NTG_RATE", "Orchestra/TNG_RATE", "Orchestra/NITRO_RATE"],
    "MIL_VOL": ["Orchestra/MIL_VOL", "Orchestra/MILR_VOL", "Orchestra/MILRINONE_VOL"],
    "MIL_RATE": ["Orchestra/MIL_RATE", "Orchestra/MILR_RATE", "Orchestra/MILRINONE_RATE"],
    "ATRO_VOL": ["Orchestra/ATRO_VOL", "Orchestra/ATROPINE_VOL"],
    "ATRO_RATE": ["Orchestra/ATRO_RATE", "Orchestra/ATROPINE_RATE"],
    "URA_RATE": ["Orchestra/URA_RATE"],
    "URA_VOL": ["Orchestra/URA_VOL", "Orchestra/URAPIDIL_VOL"],
    "PPF20_VOL": [
        "Orchestra/PPF20_VOL",
    ],
    "PPF20_RATE": [
        "Orchestra/PPF20_RATE",
    ],
    "REMI_VOL": [
        "Orchestra/REMI_VOL",
    ],
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
# Kept for backward compatibility in older references; all active tracks are now in MEDICATION_TRACK_CANDIDATES.
ADDITIONAL_MEDICATION_TRACK_CANDIDATES: Dict[str, List[str]] = {}

NON_PROPOFOL_MED_KEYS: List[str] = [
    "NOR_RATE",
    "NOR_VOL",
    "EPH_VOL",
    "EPH_RATE",
    "PHE_RATE",
    "PHE_VOL",
    "EPI_RATE",
    "EPI_VOL",
    "DOPA_RATE",
    "DOPA_VOL",
    "ESMO_RATE",
    "ESMO_VOL",
    "NICA_RATE",
    "NICA_VOL",
    "NPS_RATE",
    "NPS_VOL",
    "NTG_VOL",
    "NTG_RATE",
    "MIL_VOL",
    "MIL_RATE",
    "ATRO_VOL",
    "ATRO_RATE",
    "URA_RATE",
    "URA_VOL",
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
    "NOR_VOL",
    "EPH_VOL",
    "EPH_RATE",
    "PHE_RATE",
    "PHE_VOL",
    "EPI_RATE",
    "EPI_VOL",
    "DOPA_RATE",
    "DOPA_VOL",
}

ANESTHETIC_DEPTH_MED_KEYS = {
    "PPF20_VOL",
    "PPF20_RATE",
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
    "PHE_VOL": "vasopressor",
    "NOR_RATE": "vasopressor",
    "NOR_VOL": "vasopressor",
    "EPH_VOL": "vasopressor",
    "EPH_RATE": "vasopressor",
    "EPI_RATE": "inopressor",
    "EPI_VOL": "inopressor",
    "DOPA_RATE": "inopressor",
    "DOPA_VOL": "inopressor",
    "ESMO_RATE": "anti_sympathetic",
    "ESMO_VOL": "anti_sympathetic",
    "NICA_RATE": "anti_sympathetic",
    "NICA_VOL": "anti_sympathetic",
    "NPS_RATE": "anti_sympathetic",
    "NPS_VOL": "anti_sympathetic",
    "NTG_VOL": "vasodilator",
    "NTG_RATE": "vasodilator",
    "MIL_VOL": "inodilator",
    "MIL_RATE": "inodilator",
    "ATRO_VOL": "chronotropic",
    "ATRO_RATE": "chronotropic",
    "URA_RATE": "anti_sympathetic",
    "URA_VOL": "anti_sympathetic",
    "PPF20_VOL": "hypnotic_iv",
    "PPF20_RATE": "hypnotic_iv",
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
    "PHE_VOL": "phenylephrine",
    "EPH_VOL": "ephedrine",
    "EPH_RATE": "ephedrine",
    "NOR_RATE": "norepinephrine",
    "NOR_VOL": "norepinephrine",
    "EPI_RATE": "epinephrine",
    "EPI_VOL": "epinephrine",
    "DOPA_RATE": "dopamine",
    "DOPA_VOL": "dopamine",
    "ESMO_RATE": "esmolol",
    "ESMO_VOL": "esmolol",
    "NICA_RATE": "nicardipine",
    "NICA_VOL": "nicardipine",
    "NPS_RATE": "nitroprusside",
    "NPS_VOL": "nitroprusside",
    "URA_RATE": "urapidil",
    "URA_VOL": "urapidil",
    "NTG_VOL": "nitroglycerin",
    "NTG_RATE": "nitroglycerin",
    "MIL_VOL": "milrinone",
    "MIL_RATE": "milrinone",
    "ATRO_VOL": "atropine",
    "ATRO_RATE": "atropine",
    "PPF20_VOL": "propofol",
    "PPF20_RATE": "propofol",
    "REMI_VOL": "remifentanil",
    "REMI_RATE": "remifentanil",
    "RFTN20_VOL": "remifentanil",
    "RFTN50_VOL": "remifentanil",
    "RFTN20_RATE": "remifentanil",
    "RFTN50_RATE": "remifentanil",
    "ROC_VOL": "rocuronium",
    "ROC_RATE": "rocuronium",
}

LLM_MAX_TOKENS_DEFAULT = 2048
LEAK_TOKEN_RE = re.compile(
    r"(?is)\b("
    r"wait|strategy|constraint\s*check|analyze\s+the\s+input\s+data|"
    r"self-?correction|content\s+requirements|drafting|thinking\s+process|analysis:"
    r")\b"
)

Q_SUBJECTIVE_HINT_PATTERNS: Tuple[str, ...] = (
    r"提示",
    r"显示",
    r"考虑",
    r"怀疑",
    r"警惕",
    r"符合",
    r"循环稳定",
    r"循环不稳",
    r"血流动力学稳定",
    r"麻醉深度不足",
    r"麻醉过深",
    r"麻醉过浅",
)

MISSING_INDICATOR_PATTERNS: Tuple[str, ...] = (
    r"(?i)(?:MAP|MBP|SBP|DBP|HR|SpO2|SPO2|BIS|ETCO2|EtCO2|SVV|PPV|CVP|CO|CI|SV|SVR|BT|rSO2|RSO2).{0,12}(?:缺失|缺少|暂无|无有效|未提供|不可用|missing|unavailable|not\s+available|no\s+valid)",
    r"(?i)(?:缺失|缺少|暂无|无有效|未提供|不可用|missing|unavailable|not\s+available|no\s+valid).{0,12}(?:MAP|MBP|SBP|DBP|HR|SpO2|SPO2|BIS|ETCO2|EtCO2|SVV|PPV|CVP|CO|CI|SV|SVR|BT|rSO2|RSO2)",
)

# Cache available track names per case to avoid repeated metadata calls.
_CASE_TRACK_NAME_CACHE: Dict[int, Optional[set[str]]] = {}

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
    "SBP": [
        "Solar8000/ART_SBP",
        "IntelliVue/ABP_SBP",
        "SNUADC/ART_SBP",
        "Solar8000/NIBP_SBP",
        "IntelliVue/NIBP_SBP",
    ],
    "DBP": [
        "Solar8000/ART_DBP",
        "IntelliVue/ABP_DBP",
        "SNUADC/ART_DBP",
        "Solar8000/NIBP_DBP",
        "IntelliVue/NIBP_DBP",
    ],
    "SPO2": [
        "Solar8000/PLETH_SPO2",
        "Solar8000/SPO2",
        "IntelliVue/SpO2",
        "Primus/SPO2",
    ],
    "BIS": ["BIS/BIS"],
    "ETCO2": [
        "Solar8000/ETCO2",
        "Solar8000/ETCO2_MMHG",
        "Primus/ETCO2",
        "Primus/ETCO2_MMHG",
        "IntelliVue/EtCO2",
    ],
    "SVV": [
        "Vigileo/SVV",
        "EV1000/SVV",
        "Solar8000/SVV",
    ],
    "CVP": [
        "Solar8000/CVP",
        "IntelliVue/CVP",
        "SNUADC/CVP",
    ],
    "CO": [
        "Vigileo/CO",
        "EV1000/CO",
    ],
    "CI": [
        "Vigileo/CI",
        "EV1000/CI",
    ],
    "SV": [
        "Vigileo/SV",
        "EV1000/SV",
    ],
    "PPV": [
        "Vigileo/PPV",
        "EV1000/PPV",
        "Solar8000/PPV",
        "IntelliVue/PPV",
    ],
    "SVR": [
        "EV1000/SVR",
        "Vigileo/SVR",
    ],
    "BT": [
        "Solar8000/BT",
        "IntelliVue/BT",
        "SNUADC/BT",
    ],
    "RSO2_L": [
        "INVOS/rSO2_L",
        "INVOS/RSO2_L",
    ],
    "RSO2_R": [
        "INVOS/rSO2_R",
        "INVOS/RSO2_R",
    ],
}

VITAL_DISPLAY = {
    "HR": "Heart Rate (HR)",
    "MBP": "Mean Arterial Pressure (MBP)",
    "SBP": "Systolic BP (SBP)",
    "DBP": "Diastolic BP (DBP)",
    "SPO2": "SpO2",
    "BIS": "BIS",
    "ETCO2": "EtCO2",
    "SVV": "SVV",
    "CVP": "CVP",
    "CO": "Cardiac Output (CO)",
    "CI": "Cardiac Index (CI)",
    "SV": "Stroke Volume (SV)",
    "PPV": "PPV",
    "SVR": "Systemic Vascular Resistance (SVR)",
    "BT": "Body Temperature (BT)",
    "RSO2_L": "rSO2 Left",
    "RSO2_R": "rSO2 Right",
}

VITAL_UNIT = {
    "HR": "bpm",
    "MBP": "mmHg",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "SPO2": "%",
    "BIS": "",
    "ETCO2": "mmHg",
    "SVV": "%",
    "CVP": "mmHg",
    "CO": "L/min",
    "CI": "L/(min·m²)",
    "SV": "mL",
    "PPV": "%",
    "SVR": "dyn·s·cm⁻5",
    "BT": "℃",
    "RSO2_L": "%",
    "RSO2_R": "%",
}

CANONICAL_UNIT_GUIDE = {
    "MAP": "mmHg",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "HR": "bpm",
    "SpO2": "%",
    "BIS": "index",
    "EtCO2": "mmHg",
    "CO": "L/min",
    "CI": "L/(min·m²)",
    "SV": "mL",
    "SVV": "%",
    "PPV": "%",
    "CVP": "mmHg",
    "SVR": "dyn·s·cm⁻5",
    "BT": "℃",
    "rSO2": "%",
    "infusion_rate": "mL/h",
    "bolus_volume": "mL",
    "volatile_concentration": "vol%",
    "time": "s or min",
}

VITALDB_INDICATOR_SOURCE_HINTS: Dict[str, Dict[str, Any]] = {
    "EtCO2": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/ETCO2",
            "Solar8000/ETCO2_MMHG",
            "Primus/ETCO2",
            "Primus/ETCO2_MMHG",
            "IntelliVue/EtCO2",
        ],
    },
    "SpO2": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/PLETH_SPO2",
            "Solar8000/SPO2",
            "IntelliVue/SpO2",
            "Primus/SPO2",
        ],
    },
    "ECG": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/ECG_II",
            "Solar8000/ECG_V5",
            "IntelliVue/ECG_II",
            "IntelliVue/ECG_V5",
            "SNUADC/ECG_II",
            "SNUADC/ECG_V5",
        ],
    },
    "HR": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/HR",
            "IntelliVue/HR",
            "SNUADC/HR",
            "Primus/HR",
        ],
    },
    "SBP": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/ART_SBP",
            "IntelliVue/ABP_SBP",
            "SNUADC/ART_SBP",
            "Solar8000/NIBP_SBP",
            "IntelliVue/NIBP_SBP",
        ],
    },
    "DBP": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/ART_DBP",
            "IntelliVue/ABP_DBP",
            "SNUADC/ART_DBP",
            "Solar8000/NIBP_DBP",
            "IntelliVue/NIBP_DBP",
        ],
    },
    "MAP": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/ART_MBP",
            "IntelliVue/ABP_MBP",
            "SNUADC/ART_MBP",
            "Solar8000/NIBP_MBP",
            "IntelliVue/NIBP_MBP",
        ],
    },
    "BT": {
        "source_type": "waveform",
        "vitaldb_tags": [
            "Solar8000/BT",
            "IntelliVue/BT",
            "SNUADC/BT",
        ],
    },
    "BIS": {
        "source_type": "waveform",
        "vitaldb_tags": ["BIS/BIS"],
    },
    "rSO2_L": {
        "source_type": "waveform",
        "vitaldb_tags": ["INVOS/rSO2_L", "INVOS/RSO2_L"],
    },
    "rSO2_R": {
        "source_type": "waveform",
        "vitaldb_tags": ["INVOS/rSO2_R", "INVOS/RSO2_R"],
    },
    "CO": {
        "source_type": "waveform",
        "vitaldb_tags": ["Vigileo/CO", "EV1000/CO"],
    },
    "CI": {
        "source_type": "waveform",
        "vitaldb_tags": ["Vigileo/CI", "EV1000/CI"],
    },
    "SV": {
        "source_type": "waveform",
        "vitaldb_tags": ["Vigileo/SV", "EV1000/SV"],
    },
    "PPV": {
        "source_type": "waveform",
        "vitaldb_tags": ["Vigileo/PPV", "EV1000/PPV", "Solar8000/PPV", "IntelliVue/PPV"],
    },
    "SVR": {
        "source_type": "waveform",
        "vitaldb_tags": ["EV1000/SVR", "Vigileo/SVR"],
    },
    "ABG": {
        "source_type": "lab",
        "vitaldb_tags": ["Lab_results ABGA"],
    },
    "TEG": {
        "source_type": "no_direct_key",
        "vitaldb_tags": [],
        "note": "VitalDB无直接对应Key",
    },
    "ACT": {
        "source_type": "no_direct_key",
        "vitaldb_tags": [],
        "note": "VitalDB无直接对应Key",
    },
    "Urine Output": {
        "source_type": "clinical_information",
        "vitaldb_tags": ["Clinical Information intraop_uo"],
    },
    "Blood Loss": {
        "source_type": "clinical_information",
        "vitaldb_tags": ["Clinical Information intraop_ebl"],
    },
}

MEDICATION_DISPLAY = {
    "PHE_RATE": "去氧肾上腺素泵速",
    "PHE_VOL": "去氧肾上腺素累计量",
    "EPH_VOL": "麻黄碱累计量",
    "EPH_RATE": "麻黄碱泵速",
    "EPI_RATE": "肾上腺素泵速",
    "EPI_VOL": "肾上腺素累计量",
    "NOR_RATE": "去甲肾上腺素泵速",
    "NOR_VOL": "去甲肾上腺素累计量",
    "DOPA_RATE": "多巴胺泵速",
    "DOPA_VOL": "多巴胺累计量",
    "ESMO_RATE": "艾司洛尔泵速",
    "ESMO_VOL": "艾司洛尔累计量",
    "NICA_RATE": "尼卡地平泵速",
    "NICA_VOL": "尼卡地平累计量",
    "NPS_RATE": "硝普钠泵速",
    "NPS_VOL": "硝普钠累计量",
    "URA_RATE": "乌拉地尔泵速",
    "URA_VOL": "乌拉地尔累计量",
    "NTG_VOL": "硝酸甘油累计量",
    "NTG_RATE": "硝酸甘油泵速",
    "MIL_VOL": "米力农累计量",
    "MIL_RATE": "米力农泵速",
    "ATRO_VOL": "阿托品累计量",
    "ATRO_RATE": "阿托品泵速",
    "PPF20_VOL": "丙泊酚累计量",
    "PPF20_RATE": "丙泊酚速率",
    "REMI_VOL": "瑞芬太尼累计量",
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
    "sbp_low_mmhg": 90.0,
    "sbp_high_mmhg": 180.0,
    "sbp_relative_change_pct": 30.0,
    "dbp_low_mmhg": 60.0,
    "dbp_high_mmhg": 100.0,
    "dbp_relative_change_pct": 30.0,
    "hr_tachycardia_bpm": 100.0,
    "hr_bradycardia_bpm": 50.0,
    "hr_relative_change_pct": 20.0,
    "spo2_low_pct": 94.0,
    "spo2_attention_pct": 95.0,
    "spo2_attention_persist_sec": 20.0,
    "spo2_severe_low_pct": 90.0,
    "spo2_drop_from_baseline_pct": 4.0,
    "bis_light": 60.0,
    "bis_deep": 40.0,
    "etco2_low_mmhg": 30.0,
    "etco2_high_mmhg": 50.0,
    "etco2_severe_low_mmhg": 25.0,
    "etco2_severe_high_mmhg": 60.0,
    "etco2_missing_alert_sec": 2.0,
    "etco2_zeroing_value_mmhg": 2.0,
    "etco2_zeroing_hint_sec": 6.0,
    "co_low_l_min": 4.0,
    "co_high_l_min": 8.0,
    "ci_low_l_min_m2": 2.5,
    "ci_high_l_min_m2": 4.0,
    "sv_low_ml": 60.0,
    "sv_high_ml": 100.0,
    "svv_high_pct": 13.0,
    "svv_severe_high_pct": 18.0,
    "ppv_high_pct": 13.0,
    "ppv_severe_high_pct": 18.0,
    "cvp_low_mmhg": 2.0,
    "cvp_high_mmhg": 15.0,
    "svr_low_dyns_cm5": 800.0,
    "svr_high_dyns_cm5": 1600.0,
    "bt_low_c": 36.0,
    "bt_fever_c": 37.5,
    "bt_high_fever_c": 38.0,
    "rso2_low_abs_pct": 55.0,
    "rso2_warn_abs_pct": 60.0,
    "rso2_drop_from_baseline_pct": 20.0,
    "critical_window_sec": 30.0,
    "hemodynamic_window_sec": 60.0,
    "slow_trend_window_sec": 120.0,
}

ADVERSE_EVENT_CRITICAL_TYPES = {
    "major_bleeding",
    "anuria_critical",
    "hyperkalemia_critical",
    "hypokalemia_critical",
    "malignant_arrhythmia",
    "shock_pattern",
    "suspected_anaphylaxis_pattern",
    "abg_hypoxemia",
    "abg_hypercapnia",
    "abg_metabolic_acidosis_hyperlactatemia",
}

ADVERSE_EVENT_WARNING_TYPES = {
    "bleeding_warning",
    "oliguria_warning",
    "arrhythmia_event",
    "hyperglycemia_warning",
    "hyperglycemia_severe",
    "allergy_history",
    "abg_metabolic_acidosis_warning",
    "abg_be_negative_large",
    "coagulation_low",
    "coagulation_high",
    "act_abnormal",
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
Q must be objective only (background + recent physiologic values/trends + intervention question), with no subjective clinical interpretation hints.
In 【决策干预（Miller）】, use three-part structure: 诊断依据：...; 具体干预：...; 原文摘录："...[M10#...]".
In 【决策干预（VitalDB）】, output an executable action order (drug + direction + magnitude/target + reassessment time + escalation/stop condition) with normalized units.
Do not output any bullets, headings, checklists, drafting notes, or instruction echoes.
""".strip()

GOLDEN_ACTION_KEYWORDS: Dict[str, List[str]] = {
    "PHE_RATE": ["去氧肾上腺素", "苯肾上腺素", "phenylephrine"],
    "PHE_VOL": ["去氧肾上腺素", "苯肾上腺素", "phenylephrine"],
    "EPH_VOL": ["麻黄碱", "ephedrine"],
    "EPH_RATE": ["麻黄碱", "ephedrine"],
    "EPI_RATE": ["肾上腺素", "epinephrine"],
    "EPI_VOL": ["肾上腺素", "epinephrine"],
    "NOR_RATE": ["去甲肾上腺素", "norepinephrine"],
    "NOR_VOL": ["去甲肾上腺素", "norepinephrine"],
    "DOPA_RATE": ["多巴胺", "dopamine"],
    "DOPA_VOL": ["多巴胺", "dopamine"],
    "ESMO_RATE": ["艾司洛尔", "esmolol"],
    "ESMO_VOL": ["艾司洛尔", "esmolol"],
    "NICA_RATE": ["尼卡地平", "nicardipine"],
    "NICA_VOL": ["尼卡地平", "nicardipine"],
    "NPS_RATE": ["硝普钠", "nitroprusside"],
    "NPS_VOL": ["硝普钠", "nitroprusside"],
    "URA_RATE": ["乌拉地尔", "urapidil"],
    "URA_VOL": ["乌拉地尔", "urapidil"],
    "NTG_VOL": ["硝酸甘油", "nitroglycerin", "glyceryl trinitrate"],
    "NTG_RATE": ["硝酸甘油", "nitroglycerin", "glyceryl trinitrate"],
    "MIL_VOL": ["米力农", "milrinone"],
    "MIL_RATE": ["米力农", "milrinone"],
    "ATRO_VOL": ["阿托品", "atropine"],
    "ATRO_RATE": ["阿托品", "atropine"],
    "PPF20_VOL": ["丙泊酚", "propofol"],
    "PPF20_RATE": ["丙泊酚", "propofol"],
    "REMI_VOL": ["瑞芬太尼", "remifentanil"],
    "REMI_RATE": ["瑞芬太尼", "remifentanil"],
    "RFTN20_VOL": ["瑞芬太尼", "remifentanil"],
    "RFTN50_VOL": ["瑞芬太尼", "remifentanil"],
    "RFTN20_RATE": ["瑞芬太尼", "remifentanil"],
    "RFTN50_RATE": ["瑞芬太尼", "remifentanil"],
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
        "Q: 67岁男性，ASA III，胸外科维持期，近5分钟 MAP 72→58 mmHg、HR 86→102 bpm、SpO2 98→97%、BIS 52→66，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：当前关键矛盾是循环稳定性与麻醉深度的平衡。若在低灌注状态下盲目加深镇静，可能进一步加重低血压并影响器官灌注。\n"
        "【决策干预（Miller）】：诊断依据：MAP持续低于65 mmHg且BIS上行; 具体干预：先滴定升压药0.1-0.3 mL/h并2 min复评，MAP≥65 mmHg后再小步调整镇静; 原文摘录:\"treat hypotension before deepening anesthesia\" [M10#1 | 术中相关章节: Hemodynamic management | p.1493]。\n"
        "【决策干预（VitalDB）】：立即按logged_action同类升压药将泵速上调0.1-0.3 mL/h，目标MAP≥65 mmHg；2 min复评MAP/HR，若MAP仍<65 mmHg再上调同幅度，若HR>110 bpm或MAP>85 mmHg则回调0.1 mL/h。\n"
        "### End Example\n"
    ),
    "bolus_like_event": (
        "### Example (bolus_like_event)\n"
        "<think>患者短时刺激期体征上冲，单次追加药物应以短效、可回退为原则。需避免过度镇静后低血压。</think>\n"
        "Q: 54岁女性，腹部手术刺激期，近3分钟 MAP 78→84 mmHg、HR 78→108 bpm、SpO2 99→99%、BIS 47→64，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：短时、可逆的生理波动更适合短效追加干预；持续上调可能带来过量风险。需要结合血压、心率与麻醉深度的同步变化判断。\n"
        "【决策干预（Miller）】：诊断依据：BIS和HR同步上冲且MAP未低于65 mmHg; 具体干预：同类短效药单次追加0.5-1.0 mL，1-2 min复评后决定是否再追加0.5 mL; 原文摘录:\"short-acting incremental dosing with rapid reassessment\" [M10#2 | 术中相关章节: Analgesic titration | p.1521]。\n"
        "【决策干预（VitalDB）】：先按logged_action同类药物单次追加0.5-1.0 mL，再观察1-2 min；若BIS仍>60或HR>100 bpm则再追加0.5 mL，若MAP降至<65 mmHg则停止追加并改为维持泵速。\n"
        "### End Example\n"
    ),
    "arrhythmia_event": (
        "### Example (arrhythmia_event)\n"
        "<think>出现心律事件时，先判断血流动力学稳定性，再决定是否立即药理/电复律路径。麻醉深度与氧合通气也需并行评估。</think>\n"
        "Q: 69岁男性，泌尿外科术中突发心律失常标注，当前 MAP 62 mmHg、HR 42 bpm、SpO2 95%、BIS 45，且近2分钟MAP与HR均下降，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：处理顺序应先看灌注与血压稳定性，再区分可观察与需立即干预的节律。同时排查缺氧、二氧化碳潴留、电解质异常及麻醉深度不匹配。\n"
        "【决策干预（Miller）】：诊断依据：心律事件伴MAP<65 mmHg和HR<50 bpm; 具体干预：先执行不稳定节律路径并给予同类急救药物追加0.5 mL，30-60 s复评后再决定升级; 原文摘录:\"hemodynamic instability determines urgency of treatment\" [M10#1 | 术中相关章节: Perioperative arrhythmia | p.1608]。\n"
        "【决策干预（VitalDB）】：若持续MAP<65 mmHg且HR<50 bpm，先给予同类急救药物追加0.5 mL并准备升级流程，30-60 s复评；若MAP回升≥65 mmHg则转入保守滴定并每2 min复评。\n"
        "### End Example\n"
    ),
    "unlabeled_context_snapshot": (
        "### Example (unlabeled_context_snapshot)\n"
        "<think>无明确事件标签时，依据趋势而非单点，优先识别威胁灌注与氧合的指标。在信息不全时给出保守且可复评的决策。</think>\n"
        "Q: 61岁女性，骨科维持期无明确事件标签，近5分钟 MAP 70→63 mmHg、HR 76→82 bpm、SpO2 98→96%、BIS 43→41，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：应以 MAP/SpO2/HR 的连续趋势为主线，避免仅凭单一瞬时异常下结论。信息缺失时优先采取可逆、可滴定的策略。\n"
        "【决策干预（Miller）】：诊断依据：MAP持续下降并接近65 mmHg阈值; 具体干预：先小步调整同类循环支持0.1-0.2 mL/h并2 min复评，必要时再加0.1 mL/h; 原文摘录:\"use small titratable steps with frequent reassessment\" [M10#3 | 术中相关章节: Intraoperative hypotension | p.1498]。\n"
        "【决策干预（VitalDB）】：先按logged_action同类药物小步调整0.1-0.2 mL/h，目标MAP维持65-80 mmHg；2 min复评MAP/HR/SpO2，若MAP继续下降再加0.1 mL/h，若MAP>85 mmHg则回退至前一档。\n"
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
    rate_sustained_pre_window_sec: float
    rate_sustained_post_window_sec: float
    rate_sustained_min_abs_delta: float
    rate_sustained_min_ratio: float
    rate_sustained_min_points: int
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
    joint_link_max_gap_sec: float
    joint_require_med_link: bool
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
    page_chapter_map: Dict[str, Dict[str, str]]

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


def _to_snapshot_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        return text
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _collect_row_fields(row: pd.Series, keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in row:
            continue
        value = _to_snapshot_scalar(row[key])
        if value is None:
            continue
        out[str(key)] = value
    return out


def _collect_row_prefixed_fields(row: pd.Series, prefixes: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in row.index:
        key_text = str(key)
        if not any(key_text.startswith(pref) for pref in prefixes):
            continue
        value = _to_snapshot_scalar(row[key])
        if value is None:
            continue
        out[key_text] = value
    return out


def _collect_row_all_valid_fields(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in row.index:
        value = _to_snapshot_scalar(row[key])
        if value is None:
            continue
        out[str(key)] = value
    return out


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


def _normalize_page_key(value: Any) -> str:
    raw = _coerce_text(value)
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def _normalize_chapter_metadata(chapter: str, section: str) -> Tuple[str, str]:
    chapter_clean = re.sub(r"\s+", " ", _coerce_text(chapter)).strip(" .;,-")
    section_clean = re.sub(r"\s+", " ", _coerce_text(section)).strip(" .;,-")

    # Trim trailing body text accidentally attached after page numbers.
    chapter_clean = re.sub(r"\s+\d{2,4}\s+.*$", "", chapter_clean).strip(" .;,-")
    section_clean = re.sub(r"\s+\d{2,4}\s+.*$", "", section_clean).strip(" .;,-")

    # Reject obviously invalid "chapter numbers" from body-line noise.
    m = re.match(r"^\s*(\d{1,4})\b", chapter_clean)
    if m:
        num = int(m.group(1))
        if num > 150:
            chapter_clean = ""

    if chapter_clean.endswith("-"):
        chapter_clean = chapter_clean[:-1].strip()
    if section_clean.endswith("-"):
        section_clean = section_clean[:-1].strip()

    return chapter_clean, section_clean


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


def _build_page_chapter_map(passages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    per_page: Dict[str, Counter] = {}
    for item in passages:
        if not isinstance(item, dict):
            continue
        page_key = _normalize_page_key(item.get("page") or item.get("page_no") or item.get("page_index"))
        if not page_key:
            continue

        chapter = _pick_first_nonempty(item, ["chapter", "chapter_title", "chapter_name", "chapter_id"])
        section = _pick_first_nonempty(item, ["section", "section_title", "section_name"])
        if not chapter:
            infer_num, infer_title = _infer_chapter_from_text(item.get("text"))
            if infer_num and infer_title:
                chapter = f"{infer_num} {infer_title}"
                section = section or infer_title
            elif infer_num:
                chapter = infer_num
        chapter, section = _normalize_chapter_metadata(chapter, section)
        if not chapter and not section:
            continue
        if page_key not in per_page:
            per_page[page_key] = Counter()
        per_page[page_key][(chapter, section)] += 1

    page_map: Dict[str, Dict[str, str]] = {}
    for page_key, counter in per_page.items():
        if not counter:
            continue
        (chapter, section), _ = counter.most_common(1)[0]
        page_map[page_key] = {"chapter": chapter, "section": section}
    return page_map


def _apply_page_chapter_map(passages: List[Dict[str, Any]], page_map: Dict[str, Dict[str, str]]) -> None:
    if not page_map:
        return
    for item in passages:
        if not isinstance(item, dict):
            continue
        page_key = _normalize_page_key(item.get("page") or item.get("page_no") or item.get("page_index"))
        if not page_key or page_key not in page_map:
            continue
        chapter = _coerce_text(item.get("chapter"))
        section = _coerce_text(item.get("section"))
        mapped = page_map.get(page_key, {})
        mapped_chapter = _coerce_text(mapped.get("chapter"))
        mapped_section = _coerce_text(mapped.get("section"))
        if not chapter and mapped_chapter:
            item["chapter"] = mapped_chapter
        if not section and mapped_section:
            item["section"] = mapped_section


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
            chapter = f"{inferred_chapter} {inferred_title}".strip() if inferred_title else inferred_chapter
        if inferred_title and not section:
            section = inferred_title
    chapter, section = _normalize_chapter_metadata(chapter, section)
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


def _is_intraop_related(chapter: str, section: str, text: str = "") -> bool:
    low = " ".join([_coerce_text(chapter), _coerce_text(section), _coerce_text(text)]).lower()
    if not low:
        return False
    keywords = (
        "intraoperative",
        "monitor",
        "monitoring",
        "anesthesia",
        "surgery",
        "hemodynamic",
        "oxygenation",
        "airway",
        "pain",
        "nociception",
        "术中",
        "麻醉",
        "监测",
        "并发症",
        "循环",
        "氧合",
    )
    return any(k in low for k in keywords)


def _chapter_display_name(parts: Dict[str, Any]) -> str:
    chapter = _coerce_text(parts.get("chapter"))
    section = _coerce_text(parts.get("section"))
    if re.fullmatch(r"\d{1,3}", chapter) and section:
        return section
    if chapter:
        return chapter
    if section:
        return section
    return "章节定位不足"


def _chapter_from_injected_prefix(text: Any) -> str:
    src = _coerce_text(text)
    if not src:
        return ""
    match = re.search(r"\[(?:书籍|Book)[^\]]*(?:章节|Chapter)\s*[:：]\s*([^,\]|]+)", src, flags=re.IGNORECASE)
    return _coerce_text(match.group(1)) if match else ""


def _format_miller_locator(item: Dict[str, Any], rank: Any = None) -> str:
    parts = _miller_locator_parts(item)
    rank_text = str(rank if rank is not None else item.get("rank", "?")).strip() or "?"
    chapter_name = _chapter_display_name(parts)
    if chapter_name == "章节定位不足":
        chapter_name = _chapter_from_injected_prefix(item.get("text")) or chapter_name
    chapter_prefix = "术中相关章节" if _is_intraop_related(parts.get("chapter", ""), parts.get("section", ""), item.get("text", "")) else "相关章节"
    page_text = _normalize_page_key(parts.get("page")) or _normalize_page_key(item.get("page")) or "?"
    return f"[M10#{rank_text} | {chapter_prefix}: {chapter_name} | p.{page_text}]"


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
                    ("pdf_page", "pdf_page"),
                    ("page_label", "page_label"),
                    ("display_locator", "display_locator"),
                    ("locator", "display_locator"),
                    ("chapter_source", "chapter_source"),
                    ("chapter_confidence", "chapter_confidence"),
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
    page_chapter_map = _build_page_chapter_map(passages)
    _apply_page_chapter_map(passages, page_chapter_map)
    term_freqs, doc_freqs, doc_lengths, avg_doc_length = _build_bm25_state(passages)
    return MillerRetriever(
        passages=passages,
        embeddings=embeddings,
        term_freqs=term_freqs,
        doc_freqs=doc_freqs,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        page_chapter_map=page_chapter_map,
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
            r"SBP偏低（<\s*([\d.]+) mmHg，持续约([\d.]+)s）",
            r"low SBP (<\1 mmHg for about \2 s)",
        ),
        (
            r"SBP偏高（>\s*([\d.]+) mmHg，持续约([\d.]+)s）",
            r"high SBP (>\1 mmHg for about \2 s)",
        ),
        (
            r"DBP偏低（<\s*([\d.]+) mmHg，持续约([\d.]+)s）",
            r"low DBP (<\1 mmHg for about \2 s)",
        ),
        (
            r"DBP偏高（>\s*([\d.]+) mmHg，持续约([\d.]+)s）",
            r"high DBP (>\1 mmHg for about \2 s)",
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
            r"重度低氧血症（SpO2 < ([\d.]+)%，当前约([\d.]+)%）",
            r"severe hypoxemia (SpO2 < \1%, current about \2%)",
        ),
        (
            r"血氧下降（SpO2 < ([\d.]+)%，持续约([\d.]+)s）",
            r"oxygen desaturation (SpO2 < \1% for about \2 s)",
        ),
        (
            r"SpO2高敏感持续预警（[≤<]=?([\d.]+)%，持续约([\d.]+)s）",
            r"SpO2 high-sensitivity persistent warning (<=\1% for about \2 s)",
        ),
        (
            r"SpO2处于高敏感预警区（[≤<]=?([\d.]+)%），需严密观察呼吸道与通气状态",
            r"SpO2 in high-sensitivity warning zone (<=\1%); closely assess airway and ventilation",
        ),
        (
            r"SpO2较基线下降明显（([\d.]+)%）",
            r"SpO2 dropped from baseline (\1%)",
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
            r"HR较基线(升高|下降)明显（([\d.]+)%）",
            r"marked HR \1 from baseline (\2%)",
        ),
        (
            r"HR相对变化触发：变化幅度达到(临床|个体化)关注阈值（>=?([\d.]+)%）",
            r"relative HR change trigger reached threshold (>=\2%)",
        ),
        (
            r"EtCO2信号持续缺失（约([\d.]+)s，除校零外需立即排查气道/回路）",
            r"EtCO2 signal missing for about \1 s; urgent airway/circuit check required unless zeroing",
        ),
        (
            r"EtCO2信号缺失（约([\d.]+)s），前序出现近零值，疑似监测校零/复位",
            r"EtCO2 signal missing for about \1 s with preceding near-zero values, possible monitor zeroing/reset",
        ),
        (
            r"EtCO2重度异常偏低（<([\d.]+) mmHg，持续约([\d.]+)s）",
            r"severe low EtCO2 (<\1 mmHg for about \2 s)",
        ),
        (
            r"EtCO2重度异常偏高（>([\d.]+) mmHg，持续约([\d.]+)s）",
            r"severe high EtCO2 (>\1 mmHg for about \2 s)",
        ),
        (
            r"EtCO2偏低（<([\d.]+) mmHg，持续约([\d.]+)s）",
            r"low EtCO2 (<\1 mmHg for about \2 s)",
        ),
        (
            r"EtCO2偏高（>([\d.]+) mmHg，持续约([\d.]+)s）",
            r"high EtCO2 (>\1 mmHg for about \2 s)",
        ),
        (
            r"低体温（BT < ([\d.]+)℃，持续约([\d.]+)s）",
            r"hypothermia (BT < \1°C for about \2 s)",
        ),
        (
            r"发热（BT > ([\d.]+)℃，持续约([\d.]+)s）",
            r"fever (BT > \1°C for about \2 s)",
        ),
        (
            r"高热（BT ≥ ([\d.]+)℃，持续约([\d.]+)s）",
            r"high fever (BT >= \1°C for about \2 s)",
        ),
        (
            r"脑氧饱和度异常（rSO2最小约([\d.]+)% < ([\d.]+)%）",
            r"cerebral desaturation (rSO2 min about \1% < \2%)",
        ),
        (
            r"rSO2持续低值（L约([\d.]+)s, R约([\d.]+)s）",
            r"persistent low rSO2 (left about \1 s, right about \2 s)",
        ),
        (
            r"ABG低氧血症风险（PaO2≈([\d.]+) mmHg）",
            r"ABG hypoxemia risk (PaO2 about \1 mmHg)",
        ),
        (
            r"ABG二氧化碳潴留风险（PaCO2≈([\d.]+) mmHg）",
            r"ABG hypercapnia risk (PaCO2 about \1 mmHg)",
        ),
        (
            r"ABG酸中毒\+高乳酸风险（pH≈([\d.]+), Lactate≈([\d.]+)）",
            r"ABG acidosis with hyperlactatemia (pH about \1, lactate about \2)",
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
    persistence = assess.get("persistence_seconds", {}) if isinstance(assess, dict) else {}
    sensitivity_policy = assess.get("sensitivity_policy", {}) if isinstance(assess, dict) else {}
    personalized = sensitivity_policy.get("personalized_thresholds", {}) if isinstance(sensitivity_policy, dict) else {}
    context = snapshot.get("preop_context", [])
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}

    map_now = _safe_float(recent.get("MAP_mmhg"))
    sbp_now = _safe_float(recent.get("SBP_mmhg"))
    dbp_now = _safe_float(recent.get("DBP_mmhg"))
    hr_now = _safe_float(recent.get("HR_bpm"))
    spo2_now = _safe_float(recent.get("SpO2_pct"))
    bis_now = _safe_float(recent.get("BIS"))
    etco2_now = _safe_float(recent.get("EtCO2_mmhg"))
    co_now = _safe_float(recent.get("CO_L_min"))
    ci_now = _safe_float(recent.get("CI_L_min_m2"))
    sv_now = _safe_float(recent.get("SV_ml"))
    svv_now = _safe_float(recent.get("SVV_pct"))
    ppv_now = _safe_float(recent.get("PPV_pct"))
    cvp_now = _safe_float(recent.get("CVP_mmhg"))
    svr_now = _safe_float(recent.get("SVR_dyns_cm5"))
    bt_now = _safe_float(recent.get("BT_c"))
    rso2_l_now = _safe_float(recent.get("rSO2_L_pct"))
    rso2_r_now = _safe_float(recent.get("rSO2_R_pct"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))
    sbp_change_pct = _safe_float(baseline.get("SBP_change_from_baseline_pct"))
    dbp_change_pct = _safe_float(baseline.get("DBP_change_from_baseline_pct"))
    hr_change_pct = _safe_float(baseline.get("HR_change_from_baseline_pct"))
    spo2_drop_pct = _safe_float(baseline.get("SpO2_drop_from_baseline_pct"))
    rso2_l_drop_pct = _safe_float(baseline.get("rSO2_L_drop_from_baseline_pct"))
    rso2_r_drop_pct = _safe_float(baseline.get("rSO2_R_drop_from_baseline_pct"))
    bis_gt_60_sec = _safe_float(persistence.get("bis_gt_60")) or 0.0
    bis_lt_40_sec = _safe_float(persistence.get("bis_lt_40")) or 0.0
    hr_relative_limit = _safe_float(personalized.get("hr_relative_change_pct")) or float(ANES_THRESHOLDS["hr_relative_change_pct"])
    spo2_attention_limit = _safe_float(personalized.get("spo2_attention_pct")) or float(ANES_THRESHOLDS["spo2_attention_pct"])
    spo2_drop_limit = _safe_float(personalized.get("spo2_drop_from_baseline_pct")) or float(ANES_THRESHOLDS["spo2_drop_from_baseline_pct"])
    surgery_group = _coerce_text(patient.get("surgery_group"))
    surgery = _coerce_text(snapshot.get("surgery_type"))
    med_key = _coerce_text(anchor.get("medication_key"))

    bis_mode = str(getattr(cfg, "miller_bis_intent_mode", "dynamic") or "dynamic").strip().lower()
    if bis_mode not in {"full", "paired_only", "dynamic", "off"}:
        bis_mode = "dynamic"
    allow_isolated_bis = bis_mode in {"full", "dynamic"}
    allow_paired_bis = bis_mode in {"full", "paired_only", "dynamic"}

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
    if sbp_now is not None and (sbp_now < ANES_THRESHOLDS["sbp_low_mmhg"] or sbp_now > ANES_THRESHOLDS["sbp_high_mmhg"]):
        _append_intent(intents, "systolic blood pressure out of intraoperative range")
    if dbp_now is not None and (dbp_now < ANES_THRESHOLDS["dbp_low_mmhg"] or dbp_now > ANES_THRESHOLDS["dbp_high_mmhg"]):
        _append_intent(intents, "diastolic blood pressure out of intraoperative range")
    if sbp_change_pct is not None and abs(sbp_change_pct) >= ANES_THRESHOLDS["sbp_relative_change_pct"]:
        _append_intent(intents, "relative SBP change beyond 30% from baseline")
    if dbp_change_pct is not None and abs(dbp_change_pct) >= ANES_THRESHOLDS["dbp_relative_change_pct"]:
        _append_intent(intents, "relative DBP change beyond 30% from baseline")
    if hr_change_pct is not None and abs(hr_change_pct) >= hr_relative_limit:
        _append_intent(intents, "relative HR change from baseline")
    if spo2_now is not None and spo2_now < 90.0:
        _append_intent(intents, "intraoperative hypoxemia")
        _append_intent(intents, "oxygenation-first management")
    elif spo2_now is not None and spo2_now < 94.0:
        _append_intent(intents, "oxygen desaturation during anesthesia")
    elif (
        spo2_now is not None
        and spo2_now <= spo2_attention_limit
        and spo2_drop_pct is not None
        and spo2_drop_pct >= spo2_drop_limit
    ):
        _append_intent(intents, "early oxygenation warning from SpO2 baseline drop")
    if hr_now is not None and hr_now > 100.0:
        _append_intent(intents, "intraoperative tachycardia")
    elif hr_now is not None and hr_now < 50.0:
        _append_intent(intents, "intraoperative bradycardia")
    if etco2_now is not None and etco2_now < ANES_THRESHOLDS["etco2_low_mmhg"]:
        _append_intent(intents, "low end-tidal CO2 under anesthesia")
    elif etco2_now is not None and etco2_now > ANES_THRESHOLDS["etco2_high_mmhg"]:
        _append_intent(intents, "high end-tidal CO2 under anesthesia")
    if svv_now is not None and svv_now >= ANES_THRESHOLDS["svv_high_pct"]:
        _append_intent(intents, "possible hypovolemia suggested by elevated SVV")
    if ppv_now is not None and ppv_now >= ANES_THRESHOLDS["ppv_high_pct"]:
        _append_intent(intents, "possible preload responsiveness suggested by elevated PPV")
    if cvp_now is not None and cvp_now <= ANES_THRESHOLDS["cvp_low_mmhg"]:
        _append_intent(intents, "low preload signal from CVP")
    elif cvp_now is not None and cvp_now >= ANES_THRESHOLDS["cvp_high_mmhg"]:
        _append_intent(intents, "high filling pressure signal from CVP")
    if co_now is not None and co_now < ANES_THRESHOLDS["co_low_l_min"]:
        _append_intent(intents, "low cardiac output state")
    if ci_now is not None and ci_now < ANES_THRESHOLDS["ci_low_l_min_m2"]:
        _append_intent(intents, "low cardiac index with hypoperfusion risk")
    if sv_now is not None and sv_now < ANES_THRESHOLDS["sv_low_ml"]:
        _append_intent(intents, "low stroke volume physiology")
    if svr_now is not None and svr_now < ANES_THRESHOLDS["svr_low_dyns_cm5"]:
        _append_intent(intents, "systemic vasodilation pattern")
    elif svr_now is not None and svr_now > ANES_THRESHOLDS["svr_high_dyns_cm5"]:
        _append_intent(intents, "increased afterload pattern")
    if bt_now is not None and bt_now < ANES_THRESHOLDS["bt_low_c"]:
        _append_intent(intents, "perioperative hypothermia risk")
    elif bt_now is not None and bt_now >= ANES_THRESHOLDS["bt_high_fever_c"]:
        _append_intent(intents, "intraoperative high fever risk")
    rso2_vals = [v for v in [rso2_l_now, rso2_r_now] if v is not None]
    if rso2_vals and min(rso2_vals) < ANES_THRESHOLDS["rso2_low_abs_pct"]:
        _append_intent(intents, "cerebral desaturation under anesthesia")
    rso2_drop_vals = [v for v in [rso2_l_drop_pct, rso2_r_drop_pct] if v is not None]
    if rso2_drop_vals and max(rso2_drop_vals) >= ANES_THRESHOLDS["rso2_drop_from_baseline_pct"]:
        _append_intent(intents, "relative cerebral oxygenation drop from baseline")

    map_abn = map_now is not None and map_now < ANES_THRESHOLDS["map_hypotension_mmhg"]
    sbp_abn = sbp_now is not None and (sbp_now < ANES_THRESHOLDS["sbp_low_mmhg"] or sbp_now > ANES_THRESHOLDS["sbp_high_mmhg"])
    dbp_abn = dbp_now is not None and (dbp_now < ANES_THRESHOLDS["dbp_low_mmhg"] or dbp_now > ANES_THRESHOLDS["dbp_high_mmhg"])
    hr_abn = hr_now is not None and (hr_now > ANES_THRESHOLDS["hr_tachycardia_bpm"] or hr_now < ANES_THRESHOLDS["hr_bradycardia_bpm"])
    spo2_abn = spo2_now is not None and spo2_now < ANES_THRESHOLDS["spo2_low_pct"]
    etco2_abn = etco2_now is not None and (
        etco2_now < ANES_THRESHOLDS["etco2_low_mmhg"] or etco2_now > ANES_THRESHOLDS["etco2_high_mmhg"]
    )
    rso2_abn = bool(rso2_vals) and min(rso2_vals) < ANES_THRESHOLDS["rso2_low_abs_pct"]
    hemo_or_oxy_abn = map_abn or sbp_abn or dbp_abn or hr_abn or spo2_abn or etco2_abn or rso2_abn

    if allow_isolated_bis:
        isolated_bis_gate = True
        if bis_mode == "dynamic":
            if bis_now is not None and bis_now > 60.0:
                isolated_bis_gate = bool(bis_gt_60_sec >= 90.0 or bis_now >= 70.0)
            elif bis_now is not None and bis_now < 40.0:
                isolated_bis_gate = bool(bis_lt_40_sec >= 90.0 or bis_now <= 35.0)
        if bis_now is not None and bis_now > 60.0 and isolated_bis_gate:
            _append_intent(intents, "high BIS during general anesthesia")
            if not hemo_or_oxy_abn:
                _append_intent(intents, "isolated high BIS with stable hemodynamics")
                _append_intent(intents, "prevention of intraoperative awareness")
                _append_intent(intents, "hypnotic titration while preserving perfusion")
        elif bis_now is not None and bis_now < 40.0 and isolated_bis_gate:
            _append_intent(intents, "low BIS during general anesthesia")
            if not hemo_or_oxy_abn:
                _append_intent(intents, "isolated low BIS with stable hemodynamics")
                _append_intent(intents, "avoid excessive anesthetic depth")

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
    if med_key in {"PPF20_VOL", "PPF20_RATE"}:
        _append_intent(intents, "propofol adjustment during general anesthesia")
    if med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE", "MAC_RATE"}:
        _append_intent(intents, "volatile anesthetic adjustment")
    if med_key in {"PHE_RATE", "PHE_VOL", "EPH_VOL", "EPH_RATE", "NOR_RATE", "NOR_VOL", "EPI_RATE", "EPI_VOL"}:
        _append_intent(intents, "vasopressor choice during intraoperative hypotension")

    def _allow_bis_hint(hint: str) -> bool:
        low = str(hint or "").lower()
        if "bis" not in low:
            return True
        if bis_mode == "off":
            return False
        if bis_mode == "full":
            return True
        if bis_mode == "dynamic":
            if hemo_or_oxy_abn:
                return True
            if bis_gt_60_sec >= 90.0 or bis_lt_40_sec >= 90.0:
                return True
            awareness_terms = ("awareness", "hypnotic", "anesthetic depth", "stimulation", "emg")
            return any(term in low for term in awareness_terms)
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
    persistence = (
        assess.get("persistence_seconds", {})
        if isinstance(assess, dict)
        else {}
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
    sbp_now = recent.get("SBP_mmhg")
    dbp_now = recent.get("DBP_mmhg")
    hr_now = recent.get("HR_bpm")
    spo2_now = recent.get("SpO2_pct")
    bis_now = recent.get("BIS")
    map_now_f = _safe_float(map_now)
    sbp_now_f = _safe_float(sbp_now)
    dbp_now_f = _safe_float(dbp_now)
    hr_now_f = _safe_float(hr_now)
    spo2_now_f = _safe_float(spo2_now)
    bis_now_f = _safe_float(bis_now)
    bis_gt_60_sec = _safe_float(persistence.get("bis_gt_60")) or 0.0
    bis_lt_40_sec = _safe_float(persistence.get("bis_lt_40")) or 0.0
    etco2_now_f = _safe_float(recent.get("EtCO2_mmhg"))
    co_now_f = _safe_float(recent.get("CO_L_min"))
    ci_now_f = _safe_float(recent.get("CI_L_min_m2"))
    sv_now_f = _safe_float(recent.get("SV_ml"))
    ppv_now_f = _safe_float(recent.get("PPV_pct"))
    bt_now_f = _safe_float(recent.get("BT_c"))
    svr_now_f = _safe_float(recent.get("SVR_dyns_cm5"))
    rso2_l_now_f = _safe_float(recent.get("rSO2_L_pct"))
    rso2_r_now_f = _safe_float(recent.get("rSO2_R_pct"))
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = _coerce_text(anchor.get("medication_key"))
    intervention_type = _coerce_text(snapshot.get("interpreted_intervention_type"))
    intents, rewritten = rewrite_miller_query(snapshot, cfg=cfg)
    bis_mode = str(getattr(cfg, "miller_bis_intent_mode", "dynamic") or "dynamic").strip().lower()
    if bis_mode not in {"full", "paired_only", "dynamic", "off"}:
        bis_mode = "dynamic"

    map_abn = map_now_f is not None and map_now_f < ANES_THRESHOLDS["map_hypotension_mmhg"]
    sbp_abn = sbp_now_f is not None and (sbp_now_f < ANES_THRESHOLDS["sbp_low_mmhg"] or sbp_now_f > ANES_THRESHOLDS["sbp_high_mmhg"])
    dbp_abn = dbp_now_f is not None and (dbp_now_f < ANES_THRESHOLDS["dbp_low_mmhg"] or dbp_now_f > ANES_THRESHOLDS["dbp_high_mmhg"])
    hr_abn = hr_now_f is not None and (hr_now_f > ANES_THRESHOLDS["hr_tachycardia_bpm"] or hr_now_f < ANES_THRESHOLDS["hr_bradycardia_bpm"])
    spo2_abn = spo2_now_f is not None and spo2_now_f < ANES_THRESHOLDS["spo2_low_pct"]
    etco2_abn = etco2_now_f is not None and (
        etco2_now_f < ANES_THRESHOLDS["etco2_low_mmhg"] or etco2_now_f > ANES_THRESHOLDS["etco2_high_mmhg"]
    )
    rso2_vals = [v for v in [rso2_l_now_f, rso2_r_now_f] if v is not None]
    rso2_abn = bool(rso2_vals) and min(rso2_vals) < ANES_THRESHOLDS["rso2_low_abs_pct"]
    hemo_or_oxy_abn = map_abn or sbp_abn or dbp_abn or hr_abn or spo2_abn or etco2_abn or rso2_abn
    bis_high = bis_now_f is not None and bis_now_f > ANES_THRESHOLDS["bis_light"]
    bis_low = bis_now_f is not None and bis_now_f < ANES_THRESHOLDS["bis_deep"]
    bis_abn = bis_high or bis_low
    bis_coupled = bis_abn and hemo_or_oxy_abn
    bis_isolated = bis_abn and (not hemo_or_oxy_abn)

    bis_phrase = ""
    if bis_mode != "off":
        if bis_coupled:
            if bis_high and hr_now_f is not None and hr_now_f > ANES_THRESHOLDS["hr_tachycardia_bpm"]:
                bis_phrase = "high BIS with sympathetic activation under surgical stimulation, analgesic-hypnotic balance, "
            elif bis_high:
                bis_phrase = "high BIS coupled with physiologic instability; avoid automatic deepening before perfusion/oxygenation is secured, "
            elif bis_low:
                bis_phrase = "low BIS coupled with physiologic instability; assess excessive depth versus hypoperfusion, "
        elif bis_isolated and bis_mode in {"full", "dynamic"}:
            if bis_high:
                if bis_mode == "full" or bis_gt_60_sec >= 90.0 or bis_now_f >= 70.0:
                    bis_phrase = "isolated high BIS with stable hemodynamics; prevention of intraoperative awareness and cautious hypnotic titration, "
            elif bis_low:
                if bis_mode == "full" or bis_lt_40_sec >= 90.0 or bis_now_f <= 35.0:
                    bis_phrase = "isolated low BIS with stable hemodynamics; avoid excessive anesthetic depth, "

    recent_parts: List[str] = []
    if map_now_f is not None:
        recent_parts.append(f"MAP {map_now} mmHg")
    if sbp_now_f is not None:
        recent_parts.append(f"SBP {sbp_now} mmHg")
    if dbp_now_f is not None:
        recent_parts.append(f"DBP {dbp_now} mmHg")
    if hr_now_f is not None:
        recent_parts.append(f"HR {hr_now} bpm")
    if spo2_now_f is not None:
        recent_parts.append(f"SpO2 {spo2_now}%")
    if etco2_now_f is not None:
        recent_parts.append(f"EtCO2 {etco2_now_f:.1f} mmHg")
    if co_now_f is not None:
        recent_parts.append(f"CO {co_now_f:.1f} L/min")
    if ci_now_f is not None:
        recent_parts.append(f"CI {ci_now_f:.1f} L/(min·m²)")
    if sv_now_f is not None:
        recent_parts.append(f"SV {sv_now_f:.0f} mL")
    if ppv_now_f is not None:
        recent_parts.append(f"PPV {ppv_now_f:.1f}%")
    if svr_now_f is not None:
        recent_parts.append(f"SVR {svr_now_f:.0f} dyn·s·cm⁻5")
    if bt_now_f is not None:
        recent_parts.append(f"BT {bt_now_f:.1f}℃")
    if rso2_l_now_f is not None:
        recent_parts.append(f"rSO2_L {rso2_l_now_f:.1f}%")
    if rso2_r_now_f is not None:
        recent_parts.append(f"rSO2_R {rso2_r_now_f:.1f}%")
    if bis_now_f is not None:
        recent_parts.append(f"BIS {bis_now}")
    recent_state_text = ", ".join(recent_parts) if recent_parts else "available physiologic signals not provided"

    return (
        "Perioperative anesthesia evidence retrieval query: "
        f"{age}-year-old {sex}, ASA {asa}, department {department}, surgery group {surgery_group}, "
        f"undergoing {surgery}. Current stage: {stage}. "
        f"Preoperative context: {ctx_text}. "
        f"Recent mean physiologic state: {recent_state_text}. "
        f"Risk flags: {risk_text}. "
        f"Clinical interpretation: {interp_text}. "
        f"Intent tags: {'; '.join(intents)}. "
        f"Rewritten retrieval focus: {rewritten}. "
        f"Candidate intervention context for retrieval disambiguation: medication key {med_key}, intervention type {intervention_type}; prioritize physiologic evidence first. "
        "Retrieve the most relevant Miller anesthesia evidence on anesthetic depth adjustment, analgesic titration, "
        f"hemodynamic safety thresholds, perfusion-first priority, oxygenation-first priority, {bis_phrase}"
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
            ranked["display_locator"] = _format_miller_locator(ranked, rank=rank)
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
        item["display_locator"] = _format_miller_locator(item, rank=rank)
    for bucket in (bm25_hits[: cfg.miller_top_k], dense_hits[: cfg.miller_top_k]):
        for item in bucket:
            item["text"] = _coerce_text(item.get("text"))[: cfg.miller_max_passage_chars]
            item["display_locator"] = _format_miller_locator(item, rank=item.get("rank"))
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


def _get_case_available_track_names(caseid: int) -> Optional[set[str]]:
    if int(caseid) in _CASE_TRACK_NAME_CACHE:
        return _CASE_TRACK_NAME_CACHE[int(caseid)]
    names: Optional[set[str]] = None
    try:
        if hasattr(vitaldb, "api") and hasattr(vitaldb.api, "get_track_names"):
            fn = vitaldb.api.get_track_names
            raw = None
            # Support both signatures observed in wrappers.
            try:
                raw = fn([int(caseid)])
            except Exception:
                raw = fn(int(caseid))
            if isinstance(raw, dict):
                # e.g., {caseid: [tracks]}
                vals = raw.get(int(caseid)) or raw.get(str(caseid))
                if isinstance(vals, list):
                    names = {str(x) for x in vals if str(x).strip()}
            elif isinstance(raw, list):
                if raw and isinstance(raw[0], dict):
                    # e.g., [{"caseid":xx, "tname":"Solar8000/HR"}, ...]
                    out = set()
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        tname = item.get("tname") or item.get("track_name") or item.get("name")
                        c = item.get("caseid")
                        if tname and (c is None or int(c) == int(caseid)):
                            out.add(str(tname))
                    names = out if out else None
                else:
                    # e.g., ["Solar8000/HR", ...]
                    names = {str(x) for x in raw if str(x).strip()}
    except Exception:
        names = None
    _CASE_TRACK_NAME_CACHE[int(caseid)] = names
    return names


def _filter_tracks_for_case(caseid: int, requested_tracks: Sequence[str]) -> List[str]:
    req = [str(t) for t in requested_tracks if str(t).strip()]
    available = _get_case_available_track_names(caseid)
    if not available:
        return list(dict.fromkeys(req))
    kept = [t for t in req if t in available]
    return list(dict.fromkeys(kept)) if kept else list(dict.fromkeys(req))


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
                "pdf_page": item.get("pdf_page"),
                "page_label": item.get("page_label"),
                "chapter_source": item.get("chapter_source"),
                "chapter_confidence": item.get("chapter_confidence"),
                "line_no": locator_parts.get("line_no"),
                "locator": _format_miller_locator(item, rank=item.get("rank")),
                "display_locator": _coerce_text(item.get("display_locator")) or _format_miller_locator(item, rank=item.get("rank")),
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
                "pdf_page": item.get("pdf_page"),
                "page_label": item.get("page_label"),
                "chapter_source": item.get("chapter_source"),
                "chapter_confidence": item.get("chapter_confidence"),
                "line_no": item.get("line_no"),
                "locator": item.get("locator"),
                "display_locator": item.get("display_locator"),
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
        s = s.where((s >= 20.0) & (s <= 220.0))
    elif key == "SBP":
        s = s.where((s >= 40.0) & (s <= 260.0))
    elif key == "DBP":
        s = s.where((s >= 20.0) & (s <= 180.0))
    elif key == "HR":
        s = s.where((s >= 20.0) & (s <= 220.0))
    elif key == "SPO2":
        s = s.where((s >= 50.0) & (s <= 100.0))
    elif key == "BIS":
        s = s.where((s >= 1.0) & (s <= 100.0))
    elif key == "ETCO2":
        s = s.where((s >= 0.0) & (s <= 100.0))
    elif key == "SVV":
        s = s.where((s >= 0.0) & (s <= 80.0))
    elif key == "PPV":
        s = s.where((s >= 0.0) & (s <= 80.0))
    elif key == "CVP":
        s = s.where((s >= -5.0) & (s <= 45.0))
    elif key == "CO":
        s = s.where((s >= 0.5) & (s <= 20.0))
    elif key == "CI":
        s = s.where((s >= 0.5) & (s <= 10.0))
    elif key == "SV":
        s = s.where((s >= 10.0) & (s <= 250.0))
    elif key == "SVR":
        s = s.where((s >= 100.0) & (s <= 5000.0))
    elif key == "BT":
        s = s.where((s >= 30.0) & (s <= 43.0))
    elif key in {"RSO2_L", "RSO2_R"}:
        s = s.where((s >= 15.0) & (s <= 100.0))

    # Artifact suppression:
    # 1) rolling-median baseline to reduce high-frequency spike noise;
    # 2) remove one-point "impulse" spikes that recover immediately.
    win = 5
    med = s.rolling(window=win, center=True, min_periods=3).median()
    residual = (s - med).abs()
    mad = residual.rolling(window=win, center=True, min_periods=3).median()
    scale = mad * 1.4826
    # Base tolerance prevents over-filtering when MAD is near zero.
    base_tol_by_key = {
        "MBP": 20.0,
        "SBP": 25.0,
        "DBP": 15.0,
        "HR": 20.0,
        "SPO2": 4.0,
        "BIS": 10.0,
        "ETCO2": 8.0,
        "CVP": 8.0,
    }
    base_tol = float(base_tol_by_key.get(str(key), 10.0))
    dyn_th = (6.0 * scale).fillna(0.0)
    threshold = np.maximum(dyn_th.to_numpy(dtype=float), base_tol)

    prev_v = s.shift(1)
    next_v = s.shift(-1)
    has_neighbors = prev_v.notna() & next_v.notna()
    fast_recovery = (prev_v - next_v).abs() <= (0.5 * threshold)
    spike_like = has_neighbors & fast_recovery & (residual > threshold)
    s = s.mask(spike_like)
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
    # Avoid mixed-type chunk inference warnings on wide multisource CSVs.
    df = pd.read_csv(path, low_memory=False)
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
    med_tracks = _filter_tracks_for_case(caseid, med_tracks)
    if not med_tracks:
        return None
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

    if cfg.anchor_mode in ("arrdb", "hybrid", "periodic", "joint"):
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
    tracks = _filter_tracks_for_case(caseid, tracks)
    if not tracks:
        return None
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


def _series_window_median(
    value_series: pd.Series,
    time_series: pd.Series,
    start_sec: float,
    end_sec: float,
) -> Tuple[Optional[float], int]:
    if end_sec <= start_sec:
        return None, 0
    sub = pd.DataFrame({"t": time_series, "v": value_series}).dropna()
    if sub.empty:
        return None, 0
    m = sub[(sub["t"] >= float(start_sec)) & (sub["t"] <= float(end_sec))]
    if m.empty:
        return None, 0
    return float(m["v"].median()), int(len(m))


def _rate_anchor_is_sustained(
    value_series: pd.Series,
    time_series: pd.Series,
    anchor_idx: int,
    instant_delta: float,
    cfg: PipelineConfig,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    t_now = _safe_float(time_series.iloc[int(anchor_idx)]) if len(time_series) > int(anchor_idx) else None
    if t_now is None:
        return False, None, None, None

    pre_med, pre_n = _series_window_median(
        value_series=value_series,
        time_series=time_series,
        start_sec=float(t_now) - float(cfg.rate_sustained_pre_window_sec),
        end_sec=float(t_now),
    )
    post_med, post_n = _series_window_median(
        value_series=value_series,
        time_series=time_series,
        start_sec=float(t_now),
        end_sec=float(t_now) + float(cfg.rate_sustained_post_window_sec),
    )
    min_n = int(cfg.rate_sustained_min_points)
    if pre_med is None or post_med is None or pre_n < min_n or post_n < min_n:
        return False, pre_med, post_med, None

    sustained_delta = float(post_med - pre_med)
    min_delta = max(float(cfg.rate_delta_threshold), float(cfg.rate_sustained_min_abs_delta))
    if abs(sustained_delta) < min_delta:
        return False, pre_med, post_med, sustained_delta
    if sustained_delta * float(instant_delta) <= 0:
        return False, pre_med, post_med, sustained_delta

    sub = pd.DataFrame({"t": time_series, "v": value_series}).dropna()
    post = sub[(sub["t"] >= float(t_now)) & (sub["t"] <= float(t_now) + float(cfg.rate_sustained_post_window_sec))]
    if post.empty:
        return False, pre_med, post_med, sustained_delta

    # Directional consistency: most post-window values should remain on the new side.
    if sustained_delta > 0:
        consistent = float((post["v"] >= pre_med + 0.5 * min_delta).mean())
    else:
        consistent = float((post["v"] <= pre_med - 0.5 * min_delta).mean())
    if consistent < float(cfg.rate_sustained_min_ratio):
        return False, pre_med, post_med, sustained_delta
    return True, pre_med, post_med, sustained_delta


def _paired_volume_at_anchor(
    df: pd.DataFrame,
    med_key: str,
    time_sec: float,
    search_half_window_sec: float = 5.0,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    if not med_key.endswith("_RATE"):
        return None, None, None
    vol_key = med_key.replace("_RATE", "_VOL")
    cands = medication_track_candidates().get(vol_key, [])
    if not cands:
        return vol_key, None, None
    vol_col = resolve_column(df, cands)
    if vol_col is None or vol_col not in df.columns or "Time" not in df.columns:
        return vol_key, None, None
    t = pd.to_numeric(df["Time"], errors="coerce")
    v = pd.to_numeric(df[vol_col], errors="coerce")
    sub = pd.DataFrame({"t": t, "v": v}).dropna()
    if sub.empty:
        return vol_key, vol_col, None
    near = sub[(sub["t"] >= float(time_sec) - float(search_half_window_sec)) & (sub["t"] <= float(time_sec) + float(search_half_window_sec))]
    if near.empty:
        idx = (sub["t"] - float(time_sec)).abs().idxmin()
        return vol_key, vol_col, float(sub.loc[idx, "v"])
    idx = (near["t"] - float(time_sec)).abs().idxmin()
    return vol_key, vol_col, float(near.loc[idx, "v"])


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
            sustained_pre_median = None
            sustained_post_median = None
            sustained_delta = None
            paired_volume_key = None
            paired_volume_track = None
            paired_volume_ml = None
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
            elif (
                med_key.endswith("_RATE")
                and med_key not in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE", "MAC_RATE"}
                and str(col).startswith("Orchestra/")
                and time_series is not None
            ):
                ok, pre_med, post_med, sus_delta = _rate_anchor_is_sustained(
                    value_series=s,
                    time_series=time_series,
                    anchor_idx=int(i),
                    instant_delta=d,
                    cfg=cfg,
                )
                if not ok:
                    continue
                sustained_pre_median = pre_med
                sustained_post_median = post_med
                sustained_delta = sus_delta
                paired_volume_key, paired_volume_track, paired_volume_ml = _paired_volume_at_anchor(
                    df=df,
                    med_key=med_key,
                    time_sec=t,
                )

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
                "sustained_pre_median": sustained_pre_median,
                "sustained_post_median": sustained_post_median,
                "sustained_delta": sustained_delta,
                "paired_volume_key": paired_volume_key,
                "paired_volume_track": paired_volume_track,
                "paired_volume_ml": paired_volume_ml,
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


def _format_value_with_unit(value: Any, unit: str, digits: int = 1) -> str:
    val = _safe_float(value)
    if val is None:
        return "NA"
    if not unit:
        return f"{val:.{digits}f}"
    return f"{val:.{digits}f} {unit}"


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
    start = _format_value_with_unit(summary["start"], unit)
    end = _format_value_with_unit(summary["end"], unit)
    mean = _format_value_with_unit(summary["mean"], unit)
    min_v = _format_value_with_unit(summary["min"], unit)
    max_v = _format_value_with_unit(summary["max"], unit)
    return (
        f"from {start} {trend} to {end}; "
        f"mean {mean}, range [{min_v}, {max_v}]"
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


def _tail_missing_duration_by_vital(df_window: pd.DataFrame, vital_key: str) -> float:
    if "Time" not in df_window.columns:
        return 0.0
    col = resolve_vital_column(df_window, vital_key)
    if col is None:
        return 0.0
    t = pd.to_numeric(df_window["Time"], errors="coerce")
    s = _physio_filter_series(pd.to_numeric(df_window[col], errors="coerce"), key=vital_key)
    valid_time = t.notna()
    if not valid_time.any():
        return 0.0
    tvals = t[valid_time].to_numpy(dtype=float)
    missing_mask = s[valid_time].isna().to_numpy(dtype=bool)
    return _tail_condition_duration_sec(tvals=tvals, mask_vals=missing_mask)


def _safe_get_series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    col = resolve_vital_column(df, key)
    if col is None:
        return None
    return df[col]


def _build_vitaldb_track_map(
    df_window: pd.DataFrame,
    row: pd.Series,
) -> Dict[str, Any]:
    track_map: Dict[str, Any] = {}
    key_alias = {
        "MBP": "MAP",
        "SPO2": "SpO2",
        "ETCO2": "EtCO2",
        "RSO2_L": "rSO2_L",
        "RSO2_R": "rSO2_R",
    }

    for vital_key, cands in VITAL_TRACK_CANDIDATES.items():
        col = resolve_vital_column(df_window, vital_key)
        public_key = key_alias.get(vital_key, vital_key)
        item: Dict[str, Any] = {
            "source_type": "waveform",
            "candidate_tags": cands,
            "resolved_track": col if col else "not_available",
        }
        if col and col in df_window.columns:
            valid_n = int(pd.to_numeric(df_window[col], errors="coerce").dropna().shape[0])
            item["valid_points_in_window"] = valid_n
        track_map[public_key] = item

    # ECG uses explicit tags and is not part of VITAL_TRACK_CANDIDATES.
    ecg_candidates = [
        "Solar8000/ECG_II",
        "Solar8000/ECG_V5",
        "IntelliVue/ECG_II",
        "IntelliVue/ECG_V5",
        "SNUADC/ECG_II",
        "SNUADC/ECG_V5",
    ]
    ecg_resolved = [c for c in ecg_candidates if c in df_window.columns]
    track_map["ECG"] = {
        "source_type": "waveform",
        "candidate_tags": ecg_candidates,
        "resolved_track": ecg_resolved if ecg_resolved else "not_available",
        "valid_points_in_window": int(
            sum(pd.to_numeric(df_window[c], errors="coerce").dropna().shape[0] for c in ecg_resolved)
        )
        if ecg_resolved
        else 0,
    }

    track_map["ABG"] = {
        "source_type": "lab",
        "candidate_tags": ["Lab_results ABGA"],
        "resolved_track": "from_clinical_table_fields",
        "fields": ["abga_pao2", "abga_paco2", "abga_ph", "abga_lactate", "abga_k", "abga_be"],
    }
    track_map["TEG"] = {
        "source_type": "no_direct_key",
        "candidate_tags": [],
        "resolved_track": "no_direct_key",
    }
    track_map["ACT"] = {
        "source_type": "no_direct_key",
        "candidate_tags": [],
        "resolved_track": "no_direct_key",
    }

    uo_keys = ["intraop_uo", "uo", "urine_output_ml"]
    ebl_keys = ["intraop_ebl", "ebl", "blood_loss_ml"]
    resolved_uo = next((k for k in uo_keys if k in row.index and _safe_float(row.get(k)) is not None), None)
    resolved_ebl = next((k for k in ebl_keys if k in row.index and _safe_float(row.get(k)) is not None), None)
    track_map["Urine Output"] = {
        "source_type": "clinical_information",
        "candidate_tags": ["Clinical Information intraop_uo"],
        "resolved_track": resolved_uo if resolved_uo else "not_available",
    }
    track_map["Blood Loss"] = {
        "source_type": "clinical_information",
        "candidate_tags": ["Clinical Information intraop_ebl"],
        "resolved_track": resolved_ebl if resolved_ebl else "not_available",
    }

    return track_map


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
    sbp_baseline = _median_in_time_window(df_case, "SBP", base_start, base_end)
    dbp_baseline = _median_in_time_window(df_case, "DBP", base_start, base_end)
    hr_baseline = _median_in_time_window(df_case, "HR", base_start, base_end)
    spo2_baseline = _median_in_time_window(df_case, "SPO2", base_start, base_end)
    etco2_baseline = _median_in_time_window(df_case, "ETCO2", base_start, base_end)
    co_baseline = _median_in_time_window(df_case, "CO", base_start, base_end)
    ci_baseline = _median_in_time_window(df_case, "CI", base_start, base_end)
    sv_baseline = _median_in_time_window(df_case, "SV", base_start, base_end)
    ppv_baseline = _median_in_time_window(df_case, "PPV", base_start, base_end)
    svr_baseline = _median_in_time_window(df_case, "SVR", base_start, base_end)
    bt_baseline = _median_in_time_window(df_case, "BT", base_start, base_end)
    rso2_l_baseline = _median_in_time_window(df_case, "RSO2_L", base_start, base_end)
    rso2_r_baseline = _median_in_time_window(df_case, "RSO2_R", base_start, base_end)

    mbp_current = None
    sbp_current = None
    dbp_current = None
    hr_current = None
    spo2_current = None
    etco2_current = None
    co_current = None
    ci_current = None
    sv_current = None
    ppv_current = None
    svr_current = None
    bt_current = None
    rso2_l_current = None
    rso2_r_current = None
    mbp_s = _safe_get_series(df_window, "MBP")
    sbp_s = _safe_get_series(df_window, "SBP")
    dbp_s = _safe_get_series(df_window, "DBP")
    hr_s = _safe_get_series(df_window, "HR")
    spo2_s = _safe_get_series(df_window, "SPO2")
    etco2_s = _safe_get_series(df_window, "ETCO2")
    co_s = _safe_get_series(df_window, "CO")
    ci_s = _safe_get_series(df_window, "CI")
    sv_s = _safe_get_series(df_window, "SV")
    ppv_s = _safe_get_series(df_window, "PPV")
    svr_s = _safe_get_series(df_window, "SVR")
    bt_s = _safe_get_series(df_window, "BT")
    rso2_l_s = _safe_get_series(df_window, "RSO2_L")
    rso2_r_s = _safe_get_series(df_window, "RSO2_R")
    if mbp_s is not None:
        mbp_current = _last_window_mean(mbp_s, vital_key="MBP")
    if sbp_s is not None:
        sbp_current = _last_window_mean(sbp_s, vital_key="SBP")
    if dbp_s is not None:
        dbp_current = _last_window_mean(dbp_s, vital_key="DBP")
    if hr_s is not None:
        hr_current = _last_window_mean(hr_s, vital_key="HR")
    if spo2_s is not None:
        spo2_current = _last_window_mean(spo2_s, vital_key="SPO2")
    if etco2_s is not None:
        etco2_current = _last_window_mean(etco2_s, vital_key="ETCO2")
    if co_s is not None:
        co_current = _last_window_mean(co_s, vital_key="CO")
    if ci_s is not None:
        ci_current = _last_window_mean(ci_s, vital_key="CI")
    if sv_s is not None:
        sv_current = _last_window_mean(sv_s, vital_key="SV")
    if ppv_s is not None:
        ppv_current = _last_window_mean(ppv_s, vital_key="PPV")
    if svr_s is not None:
        svr_current = _last_window_mean(svr_s, vital_key="SVR")
    if bt_s is not None:
        bt_current = _last_window_mean(bt_s, vital_key="BT")
    if rso2_l_s is not None:
        rso2_l_current = _last_window_mean(rso2_l_s, vital_key="RSO2_L")
    if rso2_r_s is not None:
        rso2_r_current = _last_window_mean(rso2_r_s, vital_key="RSO2_R")

    map_drop_pct = None
    sbp_change_pct = None
    dbp_change_pct = None
    hr_change_pct = None
    spo2_drop_pct = None
    etco2_change_pct = None
    co_change_pct = None
    ci_change_pct = None
    sv_change_pct = None
    ppv_change_pct = None
    svr_change_pct = None
    rso2_l_drop_pct = None
    rso2_r_drop_pct = None
    if mbp_baseline is not None and mbp_current is not None and mbp_baseline > 0:
        map_drop_pct = float((mbp_baseline - mbp_current) / mbp_baseline * 100.0)
    if sbp_baseline is not None and sbp_current is not None and sbp_baseline > 0:
        sbp_change_pct = float((sbp_current - sbp_baseline) / sbp_baseline * 100.0)
    if dbp_baseline is not None and dbp_current is not None and dbp_baseline > 0:
        dbp_change_pct = float((dbp_current - dbp_baseline) / dbp_baseline * 100.0)
    if hr_baseline is not None and hr_current is not None and hr_baseline > 0:
        hr_change_pct = float((hr_current - hr_baseline) / hr_baseline * 100.0)
    if spo2_baseline is not None and spo2_current is not None and spo2_baseline > 0:
        spo2_drop_pct = float((spo2_baseline - spo2_current) / spo2_baseline * 100.0)
    if etco2_baseline is not None and etco2_current is not None and etco2_baseline > 0:
        etco2_change_pct = float((etco2_current - etco2_baseline) / etco2_baseline * 100.0)
    if co_baseline is not None and co_current is not None and co_baseline > 0:
        co_change_pct = float((co_current - co_baseline) / co_baseline * 100.0)
    if ci_baseline is not None and ci_current is not None and ci_baseline > 0:
        ci_change_pct = float((ci_current - ci_baseline) / ci_baseline * 100.0)
    if sv_baseline is not None and sv_current is not None and sv_baseline > 0:
        sv_change_pct = float((sv_current - sv_baseline) / sv_baseline * 100.0)
    if ppv_baseline is not None and ppv_current is not None and ppv_baseline > 0:
        ppv_change_pct = float((ppv_current - ppv_baseline) / ppv_baseline * 100.0)
    if svr_baseline is not None and svr_current is not None and svr_baseline > 0:
        svr_change_pct = float((svr_current - svr_baseline) / svr_baseline * 100.0)
    if rso2_l_baseline is not None and rso2_l_current is not None and rso2_l_baseline > 0:
        rso2_l_drop_pct = float((rso2_l_baseline - rso2_l_current) / rso2_l_baseline * 100.0)
    if rso2_r_baseline is not None and rso2_r_current is not None and rso2_r_baseline > 0:
        rso2_r_drop_pct = float((rso2_r_baseline - rso2_r_current) / rso2_r_baseline * 100.0)

    return {
        "baseline_window_start_sec": float(base_start),
        "baseline_window_end_sec": float(base_end),
        "MAP_baseline_mmhg": mbp_baseline,
        "MAP_current_mmhg": mbp_current,
        "MAP_drop_from_baseline_pct": map_drop_pct,
        "SBP_baseline_mmhg": sbp_baseline,
        "SBP_current_mmhg": sbp_current,
        "SBP_change_from_baseline_pct": sbp_change_pct,
        "DBP_baseline_mmhg": dbp_baseline,
        "DBP_current_mmhg": dbp_current,
        "DBP_change_from_baseline_pct": dbp_change_pct,
        "HR_baseline_bpm": hr_baseline,
        "HR_current_bpm": hr_current,
        "HR_change_from_baseline_pct": hr_change_pct,
        "SpO2_baseline_pct": spo2_baseline,
        "SpO2_current_pct": spo2_current,
        "SpO2_drop_from_baseline_pct": spo2_drop_pct,
        "EtCO2_baseline_mmhg": etco2_baseline,
        "EtCO2_current_mmhg": etco2_current,
        "EtCO2_change_from_baseline_pct": etco2_change_pct,
        "CO_baseline_L_min": co_baseline,
        "CO_current_L_min": co_current,
        "CO_change_from_baseline_pct": co_change_pct,
        "CI_baseline_L_min_m2": ci_baseline,
        "CI_current_L_min_m2": ci_current,
        "CI_change_from_baseline_pct": ci_change_pct,
        "SV_baseline_ml": sv_baseline,
        "SV_current_ml": sv_current,
        "SV_change_from_baseline_pct": sv_change_pct,
        "PPV_baseline_pct": ppv_baseline,
        "PPV_current_pct": ppv_current,
        "PPV_change_from_baseline_pct": ppv_change_pct,
        "SVR_baseline_dyns_cm5": svr_baseline,
        "SVR_current_dyns_cm5": svr_current,
        "SVR_change_from_baseline_pct": svr_change_pct,
        "BT_baseline_c": bt_baseline,
        "BT_current_c": bt_current,
        "rSO2_L_baseline_pct": rso2_l_baseline,
        "rSO2_L_current_pct": rso2_l_current,
        "rSO2_L_drop_from_baseline_pct": rso2_l_drop_pct,
        "rSO2_R_baseline_pct": rso2_r_baseline,
        "rSO2_R_current_pct": rso2_r_current,
        "rSO2_R_drop_from_baseline_pct": rso2_r_drop_pct,
    }


def _build_personalized_thresholds(
    baseline_comparison: Optional[Dict[str, Optional[float]]],
) -> Dict[str, float]:
    out = {
        "map_low_mmhg": float(ANES_THRESHOLDS["map_hypotension_mmhg"]),
        "hr_tachycardia_bpm": float(ANES_THRESHOLDS["hr_tachycardia_bpm"]),
        "hr_bradycardia_bpm": float(ANES_THRESHOLDS["hr_bradycardia_bpm"]),
        "map_relative_drop_pct": float(ANES_THRESHOLDS["map_relative_drop_pct"]),
        "sbp_relative_change_pct": float(ANES_THRESHOLDS["sbp_relative_change_pct"]),
        "dbp_relative_change_pct": float(ANES_THRESHOLDS["dbp_relative_change_pct"]),
        "hr_relative_change_pct": float(ANES_THRESHOLDS["hr_relative_change_pct"]),
        "spo2_low_pct": float(ANES_THRESHOLDS["spo2_severe_low_pct"]),
        "spo2_attention_pct": float(ANES_THRESHOLDS["spo2_attention_pct"]),
        "spo2_drop_from_baseline_pct": float(ANES_THRESHOLDS["spo2_drop_from_baseline_pct"]),
        "spo2_attention_persist_sec": float(ANES_THRESHOLDS["spo2_attention_persist_sec"]),
        "etco2_missing_alert_sec": float(ANES_THRESHOLDS["etco2_missing_alert_sec"]),
        "bt_low_c": float(ANES_THRESHOLDS["bt_low_c"]),
        "bt_high_fever_c": float(ANES_THRESHOLDS["bt_fever_c"]),
        "bt_high_critical_c": float(ANES_THRESHOLDS["bt_high_fever_c"]),
        "bis_low": float(ANES_THRESHOLDS["bis_deep"]),
        "bis_high": float(ANES_THRESHOLDS["bis_light"]),
        "rso2_low_pct": float(ANES_THRESHOLDS["rso2_low_abs_pct"]),
        "rso2_drop_from_baseline_pct": float(ANES_THRESHOLDS["rso2_drop_from_baseline_pct"]),
        "co_low": float(ANES_THRESHOLDS["co_low_l_min"]),
        "co_high": float(ANES_THRESHOLDS["co_high_l_min"]),
        "ci_low": float(ANES_THRESHOLDS["ci_low_l_min_m2"]),
        "ci_high": float(ANES_THRESHOLDS["ci_high_l_min_m2"]),
        "sv_low": float(ANES_THRESHOLDS["sv_low_ml"]),
        "sv_high": float(ANES_THRESHOLDS["sv_high_ml"]),
        "svv_high": float(ANES_THRESHOLDS["svv_high_pct"]),
        "cvp_low": float(ANES_THRESHOLDS["cvp_low_mmhg"]),
        "cvp_high": float(ANES_THRESHOLDS["cvp_high_mmhg"]),
        "svr_low": float(ANES_THRESHOLDS["svr_low_dyns_cm5"]),
        "svr_high": float(ANES_THRESHOLDS["svr_high_dyns_cm5"]),
        "abg_missing_alert_sec": 8.0,
    }
    if not baseline_comparison:
        return out

    hr_baseline = _safe_float(baseline_comparison.get("HR_baseline_bpm"))
    spo2_baseline = _safe_float(baseline_comparison.get("SpO2_baseline_pct"))
    rso2_l_baseline = _safe_float(baseline_comparison.get("rSO2_L_baseline_pct"))
    rso2_r_baseline = _safe_float(baseline_comparison.get("rSO2_R_baseline_pct"))

    # Patient-personalized HR change threshold: tachy/brady-prone baselines are monitored more sensitively.
    if hr_baseline is not None:
        if hr_baseline <= 55.0 or hr_baseline >= 95.0:
            out["hr_relative_change_pct"] = 15.0
        elif hr_baseline >= 80.0:
            out["hr_relative_change_pct"] = 18.0

    # Patient-personalized SpO2 sensitivity:
    # high baseline oxygenation should trigger attention with smaller relative drops.
    if spo2_baseline is not None:
        if spo2_baseline >= 98.0:
            out["spo2_drop_from_baseline_pct"] = 3.0
            out["spo2_attention_pct"] = 95.0
        elif spo2_baseline >= 96.0:
            out["spo2_drop_from_baseline_pct"] = 3.5
            out["spo2_attention_pct"] = 94.5
        elif spo2_baseline <= 95.0:
            out["spo2_drop_from_baseline_pct"] = 2.0
            out["spo2_attention_pct"] = 94.0

    rso2_baselines = [v for v in [rso2_l_baseline, rso2_r_baseline] if v is not None]
    if rso2_baselines:
        base_min = min(rso2_baselines)
        if base_min < 60.0:
            out["rso2_drop_from_baseline_pct"] = 15.0
        elif base_min >= 70.0:
            out["rso2_drop_from_baseline_pct"] = 20.0

    return out


def build_clinical_assessment(
    df_window: pd.DataFrame,
    anchor: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    hr_s = _safe_get_series(df_window, "HR")
    mbp_s = _safe_get_series(df_window, "MBP")
    sbp_s = _safe_get_series(df_window, "SBP")
    dbp_s = _safe_get_series(df_window, "DBP")
    spo2_s = _safe_get_series(df_window, "SPO2")
    bis_s = _safe_get_series(df_window, "BIS")
    etco2_s = _safe_get_series(df_window, "ETCO2")
    svv_s = _safe_get_series(df_window, "SVV")
    ppv_s = _safe_get_series(df_window, "PPV")
    cvp_s = _safe_get_series(df_window, "CVP")
    co_s = _safe_get_series(df_window, "CO")
    ci_s = _safe_get_series(df_window, "CI")
    sv_s = _safe_get_series(df_window, "SV")
    svr_s = _safe_get_series(df_window, "SVR")
    bt_s = _safe_get_series(df_window, "BT")
    rso2_l_s = _safe_get_series(df_window, "RSO2_L")
    rso2_r_s = _safe_get_series(df_window, "RSO2_R")

    hr_last = _last_window_mean(hr_s, vital_key="HR") if hr_s is not None else None
    mbp_last = _last_window_mean(mbp_s, vital_key="MBP") if mbp_s is not None else None
    sbp_last = _last_window_mean(sbp_s, vital_key="SBP") if sbp_s is not None else None
    dbp_last = _last_window_mean(dbp_s, vital_key="DBP") if dbp_s is not None else None
    spo2_last = _last_window_mean(spo2_s, vital_key="SPO2") if spo2_s is not None else None
    bis_last = _last_window_mean(bis_s, vital_key="BIS") if bis_s is not None else None
    etco2_last = _last_window_mean(etco2_s, vital_key="ETCO2") if etco2_s is not None else None
    svv_last = _last_window_mean(svv_s, vital_key="SVV") if svv_s is not None else None
    ppv_last = _last_window_mean(ppv_s, vital_key="PPV") if ppv_s is not None else None
    cvp_last = _last_window_mean(cvp_s, vital_key="CVP") if cvp_s is not None else None
    co_last = _last_window_mean(co_s, vital_key="CO") if co_s is not None else None
    ci_last = _last_window_mean(ci_s, vital_key="CI") if ci_s is not None else None
    sv_last = _last_window_mean(sv_s, vital_key="SV") if sv_s is not None else None
    svr_last = _last_window_mean(svr_s, vital_key="SVR") if svr_s is not None else None
    bt_last = _last_window_mean(bt_s, vital_key="BT") if bt_s is not None else None
    rso2_l_last = _last_window_mean(rso2_l_s, vital_key="RSO2_L") if rso2_l_s is not None else None
    rso2_r_last = _last_window_mean(rso2_r_s, vital_key="RSO2_R") if rso2_r_s is not None else None
    personalized = _build_personalized_thresholds(baseline_comparison)
    hr_relative_limit = float(personalized["hr_relative_change_pct"])
    spo2_attention_limit = float(personalized["spo2_attention_pct"])
    spo2_drop_limit = float(personalized["spo2_drop_from_baseline_pct"])
    spo2_attention_persist_limit = float(personalized["spo2_attention_persist_sec"])
    map_relative_drop_limit = float(personalized["map_relative_drop_pct"])
    rso2_drop_limit = float(personalized["rso2_drop_from_baseline_pct"])

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
    sbp_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SBP",
        predicate=lambda s: s < ANES_THRESHOLDS["sbp_low_mmhg"],
    )
    sbp_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SBP",
        predicate=lambda s: s > ANES_THRESHOLDS["sbp_high_mmhg"],
    )
    dbp_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="DBP",
        predicate=lambda s: s < ANES_THRESHOLDS["dbp_low_mmhg"],
    )
    dbp_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="DBP",
        predicate=lambda s: s > ANES_THRESHOLDS["dbp_high_mmhg"],
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
    spo2_attention_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SPO2",
        predicate=lambda s: s <= spo2_attention_limit,
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
    etco2_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
        predicate=lambda s: s < ANES_THRESHOLDS["etco2_low_mmhg"],
    )
    etco2_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
        predicate=lambda s: s > ANES_THRESHOLDS["etco2_high_mmhg"],
    )
    etco2_severe_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
        predicate=lambda s: s < ANES_THRESHOLDS["etco2_severe_low_mmhg"],
    )
    etco2_severe_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
        predicate=lambda s: s > ANES_THRESHOLDS["etco2_severe_high_mmhg"],
    )
    etco2_missing_persist_sec = _tail_missing_duration_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
    )
    etco2_zero_like_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="ETCO2",
        predicate=lambda s: s <= ANES_THRESHOLDS["etco2_zeroing_value_mmhg"],
    )
    bt_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="BT",
        predicate=lambda s: s < ANES_THRESHOLDS["bt_low_c"],
    )
    bt_fever_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="BT",
        predicate=lambda s: s > ANES_THRESHOLDS["bt_fever_c"],
    )
    bt_high_fever_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="BT",
        predicate=lambda s: s >= ANES_THRESHOLDS["bt_high_fever_c"],
    )
    rso2_l_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="RSO2_L",
        predicate=lambda s: s < ANES_THRESHOLDS["rso2_low_abs_pct"],
    )
    rso2_r_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="RSO2_R",
        predicate=lambda s: s < ANES_THRESHOLDS["rso2_low_abs_pct"],
    )
    co_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="CO",
        predicate=lambda s: s < ANES_THRESHOLDS["co_low_l_min"],
    )
    co_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="CO",
        predicate=lambda s: s > ANES_THRESHOLDS["co_high_l_min"],
    )
    ci_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="CI",
        predicate=lambda s: s < ANES_THRESHOLDS["ci_low_l_min_m2"],
    )
    ci_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="CI",
        predicate=lambda s: s > ANES_THRESHOLDS["ci_high_l_min_m2"],
    )
    sv_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SV",
        predicate=lambda s: s < ANES_THRESHOLDS["sv_low_ml"],
    )
    sv_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SV",
        predicate=lambda s: s > ANES_THRESHOLDS["sv_high_ml"],
    )
    ppv_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="PPV",
        predicate=lambda s: s >= ANES_THRESHOLDS["ppv_high_pct"],
    )
    ppv_severe_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="PPV",
        predicate=lambda s: s >= ANES_THRESHOLDS["ppv_severe_high_pct"],
    )
    svr_low_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SVR",
        predicate=lambda s: s < ANES_THRESHOLDS["svr_low_dyns_cm5"],
    )
    svr_high_persist_sec = _tail_persistence_by_vital(
        df_window=df_window,
        vital_key="SVR",
        predicate=lambda s: s > ANES_THRESHOLDS["svr_high_dyns_cm5"],
    )

    flags: List[str] = []
    contextual_interpretation: List[str] = []
    map_absolute_triggered = False
    map_relative_triggered = False
    sbp_relative_triggered = False
    dbp_relative_triggered = False
    hr_relative_triggered = False
    spo2_relative_triggered = False
    rso2_relative_triggered = False
    critical_alarm = False

    if mbp_last is not None:
        if map_severe_persist_sec >= decision_windows["critical_window_sec"]:
            flags.append(
                f"重度低血压（MAP < 55 mmHg，持续约{map_severe_persist_sec:.0f}s，>= {decision_windows['critical_window_sec']:.0f}s）"
            )
            map_absolute_triggered = True
            critical_alarm = True
        elif map_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"低血压（MAP < 65 mmHg，持续约{map_low_persist_sec:.0f}s，>= {decision_windows['hemodynamic_window_sec']:.0f}s）"
            )
            map_absolute_triggered = True
        elif mbp_last < ANES_THRESHOLDS["map_hypotension_mmhg"]:
            flags.append(
                f"MAP低于65但持续不足{decision_windows['hemodynamic_window_sec']:.0f}s（早期预警）"
            )

    if sbp_last is not None:
        if sbp_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"SBP偏低（< {ANES_THRESHOLDS['sbp_low_mmhg']:.0f} mmHg，持续约{sbp_low_persist_sec:.0f}s）"
            )
        elif sbp_high_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"SBP偏高（> {ANES_THRESHOLDS['sbp_high_mmhg']:.0f} mmHg，持续约{sbp_high_persist_sec:.0f}s）"
            )
        elif sbp_last < ANES_THRESHOLDS["sbp_low_mmhg"] or sbp_last > ANES_THRESHOLDS["sbp_high_mmhg"]:
            flags.append(
                f"SBP超出常规范围（{sbp_last:.1f} mmHg）"
            )

    if dbp_last is not None:
        if dbp_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"DBP偏低（< {ANES_THRESHOLDS['dbp_low_mmhg']:.0f} mmHg，持续约{dbp_low_persist_sec:.0f}s）"
            )
        elif dbp_high_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"DBP偏高（> {ANES_THRESHOLDS['dbp_high_mmhg']:.0f} mmHg，持续约{dbp_high_persist_sec:.0f}s）"
            )
        elif dbp_last < ANES_THRESHOLDS["dbp_low_mmhg"] or dbp_last > ANES_THRESHOLDS["dbp_high_mmhg"]:
            flags.append(
                f"DBP超出常规范围（{dbp_last:.1f} mmHg）"
            )

    if hr_last is not None:
        if hr_tachy_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"心动过速（HR > 100 bpm，持续约{hr_tachy_persist_sec:.0f}s）"
            )
            if hr_last >= 120.0:
                critical_alarm = True
        elif hr_brady_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"心动过缓（HR < 50 bpm，持续约{hr_brady_persist_sec:.0f}s）"
            )
            if hr_last <= 40.0:
                critical_alarm = True

    if spo2_last is not None:
        if spo2_last < ANES_THRESHOLDS["spo2_severe_low_pct"]:
            flags.append(
                f"重度低氧血症（SpO2 < 90%，当前约{spo2_last:.1f}%）"
            )
            critical_alarm = True
        elif spo2_severe_persist_sec >= decision_windows["critical_window_sec"]:
            flags.append(
                f"重度低氧血症（SpO2 < 90%，持续约{spo2_severe_persist_sec:.0f}s）"
            )
            critical_alarm = True
        elif spo2_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"血氧下降（SpO2 < 94%，持续约{spo2_low_persist_sec:.0f}s）"
            )
        elif spo2_attention_persist_sec >= spo2_attention_persist_limit:
            flags.append(
                f"SpO2高敏感持续预警（≤{spo2_attention_limit:.1f}%，持续约{spo2_attention_persist_sec:.0f}s）"
            )
        elif spo2_last <= spo2_attention_limit:
            flags.append(
                f"SpO2处于高敏感预警区（≤{spo2_attention_limit:.1f}%），需严密观察呼吸道与通气状态"
            )

    etco2_missing_alert_sec = float(ANES_THRESHOLDS["etco2_missing_alert_sec"])
    etco2_zeroing_hint_sec = float(ANES_THRESHOLDS["etco2_zeroing_hint_sec"])
    etco2_zeroing_suspected = (
        etco2_missing_persist_sec >= etco2_missing_alert_sec
        and etco2_zero_like_persist_sec >= etco2_zeroing_hint_sec
    )
    if etco2_missing_persist_sec >= etco2_missing_alert_sec:
        if etco2_zeroing_suspected:
            flags.append(
                f"EtCO2信号缺失（约{etco2_missing_persist_sec:.0f}s），前序出现近零值，疑似监测校零/复位"
            )
        else:
            flags.append(
                f"EtCO2信号持续缺失（约{etco2_missing_persist_sec:.0f}s，除校零外需立即排查气道/回路）"
            )
            critical_alarm = True
    elif etco2_severe_low_persist_sec >= decision_windows["critical_window_sec"]:
        flags.append(
            f"EtCO2重度异常偏低（<{ANES_THRESHOLDS['etco2_severe_low_mmhg']:.0f} mmHg，持续约{etco2_severe_low_persist_sec:.0f}s）"
        )
        critical_alarm = True
    elif etco2_severe_high_persist_sec >= decision_windows["critical_window_sec"]:
        flags.append(
            f"EtCO2重度异常偏高（>{ANES_THRESHOLDS['etco2_severe_high_mmhg']:.0f} mmHg，持续约{etco2_severe_high_persist_sec:.0f}s）"
        )
        critical_alarm = True
    elif etco2_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"EtCO2偏低（<{ANES_THRESHOLDS['etco2_low_mmhg']:.0f} mmHg，持续约{etco2_low_persist_sec:.0f}s）"
        )
    elif etco2_high_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"EtCO2偏高（>{ANES_THRESHOLDS['etco2_high_mmhg']:.0f} mmHg，持续约{etco2_high_persist_sec:.0f}s）"
        )

    if bt_last is not None:
        if bt_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"低体温（BT < {ANES_THRESHOLDS['bt_low_c']:.1f}℃，持续约{bt_low_persist_sec:.0f}s）"
            )
        elif bt_high_fever_persist_sec >= decision_windows["critical_window_sec"]:
            flags.append(
                f"高热（BT ≥ {ANES_THRESHOLDS['bt_high_fever_c']:.1f}℃，持续约{bt_high_fever_persist_sec:.0f}s）"
            )
            critical_alarm = True
        elif bt_fever_persist_sec >= decision_windows["hemodynamic_window_sec"]:
            flags.append(
                f"发热（BT > {ANES_THRESHOLDS['bt_fever_c']:.1f}℃，持续约{bt_fever_persist_sec:.0f}s）"
            )

    rso2_vals = [v for v in [rso2_l_last, rso2_r_last] if v is not None]
    rso2_min_last = min(rso2_vals) if rso2_vals else None
    if rso2_min_last is not None:
        if rso2_min_last < ANES_THRESHOLDS["rso2_low_abs_pct"]:
            flags.append(
                f"脑氧饱和度异常（rSO2最小约{rso2_min_last:.1f}% < {ANES_THRESHOLDS['rso2_low_abs_pct']:.0f}%）"
            )
        elif rso2_min_last < ANES_THRESHOLDS["rso2_warn_abs_pct"]:
            flags.append(
                f"脑氧饱和度偏低预警（rSO2最小约{rso2_min_last:.1f}%）"
            )
    if (
        rso2_l_low_persist_sec >= decision_windows["hemodynamic_window_sec"]
        or rso2_r_low_persist_sec >= decision_windows["hemodynamic_window_sec"]
    ):
        flags.append(
            f"rSO2持续低值（L约{rso2_l_low_persist_sec:.0f}s, R约{rso2_r_low_persist_sec:.0f}s）"
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
    sbp_change_pct = None
    dbp_change_pct = None
    hr_change_pct = None
    spo2_drop_pct = None
    co_change_pct = None
    ci_change_pct = None
    sv_change_pct = None
    ppv_change_pct = None
    svr_change_pct = None
    rso2_l_drop_pct = None
    rso2_r_drop_pct = None
    if baseline_comparison:
        map_drop_pct = baseline_comparison.get("MAP_drop_from_baseline_pct")
        sbp_change_pct = baseline_comparison.get("SBP_change_from_baseline_pct")
        dbp_change_pct = baseline_comparison.get("DBP_change_from_baseline_pct")
        hr_change_pct = baseline_comparison.get("HR_change_from_baseline_pct")
        spo2_drop_pct = baseline_comparison.get("SpO2_drop_from_baseline_pct")
        co_change_pct = baseline_comparison.get("CO_change_from_baseline_pct")
        ci_change_pct = baseline_comparison.get("CI_change_from_baseline_pct")
        sv_change_pct = baseline_comparison.get("SV_change_from_baseline_pct")
        ppv_change_pct = baseline_comparison.get("PPV_change_from_baseline_pct")
        svr_change_pct = baseline_comparison.get("SVR_change_from_baseline_pct")
        rso2_l_drop_pct = baseline_comparison.get("rSO2_L_drop_from_baseline_pct")
        rso2_r_drop_pct = baseline_comparison.get("rSO2_R_drop_from_baseline_pct")
        if map_drop_pct is not None and float(map_drop_pct) >= map_relative_drop_limit:
            map_relative_triggered = True
            flags.append(f"MAP较基线下降明显（{float(map_drop_pct):.1f}%）")
        if sbp_change_pct is not None and abs(float(sbp_change_pct)) >= ANES_THRESHOLDS["sbp_relative_change_pct"]:
            sbp_relative_triggered = True
            trend_txt = "升高" if float(sbp_change_pct) > 0 else "下降"
            flags.append(f"SBP较基线{trend_txt}明显（{abs(float(sbp_change_pct)):.1f}%）")
        if dbp_change_pct is not None and abs(float(dbp_change_pct)) >= ANES_THRESHOLDS["dbp_relative_change_pct"]:
            dbp_relative_triggered = True
            trend_txt = "升高" if float(dbp_change_pct) > 0 else "下降"
            flags.append(f"DBP较基线{trend_txt}明显（{abs(float(dbp_change_pct)):.1f}%）")
        if hr_change_pct is not None and abs(float(hr_change_pct)) >= hr_relative_limit:
            hr_relative_triggered = True
            trend_txt = "升高" if float(hr_change_pct) > 0 else "下降"
            flags.append(f"HR较基线{trend_txt}明显（{float(hr_change_pct):.1f}%）")
        if (
            spo2_drop_pct is not None
            and float(spo2_drop_pct) >= spo2_drop_limit
            and spo2_last is not None
            and spo2_last <= spo2_attention_limit
        ):
            spo2_relative_triggered = True
            flags.append(f"SpO2较基线下降明显（{float(spo2_drop_pct):.1f}%）")
        if co_change_pct is not None and abs(float(co_change_pct)) >= 20.0:
            trend_txt = "升高" if float(co_change_pct) > 0 else "下降"
            flags.append(f"CO较基线{trend_txt}明显（{abs(float(co_change_pct)):.1f}%）")
        if ci_change_pct is not None and abs(float(ci_change_pct)) >= 20.0:
            trend_txt = "升高" if float(ci_change_pct) > 0 else "下降"
            flags.append(f"CI较基线{trend_txt}明显（{abs(float(ci_change_pct)):.1f}%）")
        if sv_change_pct is not None and abs(float(sv_change_pct)) >= 20.0:
            trend_txt = "升高" if float(sv_change_pct) > 0 else "下降"
            flags.append(f"SV较基线{trend_txt}明显（{abs(float(sv_change_pct)):.1f}%）")
        if ppv_change_pct is not None and abs(float(ppv_change_pct)) >= 20.0:
            trend_txt = "升高" if float(ppv_change_pct) > 0 else "下降"
            flags.append(f"PPV较基线{trend_txt}明显（{abs(float(ppv_change_pct)):.1f}%）")
        if svr_change_pct is not None and abs(float(svr_change_pct)) >= 20.0:
            trend_txt = "升高" if float(svr_change_pct) > 0 else "下降"
            flags.append(f"SVR较基线{trend_txt}明显（{abs(float(svr_change_pct)):.1f}%）")
        rso2_drops = []
        if rso2_l_drop_pct is not None:
            rso2_drops.append(float(rso2_l_drop_pct))
        if rso2_r_drop_pct is not None:
            rso2_drops.append(float(rso2_r_drop_pct))
        if rso2_drops and max(rso2_drops) >= rso2_drop_limit:
            rso2_relative_triggered = True
            flags.append(f"rSO2较基线下降明显（最大约{max(rso2_drops):.1f}%）")

    if map_absolute_triggered:
        flags.append("MAP绝对阈值触发：作为器官灌注底线优先处理")
    if map_relative_triggered:
        flags.append("MAP相对下降触发：用于个体化风险分层")
    if sbp_relative_triggered:
        flags.append("SBP相对变化触发：幅度达到临床关注阈值（≥30%）")
    if dbp_relative_triggered:
        flags.append("DBP相对变化触发：幅度达到临床关注阈值（≥30%）")
    if hr_relative_triggered:
        flags.append(f"HR相对变化触发：变化幅度达到个体化关注阈值（≥{hr_relative_limit:.0f}%）")
    if spo2_relative_triggered:
        flags.append("SpO2高敏相对下降触发：即使绝对值未<90也需提前干预")
    if rso2_relative_triggered:
        flags.append("rSO2相对下降触发：建议优先保障脑灌注与氧供")
    if svv_last is not None and svv_last >= ANES_THRESHOLDS["svv_severe_high_pct"]:
        flags.append(f"SVV明显升高（{svv_last:.1f}%）提示低容量/容量反应性强")
    elif svv_last is not None and svv_last >= ANES_THRESHOLDS["svv_high_pct"]:
        flags.append(f"SVV偏高（{svv_last:.1f}%），建议复核容量状态")
    if ppv_last is not None and ppv_last >= ANES_THRESHOLDS["ppv_severe_high_pct"]:
        flags.append(f"PPV明显升高（{ppv_last:.1f}%）提示容量反应性强")
    elif ppv_last is not None and ppv_last >= ANES_THRESHOLDS["ppv_high_pct"]:
        flags.append(f"PPV偏高（{ppv_last:.1f}%），建议优先评估容量状态")
    if cvp_last is not None and cvp_last <= ANES_THRESHOLDS["cvp_low_mmhg"]:
        flags.append(f"CVP偏低（{cvp_last:.1f} mmHg），需结合容量与回心血量评估")
    elif cvp_last is not None and cvp_last >= ANES_THRESHOLDS["cvp_high_mmhg"]:
        flags.append(f"CVP偏高（{cvp_last:.1f} mmHg），需评估右心负荷与液体管理")
    if co_last is not None and co_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"CO偏低（<{ANES_THRESHOLDS['co_low_l_min']:.1f} L/min，持续约{co_low_persist_sec:.0f}s）"
        )
    elif co_last is not None and co_last < ANES_THRESHOLDS["co_low_l_min"]:
        flags.append(f"CO短时偏低（当前约{co_last:.1f} L/min）")
    elif co_last is not None and co_high_persist_sec >= decision_windows["slow_trend_window_sec"]:
        flags.append(
            f"CO持续偏高（>{ANES_THRESHOLDS['co_high_l_min']:.1f} L/min，持续约{co_high_persist_sec:.0f}s）"
        )
    if ci_last is not None and ci_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"CI偏低（<{ANES_THRESHOLDS['ci_low_l_min_m2']:.1f} L/(min·m²)，持续约{ci_low_persist_sec:.0f}s）"
        )
        if ci_last < 2.2:
            critical_alarm = True
    elif ci_last is not None and ci_last < ANES_THRESHOLDS["ci_low_l_min_m2"]:
        flags.append(f"CI短时偏低（当前约{ci_last:.1f} L/(min·m²)）")
    elif ci_last is not None and ci_high_persist_sec >= decision_windows["slow_trend_window_sec"]:
        flags.append(
            f"CI持续偏高（>{ANES_THRESHOLDS['ci_high_l_min_m2']:.1f} L/(min·m²)，持续约{ci_high_persist_sec:.0f}s）"
        )
    if sv_last is not None and sv_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"SV偏低（<{ANES_THRESHOLDS['sv_low_ml']:.0f} mL，持续约{sv_low_persist_sec:.0f}s）"
        )
    elif sv_last is not None and sv_last < ANES_THRESHOLDS["sv_low_ml"]:
        flags.append(f"SV短时偏低（当前约{sv_last:.0f} mL）")
    elif sv_last is not None and sv_high_persist_sec >= decision_windows["slow_trend_window_sec"]:
        flags.append(
            f"SV持续偏高（>{ANES_THRESHOLDS['sv_high_ml']:.0f} mL，持续约{sv_high_persist_sec:.0f}s）"
        )
    if svr_last is not None and svr_low_persist_sec >= decision_windows["hemodynamic_window_sec"]:
        flags.append(
            f"SVR偏低（<{ANES_THRESHOLDS['svr_low_dyns_cm5']:.0f} dyn·s·cm⁻5，持续约{svr_low_persist_sec:.0f}s）"
        )
    elif svr_last is not None and svr_last < ANES_THRESHOLDS["svr_low_dyns_cm5"]:
        flags.append(f"SVR短时偏低（当前约{svr_last:.0f} dyn·s·cm⁻5）")
    elif svr_last is not None and svr_high_persist_sec >= decision_windows["slow_trend_window_sec"]:
        flags.append(
            f"SVR持续偏高（>{ANES_THRESHOLDS['svr_high_dyns_cm5']:.0f} dyn·s·cm⁻5，持续约{svr_high_persist_sec:.0f}s）"
        )

    med_key = str(anchor.get("medication_key", ""))
    anchor_source = str(anchor.get("anchor_source", "medication"))
    delta = _safe_float(anchor.get("delta"))
    intervention_hint = "未触发特定干预启发式。"
    if med_key in {"PHE_RATE", "PHE_VOL"}:
        intervention_hint = "去氧肾上腺素通常用于血管扩张相关低血压且心率不低的场景；若明显心动过缓需谨慎。"
    elif med_key in {"EPH_VOL", "EPH_RATE"}:
        intervention_hint = "麻黄碱更适合低血压合并低心率；若已明显心动过速应避免继续加量。"
    elif med_key in {"NOR_RATE", "NOR_VOL"}:
        intervention_hint = "去甲肾上腺素用于难治性血管扩张性低血压；疑似低容量时应先扩容后升压。"
    elif med_key in {"EPI_RATE", "EPI_VOL"}:
        intervention_hint = "肾上腺素主要用于抢救级循环衰竭场景；不宜作为常规轻中度低血压首选。"
    elif med_key in {"NTG_VOL", "NTG_RATE"}:
        intervention_hint = "硝酸甘油用于缺血/高血压/肺水肿场景；MAP偏低时可显著恶化循环，需谨慎。"
    elif med_key in {"MIL_VOL", "MIL_RATE"}:
        intervention_hint = "米力农可改善低心排与肺高压，但有扩血管效应；低血压未纠正前避免升级。"
    elif med_key in {"ATRO_VOL", "ATRO_RATE"}:
        intervention_hint = "阿托品用于有血流动力学意义的严重心动过缓；非症状性慢心率不应机械使用。"
    elif med_key in {"PPF20_VOL", "PPF20_RATE"}:
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
    if spo2_last is not None and spo2_last < ANES_THRESHOLDS["spo2_low_pct"]:
        contextual_interpretation.append("SpO2下降提示需先保证气道和氧合，再讨论麻醉深度微调。")
    if etco2_missing_persist_sec >= etco2_missing_alert_sec and not etco2_zeroing_suspected:
        contextual_interpretation.append("EtCO2连续缺失高度提示呼吸回路/气道问题，应立即人工核查。")
    elif etco2_missing_persist_sec >= etco2_missing_alert_sec and etco2_zeroing_suspected:
        contextual_interpretation.append("EtCO2缺失前出现近零值，可能处于校零/复位过程，需快速复核监护状态。")
    elif etco2_last is not None and etco2_last < ANES_THRESHOLDS["etco2_low_mmhg"]:
        contextual_interpretation.append("EtCO2偏低需排查过度通气、低灌注或回路泄漏。")
    elif etco2_last is not None and etco2_last > ANES_THRESHOLDS["etco2_high_mmhg"]:
        contextual_interpretation.append("EtCO2偏高需警惕通气不足、二氧化碳潴留或气道阻力升高。")
    if bt_last is not None and bt_last < ANES_THRESHOLDS["bt_low_c"]:
        contextual_interpretation.append("体温低于36℃时需主动保温，避免凝血障碍和苏醒延迟。")
    elif bt_last is not None and bt_last >= ANES_THRESHOLDS["bt_high_fever_c"]:
        contextual_interpretation.append("术中高热需快速排查感染、输血反应或恶性高热相关风险。")
    if rso2_vals and min(rso2_vals) < ANES_THRESHOLDS["rso2_low_abs_pct"]:
        contextual_interpretation.append("rSO2异常提示脑灌注/氧供不足风险，应优先优化灌注和氧合。")
    if ci_last is not None and ci_last < ANES_THRESHOLDS["ci_low_l_min_m2"]:
        contextual_interpretation.append("CI偏低提示心排量不足，需结合容量状态、心肌抑制与后负荷综合判断。")
    if ppv_last is not None and ppv_last >= ANES_THRESHOLDS["ppv_high_pct"]:
        contextual_interpretation.append("PPV升高提示可能容量反应性，建议优先评估补液反应。")
    if svr_last is not None and svr_last < ANES_THRESHOLDS["svr_low_dyns_cm5"]:
        contextual_interpretation.append("SVR偏低提示外周血管扩张，需评估麻醉/炎症相关血流动力学改变。")
    if not flags:
        contextual_interpretation.append("当前样本属于维持期平稳体征，重点是监测与避免过度干预。")

    severity = "low"
    if critical_alarm or any(("重度" in f) or ("严重" in f) for f in flags):
        severity = "high"
    elif flags:
        severity = "moderate"
    sample_category = "stable_maintenance"
    if severity == "high":
        sample_category = "critical_alarm"
    elif severity == "moderate":
        sample_category = "warning_signal"

    return {
        "recent_state_mean": {
            "MAP_mmhg": mbp_last,
            "SBP_mmhg": sbp_last,
            "DBP_mmhg": dbp_last,
            "HR_bpm": hr_last,
            "SpO2_pct": spo2_last,
            "BIS": bis_last,
            "EtCO2_mmhg": etco2_last,
            "CO_L_min": co_last,
            "CI_L_min_m2": ci_last,
            "SV_ml": sv_last,
            "SVV_pct": svv_last,
            "PPV_pct": ppv_last,
            "CVP_mmhg": cvp_last,
            "SVR_dyns_cm5": svr_last,
            "BT_c": bt_last,
            "rSO2_L_pct": rso2_l_last,
            "rSO2_R_pct": rso2_r_last,
        },
        "baseline_comparison": baseline_comparison if baseline_comparison is not None else {},
        "risk_flags": flags,
        "contextual_interpretation": contextual_interpretation,
        "risk_level": severity,
        "sample_category": sample_category,
        "map_policy": {
            "absolute_primary": True,
            "relative_layered": True,
            "absolute_triggered": map_absolute_triggered,
            "relative_triggered": map_relative_triggered,
        },
        "sensitivity_policy": {
            "hr_relative_triggered": hr_relative_triggered,
            "spo2_relative_triggered": spo2_relative_triggered,
            "rso2_relative_triggered": rso2_relative_triggered,
            "etco2_missing_triggered": etco2_missing_persist_sec >= etco2_missing_alert_sec,
            "etco2_zeroing_suspected": etco2_zeroing_suspected,
            "personalized_thresholds": personalized,
            "bis_as_supportive_only": True,
        },
        "persistence_seconds": {
            "map_lt_55": map_severe_persist_sec,
            "map_lt_65": map_low_persist_sec,
            "sbp_lt_90": sbp_low_persist_sec,
            "sbp_gt_180": sbp_high_persist_sec,
            "dbp_lt_60": dbp_low_persist_sec,
            "dbp_gt_100": dbp_high_persist_sec,
            "hr_gt_100": hr_tachy_persist_sec,
            "hr_lt_50": hr_brady_persist_sec,
            "spo2_lt_90": spo2_severe_persist_sec,
            "spo2_lt_94": spo2_low_persist_sec,
            "spo2_le_attention": spo2_attention_persist_sec,
            "bis_gt_60": bis_high_persist_sec,
            "bis_lt_40": bis_low_persist_sec,
            "etco2_missing": etco2_missing_persist_sec,
            "etco2_zero_like": etco2_zero_like_persist_sec,
            "etco2_lt_25": etco2_severe_low_persist_sec,
            "etco2_lt_30": etco2_low_persist_sec,
            "etco2_gt_50": etco2_high_persist_sec,
            "etco2_gt_60": etco2_severe_high_persist_sec,
            "co_lt_low": co_low_persist_sec,
            "co_gt_high": co_high_persist_sec,
            "ci_lt_low": ci_low_persist_sec,
            "ci_gt_high": ci_high_persist_sec,
            "sv_lt_low": sv_low_persist_sec,
            "sv_gt_high": sv_high_persist_sec,
            "ppv_ge_13": ppv_high_persist_sec,
            "ppv_ge_18": ppv_severe_high_persist_sec,
            "svr_lt_low": svr_low_persist_sec,
            "svr_gt_high": svr_high_persist_sec,
            "bt_lt_36": bt_low_persist_sec,
            "bt_gt_37_5": bt_fever_persist_sec,
            "bt_ge_38": bt_high_fever_persist_sec,
            "rso2_l_lt_55": rso2_l_low_persist_sec,
            "rso2_r_lt_55": rso2_r_low_persist_sec,
        },
        "decision_windows_sec": decision_windows,
        "intervention_consideration": intervention_hint,
        "missing_data_guidance": (
            "When a physiologic indicator is unavailable, infer decisions from available objective signals "
            "and avoid mentioning missing indicators in final Q/A text."
        ),
        "drug_reference": DRUG_REFERENCE,
    }


def _row_numeric_by_keys(row: pd.Series, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row.index:
            val = _safe_float(row.get(key))
            if val is not None:
                return float(val)
    return None


def _is_malignant_arrhythmia_text(text: str) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    malignant_patterns = [
        r"\bvf\b",
        r"\bvt\b",
        r"\bvta\b",
        r"ventricular fibrillation",
        r"ventricular tachy",
        r"\bnsvt\b",
        r"torsade",
        r"torsades",
        r"asystole",
        r"pulseless",
        r"af with rvr",
        r"afib with rvr",
        r"atrial fibrillation with rapid",
        r"atrial fibrillation.*rapid ventricular",
        r"\bsvta\b",
        r"\bsvt\b",
        r"\bpsvt\b",
        r"supraventricular tachy",
        r"室颤",
        r"室性心动过速",
        r"室速",
        r"房颤.*快心室率",
        r"阵发性室上速",
    ]
    for pat in malignant_patterns:
        if re.search(pat, low):
            return True
    return False


def _is_arrhythmia_event_text(text: str) -> bool:
    low = str(text or "").lower().strip()
    if not low:
        return False
    if low in ARRDB_NORMAL_LABELS:
        return False
    rhythm_tokens = [t.strip() for t in re.split(r"[,;/]+", low) if t.strip()]
    meaningful = []
    for tok in rhythm_tokens:
        if tok in ARRDB_NORMAL_LABELS:
            continue
        if tok in {"noise", "artifact"}:
            continue
        meaningful.append(tok)
    if meaningful:
        return True
    return any(k in low for k in ["arrhythmia", "afib", "afl", "svt", "snd", "ventricular", "ectopy"])


def _extract_adverse_events(
    row: pd.Series,
    anchor: Dict[str, Any],
    clinical_assessment: Dict[str, Any],
    patient_weight_kg: Optional[float],
    ane_dur_min: Optional[float],
) -> Dict[str, Any]:
    flags: List[str] = []
    event_types: List[str] = []
    evidence: Dict[str, Any] = {}

    def _add(event_type: str, flag: str, evidence_key: Optional[str] = None, evidence_value: Any = None) -> None:
        if event_type and event_type not in event_types:
            event_types.append(event_type)
        if flag and flag not in flags:
            flags.append(flag)
        if evidence_key:
            evidence[evidence_key] = evidence_value

    ebl = _row_numeric_by_keys(row, ["intraop_ebl", "ebl", "blood_loss_ml"])
    if ebl is not None:
        evidence["ebl_ml"] = float(ebl)
        if ebl >= 1000:
            _add("major_bleeding", f"大出血风险（术中估计失血量约{ebl:.0f} mL）")
        elif ebl >= 500:
            _add("bleeding_warning", f"出血偏多预警（术中估计失血量约{ebl:.0f} mL）")

    uo = _row_numeric_by_keys(row, ["intraop_uo", "uo", "urine_output_ml"])
    if uo is not None:
        evidence["urine_output_ml"] = float(uo)
        if uo <= 5:
            _add("anuria_critical", f"危重少尿/无尿预警（术中尿量约{uo:.1f} mL）")
        elif patient_weight_kg is not None and ane_dur_min is not None and ane_dur_min > 0:
            uo_rate = float(uo) / max(0.1, patient_weight_kg) / max(0.1, ane_dur_min / 60.0)
            evidence["urine_output_ml_per_kg_h"] = float(uo_rate)
            if uo_rate < 0.5:
                _add("oliguria_warning", f"尿量偏低预警（约{uo_rate:.2f} mL/kg/h）")

    potassium = _row_numeric_by_keys(
        row,
        ["abga_k", "lab_k", "potassium", "intraop_k", "k_value", "preop_k"],
    )
    if potassium is not None:
        evidence["potassium"] = float(potassium)
        if potassium >= 5.8:
            _add("hyperkalemia_critical", f"高钾危急值风险（K≈{potassium:.2f}）")
        elif potassium <= 3.0:
            _add("hypokalemia_critical", f"低钾危急值风险（K≈{potassium:.2f}）")

    glucose = _row_numeric_by_keys(
        row,
        ["abga_glucose", "lab_glucose", "glucose", "glu", "intraop_glucose", "preop_gluc"],
    )
    if glucose is not None:
        evidence["glucose"] = float(glucose)
        if glucose >= 250:
            _add("hyperglycemia_severe", f"高血糖明显异常（Glu≈{glucose:.0f}）")
        elif glucose >= 180:
            _add("hyperglycemia_warning", f"高血糖预警（Glu≈{glucose:.0f}）")

    pao2 = _row_numeric_by_keys(
        row,
        ["abga_pao2", "lab_pao2", "pao2", "pao2_mmHg", "intraop_pao2", "preop_pao2"],
    )
    if pao2 is not None:
        evidence["pao2"] = float(pao2)
        if pao2 < 60:
            _add("abg_hypoxemia", f"ABG低氧血症风险（PaO2≈{pao2:.0f} mmHg）")

    paco2 = _row_numeric_by_keys(
        row,
        ["abga_paco2", "lab_paco2", "paco2", "paco2_mmHg", "intraop_paco2", "preop_paco2"],
    )
    if paco2 is not None:
        evidence["paco2"] = float(paco2)
        if paco2 > 60:
            _add("abg_hypercapnia", f"ABG二氧化碳潴留风险（PaCO2≈{paco2:.0f} mmHg）")
        elif paco2 > 50:
            _add("abg_hypercapnia", f"ABG通气不足预警（PaCO2≈{paco2:.0f} mmHg）")

    ph = _row_numeric_by_keys(
        row,
        ["abga_ph", "lab_ph", "ph", "intraop_ph", "preop_ph"],
    )
    lactate = _row_numeric_by_keys(
        row,
        ["abga_lactate", "lab_lactate", "lactate", "lac", "intraop_lactate"],
    )
    be = _row_numeric_by_keys(
        row,
        ["abga_be", "lab_be", "be", "base_excess", "intraop_be", "preop_be"],
    )
    if ph is not None:
        evidence["ph"] = float(ph)
    if lactate is not None:
        evidence["lactate"] = float(lactate)
    if be is not None:
        evidence["be"] = float(be)

    severe_acidosis_pattern = (
        ph is not None
        and ph < 7.25
        and lactate is not None
        and lactate >= 2.5
    )
    if severe_acidosis_pattern:
        _add(
            "abg_metabolic_acidosis_hyperlactatemia",
            f"ABG酸中毒+高乳酸风险（pH≈{ph:.2f}, Lactate≈{lactate:.1f}）",
        )
    else:
        if ph is not None and ph < 7.30:
            _add("abg_metabolic_acidosis_warning", f"ABG酸中毒预警（pH≈{ph:.2f}）")
        if lactate is not None and lactate >= 2.0:
            _add("abg_metabolic_acidosis_warning", f"ABG乳酸升高预警（Lactate≈{lactate:.1f}）")
    if be is not None and be <= -8.0:
        _add("abg_be_negative_large", f"ABG碱剩余显著负值（BE≈{be:.1f}）")

    teg_r = _row_numeric_by_keys(
        row,
        ["teg_r", "teg_r_min", "teg_rtime"],
    )
    teg_ma = _row_numeric_by_keys(
        row,
        ["teg_ma", "teg_ma_mm", "teg_max_amplitude"],
    )
    teg_k = _row_numeric_by_keys(
        row,
        ["teg_k", "teg_k_min", "teg_ktime"],
    )
    teg_ci = _row_numeric_by_keys(
        row,
        ["teg_ci", "teg_coag_index", "teg_coagulation_index"],
    )
    if teg_r is not None:
        evidence["teg_r"] = float(teg_r)
        if teg_r > 10:
            _add("coagulation_low", f"TEG低凝风险（R≈{teg_r:.1f} min）")
        elif teg_r < 5:
            _add("coagulation_high", f"TEG高凝风险（R≈{teg_r:.1f} min）")
    if teg_ma is not None:
        evidence["teg_ma"] = float(teg_ma)
        if teg_ma < 50:
            _add("coagulation_low", f"TEG低凝/血小板功能不足风险（MA≈{teg_ma:.1f} mm）")
        elif teg_ma > 70:
            _add("coagulation_high", f"TEG高凝风险（MA≈{teg_ma:.1f} mm）")
    if teg_k is not None:
        evidence["teg_k"] = float(teg_k)
        if teg_k > 3:
            _add("coagulation_low", f"TEG低凝风险（K≈{teg_k:.1f} min）")
        elif teg_k < 1:
            _add("coagulation_high", f"TEG高凝风险（K≈{teg_k:.1f} min）")
    if teg_ci is not None:
        evidence["teg_ci"] = float(teg_ci)
        if teg_ci < -3:
            _add("coagulation_low", f"TEG低凝风险（CI≈{teg_ci:.1f}）")
        elif teg_ci > 3:
            _add("coagulation_high", f"TEG高凝风险（CI≈{teg_ci:.1f}）")

    act = _row_numeric_by_keys(
        row,
        ["act", "act_sec", "activated_clotting_time"],
    )
    if act is not None:
        evidence["act"] = float(act)
        if act < 80 or act > 600:
            _add("act_abnormal", f"ACT异常（≈{act:.0f} s，需结合心外循环场景判断）")

    arr_label = str(anchor.get("arrhythmia_label", anchor.get("after", ""))).strip()
    rhythm_classes = str(row.get("rhythm_classes", "")).strip() if "rhythm_classes" in row.index else ""
    arr_text = " | ".join([x for x in [arr_label, rhythm_classes] if x]).strip()
    if arr_text:
        evidence["arrhythmia_text"] = arr_text
        if _is_malignant_arrhythmia_text(arr_text):
            _add("malignant_arrhythmia", f"恶性心律失常风险（标注：{arr_text[:120]}）")
        elif _is_arrhythmia_event_text(arr_text):
            _add("arrhythmia_event", f"心律异常事件（标注：{arr_text[:120]}）")

    allergy_text = str(row.get("allergy", "")).strip() if "allergy" in row.index else ""
    if allergy_text:
        _add("allergy_history", "既往过敏史提示（需警惕术中过敏相关反应）", "allergy_text", allergy_text[:200])

    recent = clinical_assessment.get("recent_state_mean", {}) if isinstance(clinical_assessment, dict) else {}
    baseline = clinical_assessment.get("baseline_comparison", {}) if isinstance(clinical_assessment, dict) else {}
    persist = clinical_assessment.get("persistence_seconds", {}) if isinstance(clinical_assessment, dict) else {}
    map_now = _safe_float(recent.get("MAP_mmhg"))
    hr_now = _safe_float(recent.get("HR_bpm"))
    spo2_now = _safe_float(recent.get("SpO2_pct"))
    etco2_now = _safe_float(recent.get("EtCO2_mmhg"))
    co_now = _safe_float(recent.get("CO_L_min"))
    ci_now = _safe_float(recent.get("CI_L_min_m2"))
    sv_now = _safe_float(recent.get("SV_ml"))
    ppv_now = _safe_float(recent.get("PPV_pct"))
    svr_now = _safe_float(recent.get("SVR_dyns_cm5"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))
    map_lt_65 = _safe_float(persist.get("map_lt_65")) or 0.0
    map_lt_55 = _safe_float(persist.get("map_lt_55")) or 0.0
    spo2_lt_90 = _safe_float(persist.get("spo2_lt_90")) or 0.0
    etco2_missing = _safe_float(persist.get("etco2_missing")) or 0.0
    etco2_zero_like = _safe_float(persist.get("etco2_zero_like")) or 0.0
    ci_low_persist = _safe_float(persist.get("ci_lt_low")) or 0.0
    co_low_persist = _safe_float(persist.get("co_lt_low")) or 0.0
    sv_low_persist = _safe_float(persist.get("sv_lt_low")) or 0.0
    ppv_high_persist = _safe_float(persist.get("ppv_ge_13")) or 0.0
    svr_low_persist = _safe_float(persist.get("svr_lt_low")) or 0.0
    etco2_missing_alert_sec = float(ANES_THRESHOLDS["etco2_missing_alert_sec"])
    etco2_zeroing_hint_sec = float(ANES_THRESHOLDS["etco2_zeroing_hint_sec"])
    etco2_zeroing_suspected = (
        etco2_missing >= etco2_missing_alert_sec and etco2_zero_like >= etco2_zeroing_hint_sec
    )
    if co_now is not None:
        evidence["co_l_min"] = float(co_now)
    if ci_now is not None:
        evidence["ci_l_min_m2"] = float(ci_now)
    if sv_now is not None:
        evidence["sv_ml"] = float(sv_now)
    if ppv_now is not None:
        evidence["ppv_pct"] = float(ppv_now)
    if svr_now is not None:
        evidence["svr_dyns_cm5"] = float(svr_now)

    hypotension_core = (
        (map_now is not None and map_now < ANES_THRESHOLDS["map_hypotension_mmhg"])
        or map_lt_65 >= ANES_THRESHOLDS["hemodynamic_window_sec"]
        or (map_drop_pct is not None and map_drop_pct >= 30.0)
    )
    perfusion_markers = 0
    if hr_now is not None and (hr_now > ANES_THRESHOLDS["hr_tachycardia_bpm"] or hr_now < ANES_THRESHOLDS["hr_bradycardia_bpm"]):
        perfusion_markers += 1
    if spo2_now is not None and spo2_now < ANES_THRESHOLDS["spo2_severe_low_pct"]:
        perfusion_markers += 1
    if spo2_lt_90 >= ANES_THRESHOLDS["critical_window_sec"]:
        perfusion_markers += 1
    if etco2_missing >= etco2_missing_alert_sec and not etco2_zeroing_suspected:
        perfusion_markers += 1
    if etco2_now is not None and etco2_now < ANES_THRESHOLDS["etco2_severe_low_mmhg"]:
        perfusion_markers += 1
    if ci_now is not None and ci_now < ANES_THRESHOLDS["ci_low_l_min_m2"]:
        perfusion_markers += 1
    if co_now is not None and co_now < ANES_THRESHOLDS["co_low_l_min"]:
        perfusion_markers += 1
    if sv_now is not None and sv_now < ANES_THRESHOLDS["sv_low_ml"]:
        perfusion_markers += 1
    if ppv_now is not None and ppv_now >= ANES_THRESHOLDS["ppv_high_pct"]:
        perfusion_markers += 1
    if svr_now is not None and svr_now < ANES_THRESHOLDS["svr_low_dyns_cm5"]:
        perfusion_markers += 1
    if ci_low_persist >= ANES_THRESHOLDS["hemodynamic_window_sec"]:
        perfusion_markers += 1
    if co_low_persist >= ANES_THRESHOLDS["hemodynamic_window_sec"]:
        perfusion_markers += 1
    if sv_low_persist >= ANES_THRESHOLDS["hemodynamic_window_sec"]:
        perfusion_markers += 1
    if ppv_high_persist >= ANES_THRESHOLDS["hemodynamic_window_sec"]:
        perfusion_markers += 1
    if svr_low_persist >= ANES_THRESHOLDS["hemodynamic_window_sec"]:
        perfusion_markers += 1
    if "major_bleeding" in event_types or "anuria_critical" in event_types:
        perfusion_markers += 1

    if hypotension_core and perfusion_markers >= 1:
        _add("shock_pattern", "休克/低灌注模式预警（需立即评估失血、容量与血管活性支持）")

    potential_allergy_pattern = (
        hypotension_core
        and (spo2_now is not None and spo2_now <= ANES_THRESHOLDS["spo2_low_pct"])
        and (
            (etco2_missing >= etco2_missing_alert_sec and not etco2_zeroing_suspected)
            or (etco2_now is not None and etco2_now < ANES_THRESHOLDS["etco2_low_mmhg"])
            or (hr_now is not None and hr_now > 110.0)
        )
    )
    if potential_allergy_pattern:
        _add(
            "suspected_anaphylaxis_pattern",
            "疑似过敏相关循环/呼吸模式（需结合皮疹、气道压和给药时序人工确认）",
        )

    severity = "low"
    if any(t in ADVERSE_EVENT_CRITICAL_TYPES for t in event_types):
        severity = "high"
    elif event_types:
        severity = "moderate"

    return {
        "flags": flags,
        "event_types": event_types,
        "risk_level": severity,
        "evidence": evidence,
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
    keys = [
        "dx",
        "opdiag",
        "opdiag1",
        "preop_dx",
        "comorbidity",
        "allergy",
        "lab",
        "department",
        "optype",
        "opname",
        "approach",
        "position",
        "ane_type",
        "preop_htn",
        "preop_dm",
    ]
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

    paired_volume_ml = anchor.get("paired_volume_ml")
    paired_volume_key = str(anchor.get("paired_volume_key", "") or "")
    paired_volume_text = ""
    if paired_volume_ml is not None and paired_volume_key:
        paired_label = MEDICATION_DISPLAY.get(paired_volume_key, paired_volume_key)
        paired_volume_text = f"；对应{paired_label}约 {float(paired_volume_ml):.3f} mL"

    if before is None or after is None:
        return f"{label}：速率变化 {delta:+.3f}{paired_volume_text}"
    return f"{label}：{before:.3f} -> {after:.3f}（变化 {delta:+.3f}）{paired_volume_text}"


def _collect_concurrent_medications_at_anchor(
    df_case: pd.DataFrame,
    anchor: Dict[str, Any],
    max_gap_sec: float = 5.0,
    include_inactive: bool = False,
) -> List[Dict[str, Any]]:
    if "Time" not in df_case.columns:
        return []
    t_now = _safe_float(anchor.get("time_sec"))
    if t_now is None:
        return []

    time_series = pd.to_numeric(df_case["Time"], errors="coerce")
    med_by_base: Dict[str, Dict[str, Any]] = {}
    volatile_rate_keys = {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE"}
    anchor_med_key = str(anchor.get("medication_key", "") or "")
    anchor_base = anchor_med_key.rsplit("_", 1)[0] if anchor_med_key.endswith(("_RATE", "_VOL")) else anchor_med_key

    for med_key, cands in medication_track_candidates().items():
        if not med_key.endswith(("_RATE", "_VOL")):
            continue
        col = resolve_column(df_case, cands)
        if col is None or col not in df_case.columns:
            continue
        s = pd.to_numeric(df_case[col], errors="coerce")
        sub = pd.DataFrame({"t": time_series, "v": s}).dropna()
        if sub.empty:
            continue
        idx = (sub["t"] - float(t_now)).abs().idxmin()
        row = sub.loc[idx]
        gap = abs(float(row["t"]) - float(t_now))
        if gap > float(max_gap_sec):
            continue
        value = _safe_float(row["v"])
        if value is None:
            continue

        base = med_key.rsplit("_", 1)[0]
        item = med_by_base.setdefault(
            base,
            {
                "med_base": base,
                "display_name": MEDICATION_DISPLAY.get(med_key, med_key),
                "rate_key": None,
                "rate_track": None,
                "rate_value": None,
                "rate_unit": None,
                "vol_key": None,
                "vol_track": None,
                "volume_ml": None,
                "is_anchor_base": bool(base == anchor_base),
            },
        )
        if med_key.endswith("_RATE"):
            item["rate_key"] = med_key
            item["rate_track"] = col
            item["rate_value"] = float(value)
            if med_key in volatile_rate_keys:
                item["rate_unit"] = "vol%"
            elif med_key == "MAC_RATE":
                item["rate_unit"] = "MAC"
            else:
                item["rate_unit"] = "mL/h"
            item["display_name"] = MEDICATION_DISPLAY.get(med_key, item["display_name"])
        else:
            item["vol_key"] = med_key
            item["vol_track"] = col
            item["volume_ml"] = float(value)
            if item.get("display_name", "") == med_key:
                item["display_name"] = MEDICATION_DISPLAY.get(med_key, item["display_name"])

    kept: List[Dict[str, Any]] = []
    for _, item in med_by_base.items():
        rate_v = _safe_float(item.get("rate_value"))
        rate_unit = str(item.get("rate_unit") or "")
        vol_v = _safe_float(item.get("volume_ml"))
        active = False
        if rate_v is not None:
            if rate_unit in {"vol%", "MAC"}:
                active = abs(float(rate_v)) > 1e-4
            else:
                active = abs(float(rate_v)) > 1e-6
        if vol_v is not None and vol_v > 0:
            active = True
        if active or include_inactive:
            item["is_active"] = bool(active)
            kept.append(item)

    def _rank(x: Dict[str, Any]) -> Tuple[int, int, float, str]:
        is_anchor = 0 if bool(x.get("is_anchor_base")) else 1
        has_rate = 0 if _safe_float(x.get("rate_value")) is not None else 1
        rate_abs = abs(float(_safe_float(x.get("rate_value")) or 0.0))
        name = str(x.get("display_name") or x.get("med_base") or "")
        return (is_anchor, has_rate, -rate_abs, name)

    kept.sort(key=_rank)
    return kept


def _find_vital_alert_anchors(df: pd.DataFrame, cfg: PipelineConfig) -> List[Dict[str, Any]]:
    if "Time" not in df.columns:
        return []
    t = pd.to_numeric(df["Time"], errors="coerce")
    if t.dropna().empty:
        return []

    # Build a coarse second-level grid and detect sustained critical alerts.
    sec = t.round().astype("Int64")
    work = pd.DataFrame({"sec": sec}).dropna()
    work["sec"] = work["sec"].astype(int)
    if work.empty:
        return []

    def _series_by_key(key: str) -> pd.Series:
        col = resolve_vital_column(df, key)
        if col is None:
            return pd.Series(dtype=float)
        return _physio_filter_series(pd.to_numeric(df[col], errors="coerce"), key=key)

    spo2 = _series_by_key("SPO2")
    etco2 = _series_by_key("ETCO2")
    mbp = _series_by_key("MBP")
    hr = _series_by_key("HR")

    frame = pd.DataFrame({"sec": sec, "SPO2": spo2, "ETCO2": etco2, "MBP": mbp, "HR": hr}).dropna(subset=["sec"])
    frame["sec"] = frame["sec"].astype(int)
    sec_df = frame.groupby("sec", as_index=False).median(numeric_only=True)
    if sec_df.empty:
        return []

    def _collect_runs(mask: pd.Series, min_len: int) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        start = None
        prev = None
        for i, v in enumerate(mask.tolist()):
            cur_sec = int(sec_df.iloc[i]["sec"])
            if bool(v):
                if start is None:
                    start = cur_sec
                prev = cur_sec
            else:
                if start is not None and prev is not None and (prev - start + 1) >= min_len:
                    runs.append((start, prev))
                start = None
                prev = None
        if start is not None and prev is not None and (prev - start + 1) >= min_len:
            runs.append((start, prev))
        return runs

    anchors: List[Dict[str, Any]] = []
    rules = [
        ("hypoxemia", (sec_df["SPO2"] < 90.0), 20),
        ("etco2_signal_loss_or_severe_low", (sec_df["ETCO2"].isna() | (sec_df["ETCO2"] < 25.0)), 12),
        ("map_low_perfusion", (sec_df["MBP"] < 65.0), 30),
        ("hr_extreme", ((sec_df["HR"] > 130.0) | (sec_df["HR"] < 45.0)), 20),
    ]
    for label, mask, min_len in rules:
        runs = _collect_runs(mask.fillna(False), min_len=min_len)
        for s0, s1 in runs:
            t_anchor = float((s0 + s1) / 2.0)
            anchors.append(
                {
                    "time_sec": t_anchor,
                    "medication_key": "UNLABELED_EVENT",
                    "track": f"ALERT/{label}",
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
                    "anchor_source": "physio",
                    "alert_type": label,
                    "alert_start_sec": float(s0),
                    "alert_end_sec": float(s1),
                }
            )
    anchors.sort(key=lambda x: float(x.get("time_sec", 0.0)))
    return anchors


def _build_joint_anchors(
    med_anchors: List[Dict[str, Any]],
    vital_anchors: List[Dict[str, Any]],
    max_gap_sec: float,
    require_med_link: bool = True,
) -> List[Dict[str, Any]]:
    if not vital_anchors:
        return med_anchors
    out: List[Dict[str, Any]] = []
    med_sorted = sorted(med_anchors, key=lambda x: float(x.get("time_sec", 0.0)))
    for va in vital_anchors:
        vt = float(va.get("time_sec", 0.0))
        near = None
        near_gap = None
        for ma in med_sorted:
            g = abs(float(ma.get("time_sec", 0.0)) - vt)
            if g <= max_gap_sec and (near_gap is None or g < near_gap):
                near = ma
                near_gap = g
        if near is not None:
            merged = dict(near)
            merged["anchor_source"] = "joint"
            merged["vital_alert_type"] = va.get("alert_type")
            merged["vital_alert_start_sec"] = va.get("alert_start_sec")
            merged["vital_alert_end_sec"] = va.get("alert_end_sec")
            merged["joint_unknown_without_med"] = False
            out.append(merged)
        else:
            if require_med_link:
                continue
            unk = dict(va)
            unk["anchor_source"] = "joint"
            unk["joint_unknown_without_med"] = True
            out.append(unk)
    out.sort(key=lambda x: float(x.get("time_sec", 0.0)))
    deduped: List[Dict[str, Any]] = []
    last_t = None
    for event in out:
        t = float(event.get("time_sec", 0.0))
        if last_t is None or abs(t - last_t) >= max_gap_sec:
            deduped.append(event)
            last_t = t
    return deduped


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
    height = first_valid(row, ["height", "height_cm"], "Unknown")
    weight = first_valid(row, ["weight", "weight_kg", "wt"], "Unknown")
    bmi = first_valid(row, ["bmi"], "Unknown")
    asa = first_valid(row, ["asa"], "Unknown")
    department = first_valid(row, ["department"], "Unknown")
    opname = first_valid(row, ["opname"], "Unknown surgery")
    optype = first_valid(row, ["optype"], "Unknown")
    approach = first_valid(row, ["approach"], "Unknown")
    position = first_valid(row, ["position"], "Unknown")
    ane_type = first_valid(row, ["ane_type"], "Unknown")
    concurrent_medications_all = _collect_concurrent_medications_at_anchor(
        df_case=df_case,
        anchor=anchor,
        include_inactive=True,
    )
    concurrent_medications_active = [x for x in concurrent_medications_all if bool(x.get("is_active", False))]

    clinical_source_meta = _collect_row_fields(
        row,
        [
            "caseid",
            "subjectid",
            "source_dataset",
            "case_id",
            "analysis_start_time_sec",
            "analysis_end_time_sec",
            "analyzed_duration_sec",
            "total_beats",
            "rhythm_classes",
            "ane_dur",
        ],
    )
    clinical_timeline = _collect_row_fields(
        row,
        [
            "casestart",
            "caseend",
            "anestart",
            "aneend",
            "opstart",
            "opend",
            "adm",
            "dis",
            "icu_days",
            "death_inhosp",
        ],
    )
    preop_metrics = _collect_row_prefixed_fields(row, ["preop_"])
    intraop_summary = _collect_row_prefixed_fields(row, ["intraop_"])
    airway_lines = _collect_row_fields(
        row,
        [
            "cormack",
            "airway",
            "tubesize",
            "dltubesize",
            "lmasize",
            "iv1",
            "iv2",
            "aline1",
            "aline2",
            "cline1",
            "cline2",
        ],
    )
    clinical_row_all_fields = _collect_row_all_valid_fields(row)
    baseline_comparison = build_baseline_comparison(
        df_case=df_case,
        df_window=df_window,
        anchor_time_sec=float(anchor["time_sec"]),
    )
    vitaldb_track_map = _build_vitaldb_track_map(df_window=df_window, row=row)
    clinical_assessment = build_clinical_assessment(
        df_window=df_window,
        anchor=anchor,
        baseline_comparison=baseline_comparison,
    )
    patient_weight_kg = _safe_float(weight)
    ane_dur_min = _safe_float(first_valid(row, ["ane_dur", "anesthesia_duration_min"], None))
    adverse_event_bundle = _extract_adverse_events(
        row=row,
        anchor=anchor,
        clinical_assessment=clinical_assessment,
        patient_weight_kg=patient_weight_kg,
        ane_dur_min=ane_dur_min,
    )
    adverse_event_flags = adverse_event_bundle.get("flags", []) if isinstance(adverse_event_bundle, dict) else []
    adverse_event_types = adverse_event_bundle.get("event_types", []) if isinstance(adverse_event_bundle, dict) else []
    if adverse_event_flags:
        merged_flags = list(clinical_assessment.get("risk_flags", [])) if isinstance(clinical_assessment.get("risk_flags"), list) else []
        merged_flags.extend(adverse_event_flags)
        clinical_assessment["risk_flags"] = list(dict.fromkeys(merged_flags))
        clinical_assessment["adverse_event_flags"] = adverse_event_flags
        clinical_assessment["adverse_event_types"] = adverse_event_types
        clinical_assessment["adverse_event_evidence"] = adverse_event_bundle.get("evidence", {})
        if any(t in ADVERSE_EVENT_CRITICAL_TYPES for t in adverse_event_types):
            clinical_assessment["risk_level"] = "high"
            clinical_assessment["sample_category"] = "critical_alarm"
        elif any(t in ADVERSE_EVENT_WARNING_TYPES for t in adverse_event_types) and str(clinical_assessment.get("risk_level", "")).lower() == "low":
            clinical_assessment["risk_level"] = "moderate"
            clinical_assessment["sample_category"] = "warning_signal"
    else:
        clinical_assessment["adverse_event_flags"] = []
        clinical_assessment["adverse_event_types"] = []
        clinical_assessment["adverse_event_evidence"] = {}
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
            "height_cm": height,
            "weight_kg": weight,
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
        "vitaldb_track_map": vitaldb_track_map,
        "vitaldb_indicator_source_hints": VITALDB_INDICATOR_SOURCE_HINTS,
        "clinical_assessment": clinical_assessment,
        "sample_category": clinical_assessment.get("sample_category", "stable_maintenance"),
        "miller_alignment": miller_alignment,
        "actual_intervention": actual_intervention_text,
        "interpreted_intervention_type": intervention_type,
        "concurrent_medications": concurrent_medications_all,
        "concurrent_medications_active": concurrent_medications_active,
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
            "sustained_pre_median": anchor.get("sustained_pre_median"),
            "sustained_post_median": anchor.get("sustained_post_median"),
            "sustained_delta": anchor.get("sustained_delta"),
            "paired_volume_key": anchor.get("paired_volume_key"),
            "paired_volume_track": anchor.get("paired_volume_track"),
            "paired_volume_ml": anchor.get("paired_volume_ml"),
            "vital_alert_type": anchor.get("vital_alert_type", anchor.get("alert_type")),
            "vital_alert_start_sec": anchor.get("vital_alert_start_sec", anchor.get("alert_start_sec")),
            "vital_alert_end_sec": anchor.get("vital_alert_end_sec", anchor.get("alert_end_sec")),
            "joint_unknown_without_med": anchor.get("joint_unknown_without_med"),
            "intervention_type": intervention_type,
            "before": anchor.get("before"),
            "after": anchor.get("after"),
        },
        "unit_corrections": {
            "mbp_kpa_to_mmhg_applied": bool(pd.to_numeric(df_window.get("__mbp_unit_converted__", pd.Series([0])), errors="coerce").fillna(0).max() > 0)
        },
        "clinical_table_structured": {
            "diagnosis_and_surgery": {
                "dx": _to_snapshot_scalar(first_valid(row, ["dx"], "")),
                "department": department,
                "optype": optype,
                "opname": opname,
                "approach": approach,
                "position": position,
                "ane_type": ane_type,
            },
            "preop_metrics": preop_metrics,
            "intraop_summary": intraop_summary,
            "airway_and_lines": airway_lines,
            "timeline": clinical_timeline,
            "source_meta": clinical_source_meta,
        },
        "clinical_table_all_fields": clinical_row_all_fields,
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
        elif cfg.anchor_mode == "joint":
            med_anchors = find_anchors(df_case, cfg)
            vital_anchors = _find_vital_alert_anchors(df_case, cfg)
            anchors = _build_joint_anchors(
                med_anchors,
                vital_anchors,
                max_gap_sec=float(cfg.joint_link_max_gap_sec),
                require_med_link=bool(cfg.joint_require_med_link),
            )
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
                "Q should use only background + current/trend physiologic signals (no interpretation words), and ask for maintenance intervention strategy."
            )
        if med_key.endswith("_VOL") and smoothed_delta_ml is not None and smoothed_delta_ml < 1.0:
            return (
                "Question focus: this is a mild maintenance adjustment sample. "
                "Q should use only background + current/trend physiologic signals (no interpretation words), and ask for maintenance intervention strategy."
            )
    if itype in {"continuous_infusion", "unlabeled_context_snapshot"}:
        return (
            "Question focus: this is a maintenance/state-assessment sample. "
            "Q should use only background + current/trend physiologic signals (no interpretation words), and ask for maintenance intervention strategy, not emergency rescue."
        )
    return (
        "Question focus: this is an active-decision sample. "
        "Q should use only background + current/trend physiologic signals (no interpretation words), and ask for immediate intervention strategy."
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
        "Evidence locator format to cite in output: [M10#rank | 术中相关章节: ... | p.1493]",
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


def _build_q_signal_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    vital_stats = snapshot.get("vital_stats", {}) if isinstance(snapshot.get("vital_stats"), dict) else {}
    trend_key = ""
    trend_text_block: Dict[str, Any] = {}
    for key, value in snapshot.items():
        if str(key).startswith("vital_trend_last_") and isinstance(value, dict):
            trend_key = str(key).replace("vital_trend_", "")
            trend_text_block = value
            break

    signal_order = [
        "MBP",
        "SBP",
        "DBP",
        "HR",
        "SPO2",
        "ETCO2",
        "CO",
        "CI",
        "SV",
        "SVV",
        "PPV",
        "CVP",
        "SVR",
        "BT",
        "RSO2_L",
        "RSO2_R",
        "BIS",
    ]
    rename = {"MBP": "MAP", "SPO2": "SpO2", "RSO2_L": "rSO2_L", "RSO2_R": "rSO2_R"}
    signals: Dict[str, Any] = {}
    for vital_key in signal_order:
        out_key = rename.get(vital_key, vital_key)
        summary = vital_stats.get(vital_key) if isinstance(vital_stats.get(vital_key), dict) else None
        if summary is None:
            continue

        unit = VITAL_UNIT.get(vital_key, "")
        trend = trend_label(float(summary.get("slope", 0.0)))
        start = _format_value_with_unit(summary.get("start"), unit)
        end = _format_value_with_unit(summary.get("end"), unit)
        mean = _format_value_with_unit(summary.get("mean"), unit)
        signals[out_key] = {
            "current": end,
            "trend": f"{start} -> {end} ({trend})",
            "window_mean": mean,
        }

    return {
        "trend_window": trend_key or "last_window",
        "signals": signals,
        "available_signals": list(signals.keys()),
    }


def _build_q_visible_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    patient = snapshot.get("patient_background", {}) if isinstance(snapshot.get("patient_background"), dict) else {}
    patient_background = {
        "age": patient.get("age", "Unknown"),
        "sex": patient.get("sex", "Unknown"),
        "height_cm": patient.get("height_cm", "Unknown"),
        "weight_kg": patient.get("weight_kg", "Unknown"),
        "bmi": patient.get("bmi", "Unknown"),
        "asa": patient.get("asa", "Unknown"),
        "department": patient.get("department", "Unknown"),
        "surgery_group": patient.get("surgery_group", "Unknown"),
    }
    return {
        "patient_background": patient_background,
        "surgery_type": snapshot.get("surgery_type", ""),
        "intraop_stage": snapshot.get("intraop_stage", ""),
        "intraop_signal_state": _build_q_signal_context(snapshot),
        "unit_standard": CANONICAL_UNIT_GUIDE,
    }


def _build_decision_support_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "preop_context",
        "baseline_comparison",
        "clinical_assessment",
        "clinical_table_structured",
        "clinical_table_all_fields",
        "miller_alignment",
        "actual_intervention",
        "interpreted_intervention_type",
        "anchor_detail",
        "unit_corrections",
    ]
    out = {key: snapshot.get(key) for key in keep_keys if key in snapshot}
    clinical = out.get("clinical_assessment")
    if isinstance(clinical, dict):
        cleaned = dict(clinical)
        for list_key in ("risk_flags", "contextual_interpretation"):
            value = cleaned.get(list_key)
            if isinstance(value, list):
                cleaned[list_key] = [
                    x for x in value if not _mentions_missing_indicator(str(x))
                ]
        cleaned["missing_data_guidance"] = (
            "Use only available objective physiologic signals and trends. "
            "Do not mention unavailable indicators in final Q/A."
        )
        out["clinical_assessment"] = cleaned
    return out


def _expected_action_unit(snapshot: Dict[str, Any]) -> str:
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    med_key = str(anchor.get("medication_key", "")).strip()
    anchor_source = str(anchor.get("anchor_source", "")).strip().lower()
    itype = str(snapshot.get("interpreted_intervention_type", "")).strip()
    if not itype:
        itype = str(anchor.get("intervention_type", "")).strip()

    if med_key in {"ARR_EVENT", "UNLABELED_EVENT"} or anchor_source in {"arrdb", "periodic"}:
        return ""
    if med_key in {"SEVO_ET_RATE", "SEVO_FI_RATE", "DES_ET_RATE", "DES_FI_RATE", "ISO_ET_RATE", "ISO_FI_RATE"}:
        return "vol%"
    if med_key == "MAC_RATE":
        return "MAC"
    if itype == "bolus_like_event":
        return "mL"
    if med_key.endswith("_VOL"):
        smoothed_rate = _safe_float(anchor.get("smoothed_rate_ml_per_h"))
        if smoothed_rate is not None and smoothed_rate >= 0 and itype in {"continuous_infusion", "rate_adjustment"}:
            return "mL/h"
        return "mL"
    if med_key.endswith("_RATE"):
        return "mL/h"
    return ""


def _normalize_numeric_text(text: str) -> str:
    if not text:
        return ""
    return (
        str(text)
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("／", "/")
        .replace("–", "-")
        .replace("—", "-")
        .replace("～", "~")
    )


def _unit_regex(unit: str) -> str:
    if unit == "mL/h":
        return r"ml\s*/\s*h"
    if unit == "mL":
        return r"ml(?!\s*/\s*h)"
    if unit == "vol%":
        return r"(?:vol\s*%|%\s*vol)"
    if unit == "MAC":
        return r"mac\b"
    return re.escape(unit.lower())


def _extract_unit_value(text: str, unit: str) -> Optional[Dict[str, float]]:
    if not text or not unit:
        return None
    t = _normalize_numeric_text(text).lower()
    u = _unit_regex(unit)
    range_re = rf"(\d+(?:\.\d+)?)\s*(?:-|~|to|至)\s*(\d+(?:\.\d+)?)\s*{u}"
    m_range = re.search(range_re, t, flags=re.IGNORECASE)
    if m_range:
        a = float(m_range.group(1))
        b = float(m_range.group(2))
        lo = min(a, b)
        hi = max(a, b)
        return {"value": (lo + hi) / 2.0, "lo": lo, "hi": hi}
    single_re = rf"(\d+(?:\.\d+)?)\s*{u}"
    m_single = re.search(single_re, t, flags=re.IGNORECASE)
    if m_single:
        v = float(m_single.group(1))
        return {"value": v, "lo": v, "hi": v}
    return None


def _is_value_pair_close(v1: float, v2: float, unit: str) -> bool:
    if v1 <= 0 or v2 <= 0:
        return False
    ratio = max(v1, v2) / max(min(v1, v2), 1e-9)
    diff = abs(v1 - v2)
    if unit == "mL/h":
        return (diff <= 5.0) or (ratio <= 1.6)
    if unit == "mL":
        return (diff <= 1.0) or (ratio <= 2.0)
    if unit == "vol%":
        return (diff <= 1.0) or (ratio <= 1.5)
    if unit == "MAC":
        return (diff <= 0.3) or (ratio <= 1.5)
    return ratio <= 1.8


def _matches_target_value(extracted: Dict[str, float], target: float, unit: str) -> bool:
    if target <= 0:
        return True
    lo = extracted.get("lo")
    hi = extracted.get("hi")
    if lo is not None and hi is not None and lo <= target <= hi:
        return True
    return _is_value_pair_close(float(extracted.get("value", 0.0)), target, unit)


def _expected_action_value(snapshot: Dict[str, Any]) -> Optional[float]:
    unit = _expected_action_unit(snapshot)
    if not unit:
        return None
    anchor = snapshot.get("anchor_detail", {}) if isinstance(snapshot.get("anchor_detail"), dict) else {}
    if unit == "mL/h":
        for key in ("smoothed_rate_ml_per_h", "inferred_rate_ml_per_h"):
            v = _safe_float(anchor.get(key))
            if v is not None and v > 0:
                return float(v)
    elif unit == "mL":
        for key in ("smoothed_delta_volume_ml", "delta"):
            v = _safe_float(anchor.get(key))
            if v is not None:
                return abs(float(v))
    elif unit in {"vol%", "MAC"}:
        v = _safe_float(anchor.get("after"))
        if v is not None:
            return float(v)
    actual = str(snapshot.get("actual_intervention", ""))
    parsed = _extract_unit_value(actual, unit)
    if parsed:
        return float(parsed["value"])
    return None


def _extract_question_line(text: str) -> str:
    out = _extract_qa_block(text)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].startswith("Q:"):
        return lines[0][2:].strip()
    if lines[0].startswith("Q："):
        return lines[0][2:].strip()
    return lines[0].strip()


def _has_subjective_hints_in_q(text: str) -> bool:
    q_line = _extract_question_line(text)
    if not q_line:
        return True
    for pattern in Q_SUBJECTIVE_HINT_PATTERNS:
        if re.search(pattern, q_line, flags=re.IGNORECASE):
            return True
    return False


def _mentions_missing_indicator(text: str) -> bool:
    if not text:
        return False
    for pattern in MISSING_INDICATOR_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _q_has_intervention_question(text: str) -> bool:
    q_line = _extract_question_line(text)
    if not q_line:
        return False
    has_question_mark = ("?" in q_line) or ("？" in q_line)
    has_intervention_word = ("干预" in q_line) or ("措施" in q_line) or ("处理" in q_line)
    return has_question_mark and has_intervention_word


def build_user_prompt(snapshot: Dict[str, Any], retrieval: Optional[Dict[str, Any]] = None) -> str:
    q_visible_text = json.dumps(_build_q_visible_context(snapshot), ensure_ascii=False, indent=2)
    support_text = json.dumps(_build_decision_support_context(snapshot), ensure_ascii=False, indent=2)
    fewshot = _fewshot_text_for_snapshot(snapshot)
    golden = _golden_action_hint(snapshot)
    med_key = golden["medication_key"]
    actual = golden["actual_intervention"]
    kws = golden["keywords"]
    kw_text = ", ".join(kws) if kws else "N/A"
    decision_unit = _expected_action_unit(snapshot)
    decision_unit_text = decision_unit if decision_unit else "N/A (non-dose sample)"
    q_focus = _question_focus_instruction(snapshot)
    evidence_block = _format_miller_evidence(retrieval)
    has_evidence = bool(retrieval and isinstance(retrieval.get("results"), list) and retrieval.get("results"))
    evidence_rule = (
        "- 【决策干预（Miller）】必须严格遵循“诊断依据：...; 具体干预：...; 原文摘录：...”的模板结构。\n"
        "- 诊断依据：必须基于当前患者客观生理信息（至少包含MAP/HR/SpO2/BIS中的可用项及其趋势），不得使用空泛主观结论。\n"
        "- 具体干预：必须输出根据Miller推测出的、最符合患者当前情况的具体干预行动（必须包含具体用药和用药剂量）。\n"
        "- 原文摘录：必须直接来自检索证据原文（尽量保留英文原句关键短语），不要只写概述，并在结尾附证据定位标签，如 [M10#1 | 术中相关章节: ... | p.1493]。\n"
        "- 【决策干预（Miller）】不能只贴引文，必须把证据映射到当前患者生理状态并给出可执行决策。\n"
        "- 若某生理指标缺失，Q/A中不要提及该指标“缺失/暂无/不可用”，仅基于其余可用指标给出判断与干预。\n"
        if has_evidence
        else "- 若无检索证据，仍保持“诊断依据+具体干预+原文摘录”三段结构；在原文摘录末尾标注“证据定位不足”，并基于当前生理信号给出保守可执行策略。若某生理指标缺失，不要在Q/A中提及其缺失。\n"
    )
    return (
        "Q-visible context (the ONLY allowed source for the Q line):\n"
        f"{q_visible_text}\n\n"
        "Decision-support context (for A and intervention lines only; do NOT leak into Q wording):\n"
        f"{support_text}\n\n"
        f"{fewshot}\n"
        f"{evidence_block}"
        "Supervision policy for this dataset:\n"
        f"- logged_action (golden): {actual}\n"
        f"- medication_key: {med_key}\n"
        f"- expected drug keywords: {kw_text}\n"
        f"- required actionable dose/rate unit for both Miller and VitalDB lines: {decision_unit_text}\n"
        "You MUST align 【决策干预（VitalDB）】 with logged_action (same drug class/category), and provide a concrete executable order "
        "(drug + direction + dose/rate magnitude or target + reassessment time + escalation/stop condition). "
        "Do not output a contradictory drug.\n"
        f"{q_focus}\n"
        "Q-line constraints:\n"
        "- Q must only contain patient background + available intraoperative physiologic current values and trends.\n"
        "- Q should end with wording equivalent to: “结合手术背景，此时最合理的干预措施是什么？”.\n"
        "- NEVER include any clinical inference/interpretation words in Q (e.g., 提示/显示/稳定/不稳定/不足/过深/过浅).\n"
        "- If a physiologic indicator is unavailable, OMIT it from Q entirely; do not mention missing/unavailable/暂无/无效.\n"
        "- Q must NOT include guideline citation, retrieval locator, logged_action, or anchor metadata.\n"
        "Unit normalization policy:\n"
        "- Use MAP mmHg, HR bpm, SpO2 %, BIS index; infusion in mL/h; bolus in mL; volatile concentration in vol%.\n"
        "- Do not mix pressure units (e.g., do not use kPa once converted to mmHg).\n"
        "- In this sample, both 【决策干预（Miller）】 and 【决策干预（VitalDB）】 executable dose/rate must use the same actionable unit target.\n"
        "- If Miller source uses a different unit (e.g., μg/kg/min), convert or annotate to the VitalDB-comparable unit when possible.\n"
        "- If precise conversion is impossible due to missing body weight/drug concentration, keep original units and briefly state why conversion is not possible; never fabricate numbers.\n"
        "Clinical priority policy:\n"
        "- MAP absolute threshold is the perfusion floor; relative MAP drop is layered risk stratification.\n"
        "- SpO2 is a high-sensitivity oxygenation signal: sustained decline toward 95% deserves early attention, and <90% is critical.\n"
        "- EtCO2 is a high-sensitivity airway/ventilation signal: persistent disappearance (outside zeroing) or severe deviation requires immediate troubleshooting.\n"
        "- BIS must be interpreted with MAP/HR/SpO2 and surgical stimulation; do not use BIS as a standalone trigger.\n"
        "- 【决策干预（Miller）】 MUST be grounded primarily in the retrieved excerpts from Miller's Anesthesia, 10th edition, when such excerpts are provided.\n"
        "- Do not present generic anesthesia knowledge as if it were a Miller 10th edition recommendation unless it is supported by the retrieved excerpts.\n"
        "- If retrieved Miller evidence is incomplete or ambiguous, explicitly stay conservative and fall back to objective physiologic signals in the snapshot.\n"
        f"{evidence_rule}"
        "You MUST output EXACTLY ONE QA pair in Chinese with this strict format:\n\n"
        "Q: <描述病人背景、术中阶段、体征趋势的流畅段落, 最后提问“结合手术背景，此时最合理的干预措施是什么？”（可同义改写）；严禁任何临床推断/病理评价/暗示性词汇>\n"
        "A: 【临床推理】：<精炼总结核心病理生理机制>\n"
        "【决策干预（Miller）】：诊断依据：<结合当前值+趋势>; 具体干预：<基于第十版米勒检索证据的具体用药与剂量>; 原文摘录：<尽量保留英文原句关键短语并附定位 [M10#1 | 术中相关章节: ... | p.1493]>\n"
        "【决策干预（VitalDB）】：<与logged_action一致的实际策略，不得与golden冲突>\n\n"
        "The final QA block MUST be exactly these 4 lines (one line per label), no extra lines before or after.\n"
        "Use labels exactly as: Q:, A:, 【临床推理】, 【决策干预（Miller）】, 【决策干预（VitalDB）】.\n"
        "Do not change brackets, punctuation, or label names.\n"
        "If any physiologic indicator is unavailable, infer decisions from available objective signals only and do not mention missing indicators in the final Q/A text.\n"
        "Use 'clinical_assessment.risk_flags', 'contextual_interpretation', "
        "'baseline_comparison', and 'drug_reference' to improve realism.\n"
        "Do not output any text outside the final QA pair.\n"
        "Forbidden: instruction echo, Analyze/Strategy/Constraint Check, bullet list, self-correction text, subjective clinical interpretation/hints in Q."
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
    if _has_subjective_hints_in_q(out):
        return False
    if _mentions_missing_indicator(out):
        return False
    if not _q_has_intervention_question(out):
        return False
    miller_line = lines[2]
    has_diag = bool(re.search(r"诊断依据\s*[:：=]", miller_line))
    has_action = bool(re.search(r"具体干预\s*[:：=]", miller_line))
    has_quote = bool(re.search(r"原文摘录\s*[:：=]", miller_line))
    if not (has_diag and has_action and has_quote):
        return False
    has_m10 = bool(re.search(r"(?i)m10\s*#\d+", miller_line))
    has_page = bool(re.search(r"(?i)\bp\.\s*\d+\b", miller_line)) or ("页" in miller_line)
    has_paragraph = bool(re.search(r"段落\s*[:：]?\s*\d+", miller_line)) or bool(
        re.search(r"(?i)\bpara(?:graph)?\s*[:：#]?\s*\d+\b", miller_line)
    )
    has_chapter_hint = any(tok in miller_line for tok in ("章节", "Chapter", "相关章节"))
    has_locator = (has_m10 and (has_page or has_paragraph or has_chapter_hint)) or (
        has_chapter_hint and (has_page or has_paragraph)
    )
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


def _extract_miller_evidence_fields(text: str) -> Dict[str, str]:
    dec = _decision_section(text)
    quote = ""
    chapter_para = ""

    m_quote = re.search(r"原文摘录\s*[:：=]\s*[\"“](.*?)[\"”]", dec, re.IGNORECASE)
    if m_quote:
        quote = m_quote.group(1).strip()
    else:
        m_quote2 = re.search(r"证据原文\s*[:：=]\s*[\"“](.*?)[\"”]", dec, re.IGNORECASE)
        if m_quote2:
            quote = m_quote2.group(1).strip()

    m_chapter = re.search(r"章节段落\s*[:：=]\s*(\[[^\]]+\]|[^；。]+)", dec, re.IGNORECASE)
    if m_chapter:
        chapter_para = m_chapter.group(1).strip()
    else:
        m_locator = re.search(r"(\[M10\s*#\d+.*?\])", dec, re.IGNORECASE)
        if m_locator:
            chapter_para = m_locator.group(1).strip()

    return {
        "evidence_quote": quote,
        "chapter_paragraph": chapter_para,
    }


def _contains_expected_unit(text: str, unit: str) -> bool:
    if not unit:
        return True
    low = text.lower()
    if unit == "mL/h":
        return bool(re.search(r"\bml\s*/\s*h\b", low, re.IGNORECASE))
    if unit == "mL":
        return bool(re.search(r"\bml\b(?!\s*/\s*h)", low, re.IGNORECASE))
    if unit == "vol%":
        return ("vol%" in low) or bool(re.search(r"\bvol\s*%\b", low, re.IGNORECASE))
    if unit == "MAC":
        return "mac" in low
    return unit.lower() in low


def _has_unit_unconvertible_reason(text: str) -> bool:
    if not text:
        return False
    hints = (
        "无法换算",
        "不能换算",
        "缺少体重",
        "缺乏体重",
        "无体重",
        "缺少浓度",
        "缺乏浓度",
        "无药物浓度",
        "无法精确换算",
    )
    return any(h in text for h in hints)


def _is_concrete_miller_instruction(text: str, snapshot: Dict[str, Any]) -> bool:
    expected_unit = _expected_action_unit(snapshot)
    dec = _decision_section(text)
    if not dec:
        return False
    lower = dec.lower()
    action_tokens = ("上调", "下调", "滴定", "追加", "减量", "增量", "维持", "暂停", "给予", "复评", "titrate", "bolus")
    has_action = any(tok in dec for tok in action_tokens) or any(tok in lower for tok in action_tokens)
    has_recheck = bool(re.search(r"\d+(?:\.\d+)?\s*(?:s|sec|秒|min|分钟)", dec, re.IGNORECASE)) or ("复评" in dec)
    has_quant = bool(re.search(r"\d+(?:\.\d+)?\s*(?:mL/h|mL|mmHg|bpm|%|vol%|MAC)", dec, re.IGNORECASE))
    has_diag_kv = bool(re.search(r"诊断依据\s*[:：=]", dec))
    has_action_kv = bool(re.search(r"具体干预\s*[:：=]", dec))
    has_quote_kv = bool(re.search(r"原文摘录\s*[:：=]", dec))
    if not expected_unit:
        return has_action and has_recheck and has_quant and has_diag_kv and has_action_kv and has_quote_kv
    has_unit = _contains_expected_unit(dec, expected_unit) or _has_unit_unconvertible_reason(dec)
    return has_action and has_recheck and has_quant and has_unit and has_diag_kv and has_action_kv and has_quote_kv


def _is_unit_consistent_across_decisions(text: str, snapshot: Dict[str, Any]) -> bool:
    expected_unit = _expected_action_unit(snapshot)
    if not expected_unit:
        return True
    miller_dec = _decision_section(text)
    vital_dec = _decision_section_vitaldb(text)
    if _contains_expected_unit(miller_dec, expected_unit) and _contains_expected_unit(vital_dec, expected_unit):
        return True
    # Allow fallback when precise conversion is impossible and explicitly justified.
    if _has_unit_unconvertible_reason(miller_dec) or _has_unit_unconvertible_reason(vital_dec):
        return True
    return False


def _extract_structured_qa_fields(qa_text: Optional[str], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not qa_text or not isinstance(qa_text, str):
        return {
            "miller_evidence_kv": {"evidence_quote": "", "chapter_paragraph": ""},
            "unit_consistency": {"expected_action_unit": "", "consistent": False},
        }
    snap = snapshot if isinstance(snapshot, dict) else {}
    expected_unit = _expected_action_unit(snap)
    return {
        "miller_evidence_kv": _extract_miller_evidence_fields(qa_text),
        "unit_consistency": {
            "expected_action_unit": expected_unit,
            "consistent": _is_unit_consistent_across_decisions(qa_text, snap),
        },
    }


def _is_action_aligned(text: str, snapshot: Dict[str, Any]) -> bool:
    hint = _golden_action_hint(snapshot)
    kws = [str(k).strip() for k in hint.get("keywords", []) if str(k).strip()]
    if not kws:
        return True
    dec = _decision_section_vitaldb(text).lower()
    return any(k.lower() in dec for k in kws)


def _is_concrete_vitaldb_instruction(text: str, snapshot: Optional[Dict[str, Any]] = None) -> bool:
    dec = _decision_section_vitaldb(text)
    if not dec:
        return False
    lower = dec.lower()
    action_tokens = ("上调", "下调", "滴定", "追加", "减量", "增量", "维持", "暂停", "给予", "复评", "titrate", "bolus")
    has_action = any(tok in dec for tok in action_tokens) or any(tok in lower for tok in action_tokens)
    has_recheck = bool(re.search(r"\d+(?:\.\d+)?\s*(?:s|sec|秒|min|分钟)", dec, re.IGNORECASE)) or ("复评" in dec)
    has_quant = bool(re.search(r"\d+(?:\.\d+)?\s*(?:mL/h|mL|mmHg|bpm|%|vol%|MAC)", dec, re.IGNORECASE))
    if snapshot is None:
        return has_action and has_recheck and has_quant
    expected_unit = _expected_action_unit(snapshot)
    if not expected_unit:
        return has_action and has_recheck and has_quant
    return has_action and has_recheck and has_quant and (
        _contains_expected_unit(dec, expected_unit) or _has_unit_unconvertible_reason(dec)
    )


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
        "Q: <只包含病人背景+术中生理信号当前值与趋势的问题，结尾询问最合理干预措施>\n"
        "A: 【临床推理】：<1-3句>\n"
        "【决策干预（Miller）】：诊断依据：...; 具体干预：...; 原文摘录：\"...\" [M10#1 | 术中相关章节: ... | p.1493]\n"
        "【决策干预（VitalDB）】：<1-2句；必须是具体可执行指令（药物+方向+幅度/目标+复评时间+下一步条件），并与Miller执行单位一致>\n\n"
        "Do not output Analyze/Strategy/Constraint Check/self-correction text.\n"
        "Do not output anything outside the 4-line QA block.\n"
        "Q line must NOT include citation tags, logged_action, anchor metadata, or subjective clinical interpretation/hints.\n"
        "Miller 诊断依据 must cite objective physiologic signals/trends (MAP/HR/SpO2/BIS when available), not vague statements.\n"
        "Use normalized units: MAP mmHg, HR bpm, SpO2 %, BIS index, infusion mL/h, bolus mL, volatile vol%.\n"
        "Miller line must include at least one M10 locator token and contain all three parts: 诊断依据 + 具体干预 + 原文摘录.\n"
        "If precise unit conversion is impossible due to missing body weight or drug concentration, keep original units and briefly explain why; never fabricate numbers.\n"
        f"Golden logged_action: {actual}\n"
        f"Golden medication_key: {med_key}\n"
        f"Expected drug keywords in 【决策干预（VitalDB）】: {kw_text}\n"
        f"Required actionable dose/rate unit in BOTH decision lines: {_expected_action_unit(snapshot) or 'N/A'}\n"
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
    if (
        _is_strict_qa(cleaned)
        and _is_action_aligned(cleaned, snapshot)
        and _is_concrete_miller_instruction(cleaned, snapshot)
        and _is_concrete_vitaldb_instruction(cleaned, snapshot)
        and _is_unit_consistent_across_decisions(cleaned, snapshot)
    ):
        return cleaned

    repaired = _repair_qa_output(client, model, raw, snapshot)
    if repaired:
        repaired_cleaned = _extract_qa_block(repaired)
        if (
            _is_strict_qa(repaired_cleaned)
            and _is_action_aligned(repaired_cleaned, snapshot)
            and _is_concrete_miller_instruction(repaired_cleaned, snapshot)
            and _is_concrete_vitaldb_instruction(repaired_cleaned, snapshot)
            and _is_unit_consistent_across_decisions(repaired_cleaned, snapshot)
        ):
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
                        rec.update(_extract_structured_qa_fields(None, rec.get("snapshot", {})))
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
                rec.update(_extract_structured_qa_fields(rec.get("llm_output"), rec.get("snapshot", {})))
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
            records[idx].update(_extract_structured_qa_fields(qa, records[idx].get("snapshot", {})))
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
    co_now = _safe_float(recent.get("CO_L_min"))
    ci_now = _safe_float(recent.get("CI_L_min_m2"))
    sv_now = _safe_float(recent.get("SV_ml"))
    ppv_now = _safe_float(recent.get("PPV_pct"))
    svr_now = _safe_float(recent.get("SVR_dyns_cm5"))
    map_drop_pct = _safe_float(baseline.get("MAP_drop_from_baseline_pct"))
    map_lt_65_persist = _safe_float(persistence.get("map_lt_65")) or 0.0
    map_lt_55_persist = _safe_float(persistence.get("map_lt_55")) or 0.0
    spo2_lt_90_persist = _safe_float(persistence.get("spo2_lt_90")) or 0.0
    ci_lt_low_persist = _safe_float(persistence.get("ci_lt_low")) or 0.0
    co_lt_low_persist = _safe_float(persistence.get("co_lt_low")) or 0.0
    sv_lt_low_persist = _safe_float(persistence.get("sv_lt_low")) or 0.0
    ppv_ge_13_persist = _safe_float(persistence.get("ppv_ge_13")) or 0.0
    svr_lt_low_persist = _safe_float(persistence.get("svr_lt_low")) or 0.0

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
    advanced_low_output = (
        (ci_now is not None and ci_now < ANES_THRESHOLDS["ci_low_l_min_m2"])
        or (co_now is not None and co_now < ANES_THRESHOLDS["co_low_l_min"])
        or (sv_now is not None and sv_now < ANES_THRESHOLDS["sv_low_ml"])
        or ci_lt_low_persist >= hemo_sec
        or co_lt_low_persist >= hemo_sec
        or sv_lt_low_persist >= hemo_sec
    )
    advanced_volume_responsive = (
        (ppv_now is not None and ppv_now >= ANES_THRESHOLDS["ppv_high_pct"])
        or ppv_ge_13_persist >= hemo_sec
    )
    advanced_vasodilation = (
        (svr_now is not None and svr_now < ANES_THRESHOLDS["svr_low_dyns_cm5"])
        or svr_lt_low_persist >= hemo_sec
    )
    if advanced_low_output:
        _add_strategy("assess_cardiac_output_and_perfusion", "CO/CI/SV提示低心排趋势，应优先复核灌注和容量。")
    if advanced_volume_responsive:
        _add_strategy("consider_fluid_responsiveness", "PPV升高提示容量反应性，建议先做容量优化评估。")
    if advanced_vasodilation:
        _add_strategy("consider_vasopressor_support", "SVR偏低提示血管扩张，可考虑血管活性支持。")

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
    if isinstance(snap, dict) and not _is_concrete_miller_instruction(cleaned, snap):
        return False
    if not _is_concrete_vitaldb_instruction(cleaned, snap if isinstance(snap, dict) else None):
        return False
    if isinstance(snap, dict) and not _is_unit_consistent_across_decisions(cleaned, snap):
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
                if valid:
                    snap = rec.get("snapshot", {}) if isinstance(rec.get("snapshot"), dict) else {}
                    valid = (
                        _is_concrete_miller_instruction(cleaned, snap)
                        and _is_concrete_vitaldb_instruction(cleaned, snap)
                        and _is_unit_consistent_across_decisions(cleaned, snap)
                    )
                rec[field] = cleaned
                rec.update(_extract_structured_qa_fields(cleaned, rec.get("snapshot", {})))
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
    parser.add_argument(
        "--rate-sustained-pre-window-sec",
        type=float,
        default=90.0,
        help="For Orchestra *_RATE anchors: pre window seconds for sustained-change check.",
    )
    parser.add_argument(
        "--rate-sustained-post-window-sec",
        type=float,
        default=120.0,
        help="For Orchestra *_RATE anchors: post window seconds for sustained-change check.",
    )
    parser.add_argument(
        "--rate-sustained-min-abs-delta",
        type=float,
        default=3.0,
        help="Minimum sustained median rate change (mL/h) to keep RATE anchor.",
    )
    parser.add_argument(
        "--rate-sustained-min-ratio",
        type=float,
        default=0.6,
        help="Minimum directional consistency ratio in post window for RATE anchor (0-1).",
    )
    parser.add_argument(
        "--rate-sustained-min-points",
        type=int,
        default=10,
        help="Minimum non-null points required in each pre/post window for sustained RATE check.",
    )
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
        choices=["medication", "arrdb", "hybrid", "periodic", "joint"],
        help="Anchor source mode: medication deltas, arrdb labels, hybrid, periodic, or joint (vital-alert + medication linkage).",
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
    parser.add_argument(
        "--joint-link-max-gap-sec",
        type=float,
        default=60.0,
        help="For joint mode: max allowed time gap between vital-alert anchor and nearest medication anchor.",
    )
    parser.add_argument(
        "--joint-allow-unknown-without-med",
        action="store_true",
        help="For joint mode: keep vital-alert anchors even when no nearby medication anchor exists.",
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
        default="dynamic",
        choices=["dynamic", "full", "paired_only", "off"],
        help="How BIS drives Miller retrieval intents: dynamic(推荐) / full / paired_only / off.",
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
        rate_sustained_pre_window_sec=max(20.0, float(args.rate_sustained_pre_window_sec)),
        rate_sustained_post_window_sec=max(20.0, float(args.rate_sustained_post_window_sec)),
        rate_sustained_min_abs_delta=max(0.5, float(args.rate_sustained_min_abs_delta)),
        rate_sustained_min_ratio=max(0.0, min(1.0, float(args.rate_sustained_min_ratio))),
        rate_sustained_min_points=max(3, int(args.rate_sustained_min_points)),
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
        joint_link_max_gap_sec=max(5.0, float(args.joint_link_max_gap_sec)),
        joint_require_med_link=(not bool(args.joint_allow_unknown_without_med)),
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

