"""Clinical/vital reference constants extracted from anes_pipeline.py."""

from typing import Any, Dict, List

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

# Medication display constants moved to anes_medication_constants.py

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
