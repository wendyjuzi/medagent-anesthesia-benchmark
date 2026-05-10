"""Shared safety checks for VitalDB QA output validation."""

import re
from typing import Any, Dict, List, Optional, Sequence


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


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


def _has_ordered_transition_near(text: str, before: float, after: float, tolerance: float) -> bool:
    matches: List[tuple[float, int, int]] = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", str(text or "")):
        try:
            matches.append((float(m.group(0)), m.start(), m.end()))
        except Exception:
            continue
    for bv, bs, be in matches:
        if abs(bv - float(before)) > tolerance:
            continue
        for av, ast, ae in matches:
            if ast <= be:
                continue
            if abs(av - float(after)) > tolerance:
                continue
            between = str(text or "")[be:ast]
            span = str(text or "")[max(0, bs - 12):min(len(str(text or "")), ae + 12)]
            if ast - be <= 80 and re.search(r"(->|→|至|到|变为|升至|降至|上调|下调|由)", between + span):
                return True
    return False


def vitaldb_logged_action_consistent(intervention_text: str, snapshot: Optional[Dict[str, Any]]) -> bool:
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
    if med_key.endswith("_VOL"):
        rate_vals = [float(x) for x in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*mL/h", text, flags=re.IGNORECASE)]
        allowed_rates: List[float] = []
        smoothed_rate = _to_float(anchor.get("smoothed_rate_ml_per_h"))
        smoothed_dt = _to_float(anchor.get("smoothed_dt_sec"))
        if smoothed_rate is not None and (smoothed_dt is None or smoothed_dt >= 30.0):
            allowed_rates.append(smoothed_rate)
        inferred_rate = _to_float(anchor.get("inferred_rate_ml_per_h"))
        dt_sec = _to_float(anchor.get("dt_sec"))
        if inferred_rate is not None and dt_sec is not None and dt_sec >= 10.0:
            allowed_rates.append(inferred_rate)
        actual_txt = str(snapshot.get("actual_intervention") or "")
        if actual_txt:
            for m in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*mL/h", actual_txt, flags=re.IGNORECASE):
                try:
                    allowed_rates.append(float(m))
                except Exception:
                    continue
        if allowed_rates:
            return bool(rate_vals) and any(
                any(abs(rv - ar) <= 2.0 for ar in allowed_rates) for rv in rate_vals
            )
        # For cumulative-volume anchors without a reliable derived rate, do not
        # force before/after cumulative volumes into the clinical answer.
        return True
    if before is not None and after is not None:
        tol = 0.75 if med_key.endswith("_RATE") else 0.5
        if med_key.endswith("_RATE") and not _has_ordered_transition_near(text, before, after, tol):
            return False
        ok = _has_number_near(values, before, tol) and _has_number_near(values, after, tol)
        if not ok:
            return False
        return True
    if delta is not None:
        return _has_number_near(values, abs(delta), 0.75) or _has_number_near(values, delta, 0.75)
    return True


def has_internal_metadata_leak(text: str) -> bool:
    t = str(text or "")
    patterns = [
        r"(?i)\bmed_key\s*=",
        r"(?i)\bmed_key\s*[：:]",
        r"(?i)\blogged_action\s*=",
        r"(?i)\blogged_action\s*[：:]",
        r"(?i)\bkeywords\s*=",
        r"(?i)\bkeywords\s*[：:]",
        r"(?i)\banchor_detail\b",
    ]
    return any(re.search(p, t) for p in patterns)
