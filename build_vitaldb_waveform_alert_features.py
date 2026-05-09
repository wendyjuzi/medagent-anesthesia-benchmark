import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


TRACK_CANDIDATES: Dict[str, List[str]] = {
    "ETCO2": [
        "Solar8000/ETCO2",
        "Solar8000/ETCO2_MMHG",
        "Primus/ETCO2",
        "Primus/ETCO2_MMHG",
        "IntelliVue/EtCO2",
    ],
    "SPO2": [
        "Solar8000/PLETH_SPO2",
        "Solar8000/SPO2",
        "IntelliVue/SpO2",
        "Primus/SPO2",
    ],
    "HR": [
        "Solar8000/HR",
        "IntelliVue/HR",
        "SNUADC/HR",
        "Primus/HR",
    ],
    "MAP": [
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
    "BT": [
        "Solar8000/BT",
        "IntelliVue/BT",
        "SNUADC/BT",
    ],
    "BIS": [
        "BIS/BIS",
    ],
    "RSO2_L": [
        "INVOS/rSO2_L",
        "INVOS/RSO2_L",
    ],
    "RSO2_R": [
        "INVOS/rSO2_R",
        "INVOS/RSO2_R",
    ],
    "SVV": [
        "EV1000/SVV",
        "Vigileo/SVV",
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
    "ECG_II": [
        "Solar8000/ECG_II",
        "IntelliVue/ECG_II",
        "SNUADC/ECG_II",
    ],
    "ECG_V5": [
        "Solar8000/ECG_V5",
        "IntelliVue/ECG_V5",
        "SNUADC/ECG_V5",
    ],
}


def _safe_get_track_names(vf: object) -> List[str]:
    if hasattr(vf, "get_track_names"):
        names = vf.get_track_names()
        if isinstance(names, (list, tuple)):
            return [str(x) for x in names]
    if hasattr(vf, "trks"):
        trks = getattr(vf, "trks")
        if isinstance(trks, dict):
            return [str(x) for x in trks.keys()]
    return []


def _pick_first_track(available: Sequence[str], candidates: Sequence[str]) -> str:
    aset = set(available)
    for name in candidates:
        if name in aset:
            return name
    return ""


def _run_max_true(mask: pd.Series) -> int:
    if mask is None or mask.empty:
        return 0
    m = mask.fillna(False).astype(bool).tolist()
    best = 0
    cur = 0
    for v in m:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _dur_sec(mask: pd.Series, interval_sec: float) -> float:
    if mask is None or mask.empty:
        return 0.0
    return float(mask.fillna(False).astype(bool).sum()) * float(interval_sec)


def _slice_between_first_last_valid(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    valid = s.notna()
    if not bool(valid.any()):
        return pd.Series(dtype=float)
    first = valid.idxmax()
    last = valid.iloc[::-1].idxmax()
    return s.loc[first:last]


def _baseline_from_prefix(s: pd.Series, n_prefix: int) -> Optional[float]:
    if s is None or s.empty:
        return None
    prefix = pd.to_numeric(s.head(n_prefix), errors="coerce").dropna()
    if prefix.empty:
        return None
    return float(prefix.median())


def _to_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _clean_range(s: pd.Series, low: float, high: float) -> pd.Series:
    if s is None or s.empty:
        return s
    out = s.copy()
    out[(out < low) | (out > high)] = pd.NA
    return out


def _add_stats(out: Dict[str, object], key: str, s: pd.Series, unit_suffix: str) -> None:
    prefix = key.lower()
    if s is None or s.empty or int(s.notna().sum()) == 0:
        out[f"{prefix}_mean_{unit_suffix}"] = None
        out[f"{prefix}_min_{unit_suffix}"] = None
        out[f"{prefix}_max_{unit_suffix}"] = None
        out[f"{prefix}_non_null"] = 0
        return
    out[f"{prefix}_mean_{unit_suffix}"] = float(s.mean())
    out[f"{prefix}_min_{unit_suffix}"] = float(s.min())
    out[f"{prefix}_max_{unit_suffix}"] = float(s.max())
    out[f"{prefix}_non_null"] = int(s.notna().sum())


def extract_case_features(caseid: int, interval_sec: float, retries: int = 1) -> Dict[str, object]:
    import vitaldb

    out: Dict[str, object] = {"caseid": int(caseid), "fetch_ok": 0, "fetch_error": ""}
    for key in TRACK_CANDIDATES:
        out[f"{key.lower()}_track"] = ""
        out[f"has_{key.lower()}_track"] = 0

    last_err = ""
    for attempt in range(max(1, retries + 1)):
        try:
            vf = vitaldb.VitalFile(int(caseid))
            available = _safe_get_track_names(vf)
            selected: List[str] = []
            chosen: Dict[str, str] = {}
            for key, cands in TRACK_CANDIDATES.items():
                trk = _pick_first_track(available, cands)
                chosen[key] = trk
                out[f"{key.lower()}_track"] = trk
                out[f"has_{key.lower()}_track"] = 1 if trk else 0
                if trk:
                    selected.append(trk)
            selected = list(dict.fromkeys(selected))
            if not selected:
                out["fetch_error"] = "no_selected_tracks"
                return out

            sig = vf.to_pandas(selected, interval_sec)
            if sig is None or sig.empty:
                out["fetch_error"] = "empty_signals"
                return out

            s_etco2 = _to_num_series(sig, chosen.get("ETCO2", ""))
            s_spo2 = _to_num_series(sig, chosen.get("SPO2", ""))
            s_hr = _to_num_series(sig, chosen.get("HR", ""))
            s_map = _to_num_series(sig, chosen.get("MAP", ""))
            s_sbp = _to_num_series(sig, chosen.get("SBP", ""))
            s_dbp = _to_num_series(sig, chosen.get("DBP", ""))
            s_bt = _to_num_series(sig, chosen.get("BT", ""))
            s_bis = _to_num_series(sig, chosen.get("BIS", ""))
            s_rso2_l = _to_num_series(sig, chosen.get("RSO2_L", ""))
            s_rso2_r = _to_num_series(sig, chosen.get("RSO2_R", ""))
            s_svv = _to_num_series(sig, chosen.get("SVV", ""))
            s_ppv = _to_num_series(sig, chosen.get("PPV", ""))
            s_cvp = _to_num_series(sig, chosen.get("CVP", ""))
            s_co = _to_num_series(sig, chosen.get("CO", ""))
            s_ci = _to_num_series(sig, chosen.get("CI", ""))
            s_sv = _to_num_series(sig, chosen.get("SV", ""))
            s_svr = _to_num_series(sig, chosen.get("SVR", ""))

            # Remove obvious monitor artifacts before computing alerts.
            s_etco2 = _clean_range(s_etco2, 0.0, 120.0)
            s_spo2 = _clean_range(s_spo2, 0.0, 100.0)
            s_hr = _clean_range(s_hr, 0.0, 250.0)
            s_map = _clean_range(s_map, 20.0, 250.0)
            s_sbp = _clean_range(s_sbp, 0.0, 300.0)
            s_dbp = _clean_range(s_dbp, 0.0, 200.0)
            s_bt = _clean_range(s_bt, 20.0, 45.0)
            s_bis = _clean_range(s_bis, 0.0, 100.0)
            s_rso2_l = _clean_range(s_rso2_l, 0.0, 100.0)
            s_rso2_r = _clean_range(s_rso2_r, 0.0, 100.0)
            s_svv = _clean_range(s_svv, 0.0, 100.0)
            s_ppv = _clean_range(s_ppv, 0.0, 100.0)
            s_cvp = _clean_range(s_cvp, -20.0, 50.0)
            s_co = _clean_range(s_co, 0.5, 20.0)
            s_ci = _clean_range(s_ci, 0.5, 10.0)
            s_sv = _clean_range(s_sv, 10.0, 250.0)
            s_svr = _clean_range(s_svr, 100.0, 5000.0)

            out["n_samples"] = int(len(sig))
            out["window_total_sec"] = float(len(sig)) * float(interval_sec)
            out["has_ecg_track"] = int(
                bool(chosen.get("ECG_II")) or bool(chosen.get("ECG_V5"))
            )

            _add_stats(out, "ETCO2", s_etco2, "mmhg")
            _add_stats(out, "SPO2", s_spo2, "pct")
            _add_stats(out, "HR", s_hr, "bpm")
            _add_stats(out, "MAP", s_map, "mmhg")
            _add_stats(out, "SBP", s_sbp, "mmhg")
            _add_stats(out, "DBP", s_dbp, "mmhg")
            _add_stats(out, "BT", s_bt, "c")
            _add_stats(out, "BIS", s_bis, "idx")
            _add_stats(out, "RSO2_L", s_rso2_l, "pct")
            _add_stats(out, "RSO2_R", s_rso2_r, "pct")
            _add_stats(out, "SVV", s_svv, "pct")
            _add_stats(out, "PPV", s_ppv, "pct")
            _add_stats(out, "CVP", s_cvp, "mmhg")
            _add_stats(out, "CO", s_co, "l_min")
            _add_stats(out, "CI", s_ci, "l_min_m2")
            _add_stats(out, "SV", s_sv, "ml")
            _add_stats(out, "SVR", s_svr, "dyns_cm5")

            # Respiratory alerts
            out["spo2_lt_90_sec"] = _dur_sec(s_spo2 < 90.0, interval_sec)
            out["spo2_lt_94_sec"] = _dur_sec(s_spo2 < 94.0, interval_sec)
            out["spo2_lt_90_max_contiguous_sec"] = float(_run_max_true(s_spo2 < 90.0)) * interval_sec
            out["spo2_lt_94_max_contiguous_sec"] = float(_run_max_true(s_spo2 < 94.0)) * interval_sec
            out["alert_spo2_hypoxemia"] = int(out["spo2_lt_90_max_contiguous_sec"] >= 30.0)

            etco2_scoped = _slice_between_first_last_valid(s_etco2)
            etco2_missing = etco2_scoped.isna()
            etco2_zero_like = etco2_scoped.notna() & (etco2_scoped <= 2.0)
            out["etco2_scoped_samples"] = int(len(etco2_scoped))
            out["etco2_missing_total_sec"] = _dur_sec(etco2_missing, interval_sec)
            out["etco2_zero_like_sec"] = _dur_sec(etco2_zero_like, interval_sec)
            out["etco2_missing_max_contiguous_sec"] = float(_run_max_true(etco2_missing)) * interval_sec
            out["alert_etco2_missing_non_zeroing"] = int(
                out["etco2_missing_max_contiguous_sec"] >= 2.0 and out["etco2_zero_like_sec"] < 6.0
            )
            out["etco2_lt_30_sec"] = _dur_sec(s_etco2 < 30.0, interval_sec)
            out["etco2_gt_50_sec"] = _dur_sec(s_etco2 > 50.0, interval_sec)

            # Hemodynamic alerts
            out["hr_gt_100_sec"] = _dur_sec(s_hr > 100.0, interval_sec)
            out["hr_lt_50_sec"] = _dur_sec(s_hr < 50.0, interval_sec)
            out["map_lt_65_sec"] = _dur_sec(s_map < 65.0, interval_sec)
            out["map_lt_55_sec"] = _dur_sec(s_map < 55.0, interval_sec)
            out["sbp_lt_90_sec"] = _dur_sec(s_sbp < 90.0, interval_sec)
            out["sbp_gt_180_sec"] = _dur_sec(s_sbp > 180.0, interval_sec)
            out["dbp_lt_60_sec"] = _dur_sec(s_dbp < 60.0, interval_sec)
            out["dbp_gt_100_sec"] = _dur_sec(s_dbp > 100.0, interval_sec)
            out["alert_map_shock_risk"] = int(out["map_lt_65_sec"] >= 60.0 or out["map_lt_55_sec"] >= 30.0)

            # Temperature
            out["bt_lt_36_sec"] = _dur_sec(s_bt < 36.0, interval_sec)
            out["bt_gt_37_5_sec"] = _dur_sec(s_bt > 37.5, interval_sec)
            out["bt_ge_38_sec"] = _dur_sec(s_bt >= 38.0, interval_sec)

            # BIS supportive
            out["bis_gt_60_sec"] = _dur_sec(s_bis > 60.0, interval_sec)
            out["bis_lt_40_sec"] = _dur_sec(s_bis < 40.0, interval_sec)

            # rSO2
            out["rso2_l_lt_55_sec"] = _dur_sec(s_rso2_l < 55.0, interval_sec)
            out["rso2_r_lt_55_sec"] = _dur_sec(s_rso2_r < 55.0, interval_sec)
            prefix_n = max(1, int(round(120.0 / max(interval_sec, 0.1))))
            base_l = _baseline_from_prefix(s_rso2_l, prefix_n)
            base_r = _baseline_from_prefix(s_rso2_r, prefix_n)
            out["rso2_l_baseline"] = base_l
            out["rso2_r_baseline"] = base_r
            if base_l is not None and base_l > 1e-6:
                drop_l = ((base_l - s_rso2_l) / base_l) * 100.0
                out["rso2_l_drop_ge_20_sec"] = _dur_sec(drop_l >= 20.0, interval_sec)
            else:
                out["rso2_l_drop_ge_20_sec"] = 0.0
            if base_r is not None and base_r > 1e-6:
                drop_r = ((base_r - s_rso2_r) / base_r) * 100.0
                out["rso2_r_drop_ge_20_sec"] = _dur_sec(drop_r >= 20.0, interval_sec)
            else:
                out["rso2_r_drop_ge_20_sec"] = 0.0

            # Volume-related indicators
            out["svv_ge_13_sec"] = _dur_sec(s_svv >= 13.0, interval_sec)
            out["svv_ge_18_sec"] = _dur_sec(s_svv >= 18.0, interval_sec)
            out["ppv_ge_13_sec"] = _dur_sec(s_ppv >= 13.0, interval_sec)
            out["ppv_ge_18_sec"] = _dur_sec(s_ppv >= 18.0, interval_sec)
            out["cvp_le_2_sec"] = _dur_sec(s_cvp <= 2.0, interval_sec)
            out["cvp_ge_15_sec"] = _dur_sec(s_cvp >= 15.0, interval_sec)
            out["co_lt_4_sec"] = _dur_sec(s_co < 4.0, interval_sec)
            out["co_gt_8_sec"] = _dur_sec(s_co > 8.0, interval_sec)
            out["ci_lt_2_5_sec"] = _dur_sec(s_ci < 2.5, interval_sec)
            out["ci_gt_4_0_sec"] = _dur_sec(s_ci > 4.0, interval_sec)
            out["sv_lt_60_sec"] = _dur_sec(s_sv < 60.0, interval_sec)
            out["sv_gt_100_sec"] = _dur_sec(s_sv > 100.0, interval_sec)
            out["svr_lt_800_sec"] = _dur_sec(s_svr < 800.0, interval_sec)
            out["svr_gt_1600_sec"] = _dur_sec(s_svr > 1600.0, interval_sec)

            out["fetch_ok"] = 1
            out["fetch_error"] = ""
            return out
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(1.0 * (2**attempt))

    out["fetch_error"] = last_err or "unknown_error"
    return out


def _normalize_caseid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "caseid" in out.columns:
        col = "caseid"
    elif "case_id" in out.columns:
        col = "case_id"
    else:
        raise ValueError("Input CSV must contain caseid or case_id")
    out["caseid"] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out = out.dropna(subset=["caseid"]).copy()
    out["caseid"] = out["caseid"].astype(int)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull VitalDB waveform tracks by caseid and build objective alert-duration features, "
            "then merge back to a case-level CSV."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Input case-level CSV with caseid column.")
    parser.add_argument("--output-csv", required=True, help="Output enriched CSV path.")
    parser.add_argument(
        "--features-only-csv",
        default="",
        help="Optional path to save per-case waveform features only.",
    )
    parser.add_argument("--interval-sec", type=float, default=2.0, help="Resampling interval seconds.")
    parser.add_argument("--max-cases", type=int, default=0, help="Process first N unique caseids (0=all).")
    parser.add_argument("--case-id-min", type=int, default=0, help="Only process caseid >= this value.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N cases.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per case when VitalDB fetch fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_csv = Path(args.output_csv)
    feature_csv = Path(args.features_only_csv) if str(args.features_only_csv).strip() else None

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = _normalize_caseid(pd.read_csv(in_csv))
    caseids = sorted(set(int(x) for x in df["caseid"].tolist() if int(x) >= int(args.case_id_min)))
    if args.max_cases > 0:
        caseids = caseids[: int(args.max_cases)]
    if not caseids:
        raise ValueError("No caseids to process.")

    rows: List[Dict[str, object]] = []
    total = len(caseids)
    ok = 0
    failed = 0
    t0 = time.time()
    for idx, caseid in enumerate(caseids, start=1):
        rec = extract_case_features(caseid=caseid, interval_sec=float(args.interval_sec), retries=int(args.retries))
        rows.append(rec)
        if int(rec.get("fetch_ok", 0)) == 1:
            ok += 1
        else:
            failed += 1
        if idx % max(1, int(args.progress_every)) == 0 or idx == total:
            elapsed = time.time() - t0
            print(
                f"[progress] {idx}/{total} ok={ok} failed={failed} "
                f"elapsed={elapsed:.1f}s avg={elapsed/max(1, idx):.2f}s/case"
            )

    feat_df = pd.DataFrame(rows)
    merged = df.merge(feat_df, on="caseid", how="left")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[done] enriched csv: {out_csv}")
    print(f"[done] rows={len(merged)} unique_caseids={merged['caseid'].nunique()}")
    print(f"[done] fetch_ok={int(pd.to_numeric(merged['fetch_ok'], errors='coerce').fillna(0).sum())}")

    if feature_csv is not None:
        feature_csv.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(feature_csv, index=False)
        print(f"[done] features only csv: {feature_csv}")


if __name__ == "__main__":
    main()
