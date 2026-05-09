import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import pandas as pd

CASES_API_URL = "https://api.vitaldb.net/cases"
LABS_API_URL = "https://api.vitaldb.net/labs"

# Candidates are ordered by preference.
HEMO_TRACK_CANDIDATES: Dict[str, Sequence[str]] = {
    "co": ("EV1000/CO", "Vigileo/CO", "CardioQ/CO"),
    "ci": ("EV1000/CI", "Vigileo/CI", "CardioQ/CI"),
    "sv": ("EV1000/SV", "Vigileo/SV", "CardioQ/SV"),
    "svr": ("EV1000/SVR", "Vigileo/SVR"),
    "svv": ("EV1000/SVV", "Vigileo/SVV"),
}

LAB_ABG_NAME_TO_COL: Dict[str, str] = {
    "po2": "abg_po2",
    "pco2": "abg_pco2",
    "ph": "abg_ph",
    "be": "abg_be",
    "k": "abg_k",
    "lac": "abg_lac",
    "sao2": "abg_sao2",
}

BASE_ABG_FALLBACK: Dict[str, Sequence[str]] = {
    "abg_po2": ("preop_pao2", "preop_po2"),
    "abg_pco2": ("preop_paco2", "preop_pco2"),
    "abg_ph": ("preop_ph",),
    "abg_be": ("preop_be",),
    "abg_k": ("preop_k",),
    "abg_lac": ("preop_lac", "preop_lactate"),
    "abg_sao2": ("preop_sao2",),
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _normalize_caseid(df: pd.DataFrame, prefer_col: str = "caseid") -> pd.DataFrame:
    out = df.copy()
    if prefer_col in out.columns:
        col = prefer_col
    elif "case_id" in out.columns:
        col = "case_id"
    else:
        raise ValueError("Input CSV must contain 'caseid' or 'case_id'")
    out["caseid"] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out = out.dropna(subset=["caseid"]).copy()
    out["caseid"] = out["caseid"].astype(int)
    return out


def _fill_if_missing(df: pd.DataFrame, col: str, default_value: object) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = default_value
    else:
        out[col] = out[col].fillna(default_value)
    return out


def _safe_float(v: object, default: float) -> float:
    try:
        x = float(v)
        if pd.isna(x):
            return default
        return x
    except Exception:  # noqa: BLE001
        return default


def _load_base_dataframe(base_clinical_csv: Path, refresh_from_vitaldb_api: bool) -> pd.DataFrame:
    if refresh_from_vitaldb_api:
        print(f"[info] downloading cases from {CASES_API_URL}")
        return pd.read_csv(CASES_API_URL, compression="gzip")
    return _load_csv(base_clinical_csv)


def _select_first_existing_numeric(
    df: pd.DataFrame,
    dst_col: str,
    src_candidates: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    if dst_col not in out.columns:
        out[dst_col] = pd.NA
    for src in src_candidates:
        if src not in out.columns:
            continue
        src_values = pd.to_numeric(out[src], errors="coerce")
        dst_values = pd.to_numeric(out[dst_col], errors="coerce")
        out[dst_col] = dst_values.where(dst_values.notna(), src_values)
    return out


def _fetch_track_case_sets() -> Dict[str, Dict[str, Set[int]]]:
    import vitaldb

    result: Dict[str, Dict[str, Set[int]]] = {}
    for metric, tracks in HEMO_TRACK_CANDIDATES.items():
        metric_sets: Dict[str, Set[int]] = {}
        for track in tracks:
            try:
                ids = vitaldb.find_cases([track])
                case_set = {int(x) for x in ids}
            except Exception as exc:  # noqa: BLE001
                case_set = set()
                print(f"[warn] find_cases failed for {track}: {exc}")
            metric_sets[track] = case_set
            print(f"[info] {track} cases={len(case_set)}")
        result[metric] = metric_sets
    return result


def _pick_track_for_case(caseid: int, track_sets: Dict[str, Set[int]]) -> str:
    for track, case_set in track_sets.items():
        if caseid in case_set:
            return track
    return ""


def _enrich_hemo_flags(df: pd.DataFrame, enable_hemo_track_enrichment: bool) -> pd.DataFrame:
    if not enable_hemo_track_enrichment:
        return df

    out = df.copy()
    case_sets_by_metric = _fetch_track_case_sets()
    case_ids = out["caseid"].astype(int).tolist()

    for metric in HEMO_TRACK_CANDIDATES.keys():
        track_col = f"{metric}_track"
        has_col = f"has_{metric}_track"
        chosen_tracks = []
        has_flags = []
        track_sets = case_sets_by_metric.get(metric, {})
        for caseid in case_ids:
            chosen = _pick_track_for_case(caseid, track_sets)
            chosen_tracks.append(chosen)
            has_flags.append(1 if chosen else 0)
        out[track_col] = chosen_tracks
        out[has_col] = has_flags

    any_cols = [f"has_{m}_track" for m in HEMO_TRACK_CANDIDATES.keys()]
    out["has_advanced_hemodynamics"] = out[any_cols].max(axis=1).astype(int)
    return out


def _build_abg_summary_from_labs() -> pd.DataFrame:
    print(f"[info] downloading labs from {LABS_API_URL}")
    labs_df = pd.read_csv(LABS_API_URL, compression="gzip")
    labs_df = labs_df.copy()
    labs_df["caseid"] = pd.to_numeric(labs_df["caseid"], errors="coerce")
    labs_df["dt"] = pd.to_numeric(labs_df["dt"], errors="coerce")
    labs_df["name"] = labs_df["name"].astype(str).str.strip().str.lower()
    labs_df["result"] = pd.to_numeric(labs_df["result"], errors="coerce")
    labs_df = labs_df.dropna(subset=["caseid", "dt", "result"])
    labs_df["caseid"] = labs_df["caseid"].astype(int)
    labs_df = labs_df[labs_df["name"].isin(set(LAB_ABG_NAME_TO_COL.keys()))]
    if labs_df.empty:
        return pd.DataFrame(columns=["caseid"])

    # Keep latest (largest dt) and near0 (smallest |dt|) result per case+item.
    latest = (
        labs_df.sort_values(["caseid", "name", "dt"])
        .groupby(["caseid", "name"], as_index=False)
        .tail(1)
        .rename(columns={"result": "latest_result", "dt": "latest_dt"})
    )
    near0 = (
        labs_df.assign(abs_dt=labs_df["dt"].abs())
        .sort_values(["caseid", "name", "abs_dt", "dt"])
        .groupby(["caseid", "name"], as_index=False)
        .head(1)
        .rename(columns={"result": "near0_result", "dt": "near0_dt"})
    )
    counts = (
        labs_df.groupby(["caseid", "name"], as_index=False)
        .size()
        .rename(columns={"size": "n_results"})
    )

    merged = latest[["caseid", "name", "latest_result"]].merge(
        near0[["caseid", "name", "near0_result"]],
        on=["caseid", "name"],
        how="outer",
    )
    merged = merged.merge(counts, on=["caseid", "name"], how="outer")

    rows = []
    for caseid, g in merged.groupby("caseid"):
        row = {"caseid": int(caseid)}
        for _, r in g.iterrows():
            name = str(r["name"])
            base = LAB_ABG_NAME_TO_COL.get(name)
            if not base:
                continue
            row[f"{base}_latest"] = r.get("latest_result")
            row[f"{base}_near0"] = r.get("near0_result")
            row[f"{base}_n"] = r.get("n_results")
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["caseid"])
    return out


def _enrich_abg_fields(df: pd.DataFrame, enrich_abg_from_labs_api: bool) -> pd.DataFrame:
    out = df.copy()

    # Build canonical ABG columns from base clinical columns first.
    for dst, srcs in BASE_ABG_FALLBACK.items():
        out = _select_first_existing_numeric(out, dst, srcs)

    if enrich_abg_from_labs_api:
        lab_summary = _build_abg_summary_from_labs()
        if not lab_summary.empty:
            out = out.merge(lab_summary, on="caseid", how="left")
            # Fill canonical columns from lab near0 values when missing.
            for lab_name, col_base in LAB_ABG_NAME_TO_COL.items():
                near0_col = f"{col_base}_near0"
                if near0_col in out.columns:
                    out[col_base] = pd.to_numeric(out[col_base], errors="coerce").where(
                        pd.to_numeric(out[col_base], errors="coerce").notna(),
                        pd.to_numeric(out[near0_col], errors="coerce"),
                    )

    abg_cols = ["abg_po2", "abg_pco2", "abg_ph", "abg_be", "abg_k", "abg_lac", "abg_sao2"]
    for col in abg_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out["has_abg_any"] = out[abg_cols].notna().any(axis=1).astype(int)
    return out


def build_multisource_csv(
    base_clinical_csv: Path,
    arr_metadata_csv: Path,
    output_csv: Path,
    arr_default_department: str,
    arr_default_opname: str,
    arr_default_ane_dur: float,
    keep_duplicate_caseids: bool,
    refresh_from_vitaldb_api: bool,
    enrich_hemo_track_flags: bool,
    enrich_abg_from_labs_api: bool,
    skip_arr_metadata: bool,
) -> Path:
    base_df = _normalize_caseid(_load_base_dataframe(base_clinical_csv, refresh_from_vitaldb_api))
    arr_df: Optional[pd.DataFrame] = None
    if (not skip_arr_metadata) and arr_metadata_csv.exists():
        arr_df = _normalize_caseid(_load_csv(arr_metadata_csv))
    elif not skip_arr_metadata:
        print(f"[warn] arr metadata csv not found, skip arr merge: {arr_metadata_csv}")

    base_df = _fill_if_missing(base_df, "department", "Unknown")
    base_df = _fill_if_missing(base_df, "opname", "Unknown surgery")
    base_df["source_dataset"] = "vitaldb_clinical"

    if arr_df is not None:
        arr_df = _fill_if_missing(arr_df, "department", arr_default_department)
        arr_df = _fill_if_missing(arr_df, "opname", arr_default_opname)
        if "ane_dur" not in arr_df.columns:
            arr_df["ane_dur"] = arr_default_ane_dur
        else:
            arr_df["ane_dur"] = arr_df["ane_dur"].apply(lambda x: _safe_float(x, arr_default_ane_dur))
        arr_df["source_dataset"] = "vitaldb_arrhythmia"

    if arr_df is not None:
        merged = pd.concat([base_df, arr_df], axis=0, ignore_index=True, sort=False)
    else:
        merged = base_df.reset_index(drop=True)
    if not keep_duplicate_caseids:
        merged = merged.drop_duplicates(subset=["caseid"], keep="first").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    merged = _enrich_hemo_flags(merged, enrich_hemo_track_flags)
    merged = _enrich_abg_fields(merged, enrich_abg_from_labs_api)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"[done] merged csv: {output_csv}")
    print(f"[done] rows={len(merged)} unique_caseids={merged['caseid'].nunique()}")
    src_counts = merged["source_dataset"].value_counts(dropna=False).to_dict()
    print(f"[done] source counts: {src_counts}")
    if "has_advanced_hemodynamics" in merged.columns:
        print(
            "[done] hemo availability:",
            {
                "co": int(merged.get("has_co_track", pd.Series(dtype=int)).sum()),
                "ci": int(merged.get("has_ci_track", pd.Series(dtype=int)).sum()),
                "sv": int(merged.get("has_sv_track", pd.Series(dtype=int)).sum()),
                "svr": int(merged.get("has_svr_track", pd.Series(dtype=int)).sum()),
                "svv": int(merged.get("has_svv_track", pd.Series(dtype=int)).sum()),
            },
        )
    if "has_abg_any" in merged.columns:
        print(f"[done] ABG any rows={int(merged['has_abg_any'].sum())}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge clinical_information.csv and VitalDB-arrhythmia metadata into one benchmark input CSV."
    )
    parser.add_argument("--base-clinical-csv", default="clinical_information.csv")
    parser.add_argument(
        "--arr-metadata-csv",
        default="downloaded_results/vitaldb-arrhythmia-1.0.0/metadata.csv",
    )
    parser.add_argument("--output-csv", default="downloaded_results/clinical_information_multisource.csv")
    parser.add_argument("--arr-default-department", default="Arrhythmia_DB")
    parser.add_argument("--arr-default-opname", default="Arrhythmia_Annotated_Case")
    parser.add_argument("--arr-default-ane-dur", type=float, default=999.0)
    parser.add_argument(
        "--keep-duplicate-caseids",
        action="store_true",
        help="Keep both source rows even when caseid overlaps between clinical and arrhythmia metadata.",
    )
    parser.add_argument(
        "--skip-arr-metadata",
        action="store_true",
        help="Do not merge VitalDB-arrhythmia metadata even when --arr-metadata-csv exists.",
    )
    parser.add_argument(
        "--refresh-from-vitaldb-api",
        action="store_true",
        help="Download base clinical table from VitalDB public API (https://api.vitaldb.net/cases).",
    )
    parser.add_argument(
        "--enrich-hemo-track-flags",
        action="store_true",
        help="Add CO/CI/SV/SVR/SVV track availability columns via vitaldb.find_cases.",
    )
    parser.add_argument(
        "--enrich-abg-from-labs-api",
        action="store_true",
        help="Download VitalDB labs table and add ABG summary columns (near0/latest/count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_multisource_csv(
        base_clinical_csv=Path(args.base_clinical_csv),
        arr_metadata_csv=Path(args.arr_metadata_csv),
        output_csv=Path(args.output_csv),
        arr_default_department=args.arr_default_department,
        arr_default_opname=args.arr_default_opname,
        arr_default_ane_dur=args.arr_default_ane_dur,
        keep_duplicate_caseids=args.keep_duplicate_caseids,
        refresh_from_vitaldb_api=args.refresh_from_vitaldb_api,
        enrich_hemo_track_flags=args.enrich_hemo_track_flags,
        enrich_abg_from_labs_api=args.enrich_abg_from_labs_api,
        skip_arr_metadata=args.skip_arr_metadata,
    )


if __name__ == "__main__":
    main()
