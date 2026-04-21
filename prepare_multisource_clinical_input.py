import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


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


def build_multisource_csv(
    base_clinical_csv: Path,
    arr_metadata_csv: Path,
    output_csv: Path,
    arr_default_department: str,
    arr_default_opname: str,
    arr_default_ane_dur: float,
    keep_duplicate_caseids: bool,
) -> Path:
    base_df = _normalize_caseid(_load_csv(base_clinical_csv))
    arr_df = _normalize_caseid(_load_csv(arr_metadata_csv))

    base_df = _fill_if_missing(base_df, "department", "Unknown")
    base_df = _fill_if_missing(base_df, "opname", "Unknown surgery")
    base_df["source_dataset"] = "vitaldb_clinical"

    arr_df = _fill_if_missing(arr_df, "department", arr_default_department)
    arr_df = _fill_if_missing(arr_df, "opname", arr_default_opname)
    if "ane_dur" not in arr_df.columns:
        arr_df["ane_dur"] = arr_default_ane_dur
    else:
        arr_df["ane_dur"] = arr_df["ane_dur"].apply(lambda x: _safe_float(x, arr_default_ane_dur))
    arr_df["source_dataset"] = "vitaldb_arrhythmia"

    merged = pd.concat([base_df, arr_df], axis=0, ignore_index=True, sort=False)
    if not keep_duplicate_caseids:
        merged = merged.drop_duplicates(subset=["caseid"], keep="first").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"[done] merged csv: {output_csv}")
    print(f"[done] rows={len(merged)} unique_caseids={merged['caseid'].nunique()}")
    src_counts = merged["source_dataset"].value_counts(dropna=False).to_dict()
    print(f"[done] source counts: {src_counts}")
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
    )


if __name__ == "__main__":
    main()
