import argparse
import json
import os
import random
import shutil
from typing import Dict, List, Optional

from anes_pipeline import (
    NON_PROPOFOL_MED_KEYS,
    PipelineConfig,
    stage1_group_and_filter,
    stage2_extract_snapshots,
    stage3_generate_qa,
)

try:
    from anes_pipeline import clean_jsonl_file
except ImportError:
    # Backward-compatible fallback.
    from clean_qa_jsonl import clean_record

    def clean_jsonl_file(
        input_jsonl: str,
        field: str = "llm_output",
        drop_invalid: bool = False,
        output_jsonl: Optional[str] = None,
    ) -> str:
        input_path = input_jsonl
        if output_jsonl:
            output_path = output_jsonl
        else:
            root, ext = os.path.splitext(input_path)
            output_path = f"{root}.cleaned{ext if ext else '.jsonl'}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec, _, is_valid = clean_record(rec, field)
                if (not is_valid) and drop_invalid:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return output_path


def build_local_cfg(args: argparse.Namespace) -> PipelineConfig:
    output_dir = args.output_dir
    group_root = os.path.join(output_dir, "Data")
    image_root = os.path.join(output_dir, "images")
    dataset_root = os.path.join(output_dir, "datasets")
    os.makedirs(dataset_root, exist_ok=True)

    return PipelineConfig(
        clinical_csv=args.clinical_csv,
        output_dir=output_dir,
        group_root=group_root,
        image_root=image_root,
        dataset_jsonl=os.path.join(dataset_root, "anes_qa_dataset.jsonl"),
        snapshot_json=os.path.join(dataset_root, "snapshots.json"),
        llm_jsonl=os.path.join(dataset_root, "llm_outputs.jsonl"),
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
        enable_llm=True,
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
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key_env=args.embedding_api_key_env,
        embedding_api_key=args.embedding_api_key,
        overwrite_jsonl=True,
        sample_rate=0.2,
        random_seed=args.random_seed,
        export_bucketed_datasets=False,
        train_mix_a_ratio=0.8,
        train_mix_seed=args.random_seed,
        train_mix_max_samples=0,
        strict_a_requires_risk_flags=False,
        strict_a_requires_objective_evidence=False,
    )


def write_jsonl(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reset_image_root(image_root: str) -> None:
    if os.path.isdir(image_root):
        shutil.rmtree(image_root, ignore_errors=True)
    os.makedirs(image_root, exist_ok=True)


def count_non_propofol_records(records: List[dict]) -> int:
    n = 0
    for rec in records:
        med_key = str(rec.get("snapshot", {}).get("anchor_detail", {}).get("medication_key", "UNKNOWN"))
        if med_key in NON_PROPOFOL_MED_KEYS:
            n += 1
    return n


def select_diverse_records(records: List[dict], target_n: int, mode: str, seed: int) -> List[dict]:
    if len(records) <= target_n:
        return records
    if mode == "sequential":
        return records[:target_n]
    if mode == "random":
        pool = records[:]
        random.Random(seed).shuffle(pool)
        return pool[:target_n]

    by_case: Dict[int, List[dict]] = {}
    for rec in records:
        caseid = int(rec.get("caseid", -1))
        by_case.setdefault(caseid, []).append(rec)

    selected: List[dict] = []
    used_ids = set()
    for caseid in sorted(by_case.keys()):
        rec0 = by_case[caseid][0]
        selected.append(rec0)
        used_ids.add(id(rec0))
        if len(selected) >= target_n:
            return selected[:target_n]

    remaining = [r for r in records if id(r) not in used_ids]
    grouped: Dict[str, List[dict]] = {}
    for rec in remaining:
        med_key = str(rec.get("snapshot", {}).get("anchor_detail", {}).get("medication_key", "UNKNOWN"))
        grouped.setdefault(med_key, []).append(rec)

    keys = sorted(grouped.keys())
    ptr = {k: 0 for k in keys}
    while len(selected) < target_n:
        picked_any = False
        for k in keys:
            i = ptr[k]
            if i < len(grouped[k]):
                selected.append(grouped[k][i])
                ptr[k] = i + 1
                picked_any = True
                if len(selected) >= target_n:
                    break
        if not picked_any:
            break
    return selected[:target_n]


def enforce_non_propofol_quota(selected: List[dict], pool: List[dict], min_non_prop: int) -> List[dict]:
    if min_non_prop <= 0:
        return selected
    cur = count_non_propofol_records(selected)
    if cur >= min_non_prop:
        return selected

    candidates = [
        rec
        for rec in pool
        if str(rec.get("snapshot", {}).get("anchor_detail", {}).get("medication_key", "UNKNOWN")) in NON_PROPOFOL_MED_KEYS
        and rec not in selected
    ]
    if not candidates:
        return selected

    out = selected[:]
    i = 0
    while cur < min_non_prop and i < len(candidates):
        replace_idx = None
        for j in range(len(out) - 1, -1, -1):
            mk = str(out[j].get("snapshot", {}).get("anchor_detail", {}).get("medication_key", "UNKNOWN"))
            if mk not in NON_PROPOFOL_MED_KEYS:
                replace_idx = j
                break
        if replace_idx is None:
            break
        out[replace_idx] = candidates[i]
        i += 1
        cur = count_non_propofol_records(out)
    return out


def print_record_quality_summary(records: List[dict]) -> None:
    total = len(records)
    if total == 0:
        print("[summary] no records")
        return

    case_counts: Dict[int, int] = {}
    med_counts: Dict[str, int] = {}
    risk_counts: Dict[str, int] = {}
    init_setup_like = 0

    for rec in records:
        caseid = int(rec.get("caseid", -1))
        case_counts[caseid] = case_counts.get(caseid, 0) + 1
        snap = rec.get("snapshot", {})
        med_key = str(snap.get("anchor_detail", {}).get("medication_key", "UNKNOWN"))
        med_counts[med_key] = med_counts.get(med_key, 0) + 1
        risk = str(snap.get("clinical_assessment", {}).get("risk_level", "unknown"))
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        itype = str(snap.get("interpreted_intervention_type", ""))
        before = snap.get("anchor_detail", {}).get("before")
        after = snap.get("anchor_detail", {}).get("after")
        try:
            b = float(before) if before is not None else None
            a = float(after) if after is not None else None
        except (TypeError, ValueError):
            b, a = None, None
        if itype == "rate_adjustment" and b is not None and a is not None and abs(b) <= 1.0 and a >= 300.0:
            init_setup_like += 1

    print(
        f"[summary] records={total} unique_cases={len(case_counts)} "
        f"non_propofol={count_non_propofol_records(records)} init_setup_like={init_setup_like}"
    )
    print(f"[summary] medication_key_counts={med_counts}")
    print(f"[summary] risk_level_counts={risk_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-size demo QA records using local OpenAI-compatible LLM")
    parser.add_argument("--clinical-csv", default="clinical_information.csv")
    parser.add_argument("--output-dir", default="Anes_Dataset_10Demo_Local")
    parser.add_argument("--target-n", type=int, default=10)
    parser.add_argument("--min-non-propofol-records", type=int, default=2)
    parser.add_argument("--chunk-cases", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--anchor-selection-mode", default="diverse", choices=["diverse", "sequential", "random"])
    parser.add_argument("--anchor-class-priority", default="", help="Compatibility arg; reserved.")
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--signal-interval-sec", type=float, default=1.0)
    parser.add_argument("--med-check-interval-sec", type=float, default=3.0)
    parser.add_argument("--window-sec", type=int, default=300)
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
    parser.add_argument("--max-anchors-per-case", type=int, default=2)
    parser.add_argument("--skip-setup-rate-anchors", action="store_true")
    parser.add_argument("--setup-rate-before-abs-max", type=float, default=1.0)
    parser.add_argument("--setup-rate-after-threshold", type=float, default=300.0)
    parser.add_argument("--setup-rate-delta-threshold", type=float, default=100.0)
    parser.add_argument("--setup-rate-early-window-sec", type=float, default=1800.0)
    parser.add_argument("--skip-medication-filter", action="store_true")
    parser.add_argument("--keep-source-duplicate-rows", action="store_true")
    parser.add_argument("--department-include", default="")

    parser.add_argument("--anchor-mode", default="medication", choices=["medication", "arrdb", "hybrid", "periodic"])
    parser.add_argument("--arrdb-annotation-dir", default="downloaded_results/vitaldb-arrhythmia-1.0.0/Annotation_Files")
    parser.add_argument("--arrdb-time-column", default="")
    parser.add_argument("--arrdb-label-column", default="")
    parser.add_argument("--arrdb-keep-normal", action="store_true")
    parser.add_argument("--periodic-anchor-step-sec", type=float, default=300.0)
    parser.add_argument("--periodic-anchor-start-sec", type=float, default=300.0)

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--llm-progress-every", type=int, default=10)
    parser.add_argument("--validate-actual-before-qa", action="store_true")
    parser.add_argument("--drop-if-actual-invalid", action="store_true")
    parser.add_argument("--drop-if-actual-uncertain", action="store_true")
    parser.add_argument("--actual-validation-model", default="")
    parser.add_argument("--actual-validation-max-tokens", type=int, default=256)

    parser.add_argument("--enable-miller-rag", action="store_true")
    parser.add_argument("--miller-corpus-path", default="")
    parser.add_argument("--miller-index-path", default="")
    parser.add_argument("--miller-top-k", type=int, default=3)
    parser.add_argument("--miller-chunk-chars", type=int, default=1200)
    parser.add_argument("--miller-chunk-overlap-chars", type=int, default=200)
    parser.add_argument("--miller-max-passage-chars", type=int, default=800)
    parser.add_argument("--embedding-backend", default="auto", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-api-key", default="")

    parser.add_argument("--auto-clean-after-generation", action="store_true")
    parser.add_argument("--auto-clean-drop-invalid", action="store_true")
    parser.add_argument(
        "--auto-clean-drop-invalidtion",
        dest="auto_clean_drop_invalid",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--auto-clean-field", default="llm_output")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignored unknown args: {' '.join(unknown)}")

    cfg = build_local_cfg(args)
    reset_image_root(cfg.image_root)
    print("=== Generate Demo Records (Local LLM) ===")

    print("[1/4] Stage 1 filtering/grouping...")
    cases_df = stage1_group_and_filter(cfg)
    if cases_df.empty:
        print("No valid case after stage 1.")
        return

    print("[2/4] Stage 2 extracting snapshots in chunks...")
    records: List[dict] = []
    start = 0
    target_non_prop = args.min_non_propofol_records if args.anchor_mode in ("medication", "hybrid") else 0
    while (len(records) < args.target_n or count_non_propofol_records(records) < target_non_prop) and start < len(cases_df):
        chunk_df = cases_df.iloc[start : start + args.chunk_cases].copy()
        start += args.chunk_cases
        chunk_records = stage2_extract_snapshots(chunk_df, cfg)
        for rec in chunk_records:
            records.append(rec)
            if len(records) >= args.target_n and count_non_propofol_records(records) >= target_non_prop:
                break
        print(
            f"  collected {len(records)}/{args.target_n}, "
            f"non-prop={count_non_propofol_records(records)}/{target_non_prop}"
        )

    if not records:
        print("No anchor snapshots extracted.")
        return

    selected = select_diverse_records(records, args.target_n, args.anchor_selection_mode, args.random_seed)
    selected = enforce_non_propofol_quota(selected, records, target_non_prop)
    selected = selected[: args.target_n]
    print_record_quality_summary(selected)

    print("[3/4] Stage 3 generating QA by local LLM...")
    stage3_generate_qa(selected, cfg)

    out_file = os.path.join(cfg.output_dir, "datasets", f"demo_{args.target_n}_records_local.jsonl")
    write_jsonl(selected, out_file)
    if args.auto_clean_after_generation:
        clean_jsonl_file(
            input_jsonl=out_file,
            field=args.auto_clean_field,
            drop_invalid=args.auto_clean_drop_invalid,
        )
    print(f"[4/4] Saved {len(selected)} records -> {out_file}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
