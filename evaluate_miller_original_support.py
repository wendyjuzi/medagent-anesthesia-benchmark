import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


MODE_KEYS = ["gpt_rag_miller10", "gpt_direct_no_rag"]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _normalize(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("\r", "\n")
    return value


def _extract_miller_from_final(final_output: str) -> str:
    text = final_output or ""
    m = re.search(r"【决策干预（Miller）】[:：]\s*(.*?)(?:\n【|$)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    return " ".join(m.group(1).split())


def _extract_output_text(mode_obj: Dict[str, Any]) -> str:
    out = str(mode_obj.get("miller_output") or "").strip()
    if out:
        return out
    final_output = str(mode_obj.get("final_output") or "").strip()
    if not final_output:
        return ""
    return _extract_miller_from_final(final_output)


def _concat_retrieval_texts(row: Dict[str, Any]) -> str:
    retrieval = row.get("miller_retrieval")
    if not isinstance(retrieval, dict):
        return ""
    results = retrieval.get("results")
    if not isinstance(results, list):
        return ""
    chunks: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        txt = str(item.get("text") or "").strip()
        if txt:
            chunks.append(txt)
    return "\n".join(chunks)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _map_lt_65_detect(text: str) -> bool:
    if re.search(r"map\s*[<≤]\s*65", text):
        return True
    if re.search(r"map[^0-9]{0,8}65", text):
        return True
    if "低血压" in text and "65" in text:
        return True
    return False


def _concept_presence(output_text: str, evidence_text: str) -> Tuple[Set[str], Set[str]]:
    out = _normalize(output_text)
    ev = _normalize(evidence_text)

    claimed: Set[str] = set()
    supported: Set[str] = set()

    concept_rules = {
        "perfusion_first": {
            "out": ["灌注优先", "先稳灌注", "先纠正灌注", "先纠正低血压", "perfusion first", "stabilize perfusion"],
            "ev": ["hypotension", "perfusion", "blood pressure", "hemodynamic", "vasopressor", "mean arterial pressure"],
        },
        "avoid_deepen_when_hypotension": {
            "out": ["避免加深", "不宜加深", "不要加深", "avoid deepening", "do not deepen"],
            "ev": ["hypotension", "map", "hemodynamic", "anesthetic depth", "depth of anesthesia"],
        },
        "bis_with_context": {
            "out": ["b i s", "bis", "结合刺激", "结合血流动力学", "不是单独触发", "not standalone trigger"],
            "ev": ["bis", "eeg", "stimulation", "artifact", "depth-of-anesthesia", "awareness"],
        },
        "opioid_titration": {
            "out": ["阿片", "瑞芬太尼", "opioid", "remifentanil", "镇痛滴定", "analgesia"],
            "ev": ["opioid", "fentanyl", "remifentanil", "analges", "noxious stimulation"],
        },
        "map65_threshold": {
            "out": ["map<65", "map <65", "map< 65", "map < 65", "map低于65", "map<65mmhg"],
            "ev": ["map", "hypotension", "mean arterial pressure", "blood pressure"],
        },
        "vasopressor_logic": {
            "out": ["去氧肾上腺素", "麻黄碱", "去甲肾上腺素", "phenylephrine", "ephedrine", "norepinephrine"],
            "ev": ["phenylephrine", "ephedrine", "norepinephrine", "vasopressor", "hypotension"],
        },
    }

    for concept, cfg in concept_rules.items():
        out_hit = _contains_any(out, cfg["out"])
        ev_hit = _contains_any(ev, cfg["ev"])
        if concept == "map65_threshold":
            out_hit = out_hit or _map_lt_65_detect(out)
        if out_hit:
            claimed.add(concept)
            if ev_hit:
                supported.add(concept)
    return claimed, supported


def _anchor_overlap(output_text: str, evidence_text: str) -> int:
    out = _normalize(output_text)
    ev = _normalize(evidence_text)
    anchors = [
        ("map", ["map", "平均动脉压", "灌注压"]),
        ("bis", ["bis", "脑电", "麻醉深度"]),
        ("hypotension", ["hypotension", "低血压"]),
        ("perfusion", ["perfusion", "灌注"]),
        ("opioid", ["opioid", "阿片", "瑞芬太尼", "remifentanil"]),
        ("propofol", ["propofol", "丙泊酚"]),
        ("vasopressor", ["vasopressor", "升压药", "去甲肾上腺素", "麻黄碱", "去氧肾上腺素"]),
    ]
    hit = 0
    for _, kws in anchors:
        out_hit = any(k in out for k in kws)
        ev_hit = any(k in ev for k in kws)
        if out_hit and ev_hit:
            hit += 1
    return hit


def _verdict_from_support(
    claimed: Set[str],
    supported: Set[str],
    anchor_hits: int,
    has_evidence: bool,
) -> Tuple[str, int, float]:
    if not has_evidence:
        return "no_evidence", -1, 0.0
    if not claimed:
        return "no_explicit_claim", -1, 0.0
    ratio = len(supported) / max(1, len(claimed))
    if ratio >= 1.0 and anchor_hits >= 1:
        return "supported_strict", 3, ratio
    if ratio >= 0.5:
        return "partially_supported", 2, ratio
    if ratio > 0:
        return "weakly_supported", 1, ratio
    return "unsupported", 0, 0.0


def _evaluate_mode(row: Dict[str, Any], mode_key: str, evidence_text: str) -> Dict[str, Any]:
    mode_obj = row.get(mode_key)
    if not isinstance(mode_obj, dict):
        return {
            "mode": mode_key,
            "verdict": "missing_mode_output",
            "score": -1,
            "support_ratio": 0.0,
            "claimed_concepts": [],
            "supported_concepts": [],
            "anchor_hits": 0,
            "miller_output": "",
        }
    output_text = _extract_output_text(mode_obj)
    claimed, supported = _concept_presence(output_text, evidence_text)
    anchor_hits = _anchor_overlap(output_text, evidence_text)
    verdict, score, ratio = _verdict_from_support(claimed, supported, anchor_hits, bool(evidence_text.strip()))
    return {
        "mode": mode_key,
        "verdict": verdict,
        "score": score,
        "support_ratio": round(ratio, 4),
        "claimed_concepts": sorted(list(claimed)),
        "supported_concepts": sorted(list(supported)),
        "anchor_hits": anchor_hits,
        "miller_output": output_text,
    }


def _winner_by_eval(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    if int(a.get("score", -1)) > int(b.get("score", -1)):
        return "rag"
    if int(a.get("score", -1)) < int(b.get("score", -1)):
        return "direct"
    return "tie"


def _build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total": len(rows)}
    for mode in MODE_KEYS:
        verdict_counts: Dict[str, int] = {}
        avg_ratio = 0.0
        valid_ratio_n = 0
        for row in rows:
            info = row.get(mode, {})
            verdict = str(info.get("verdict", "unknown")).strip().lower()
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            ratio = float(info.get("support_ratio", 0.0))
            if ratio > 0:
                avg_ratio += ratio
                valid_ratio_n += 1
        summary[f"{mode}_verdict_counts"] = verdict_counts
        summary[f"{mode}_avg_support_ratio_nonzero"] = round(avg_ratio / valid_ratio_n, 4) if valid_ratio_n else 0.0

    summary["rag_win_by_strict_support"] = sum(1 for row in rows if row.get("winner_by_strict_support") == "rag")
    summary["direct_win_by_strict_support"] = sum(1 for row in rows if row.get("winner_by_strict_support") == "direct")
    summary["tie_by_strict_support"] = sum(1 for row in rows if row.get("winner_by_strict_support") == "tie")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly evaluate whether generated Miller decisions are supported by retrieved Miller excerpts."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from compare_gpt_rag_vs_direct.py")
    parser.add_argument("--output-prefix", default="", help="Output prefix path (without extension)")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _load_jsonl(input_path)
    if not rows:
        raise ValueError("No rows loaded from input.")

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = input_path.with_suffix("")

    out_jsonl = prefix.parent / f"{prefix.name}.strict_support_eval.jsonl"
    out_csv = prefix.parent / f"{prefix.name}.strict_support_eval.csv"
    out_summary = prefix.parent / f"{prefix.name}.strict_support_eval.summary.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    eval_rows: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            evidence_text = _concat_retrieval_texts(row)
            rag_eval = _evaluate_mode(row, "gpt_rag_miller10", evidence_text)
            direct_eval = _evaluate_mode(row, "gpt_direct_no_rag", evidence_text)
            winner = _winner_by_eval(rag_eval, direct_eval)
            out_row = {
                "index": row.get("index"),
                "caseid": row.get("caseid"),
                "gpt_rag_miller10": rag_eval,
                "gpt_direct_no_rag": direct_eval,
                "winner_by_strict_support": winner,
            }
            eval_rows.append(out_row)
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    csv_fields = [
        "index",
        "caseid",
        "rag_verdict",
        "rag_score",
        "rag_support_ratio",
        "rag_claimed",
        "rag_supported",
        "direct_verdict",
        "direct_score",
        "direct_support_ratio",
        "direct_claimed",
        "direct_supported",
        "winner_by_strict_support",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in eval_rows:
            rag = row["gpt_rag_miller10"]
            direct = row["gpt_direct_no_rag"]
            writer.writerow(
                {
                    "index": row.get("index"),
                    "caseid": row.get("caseid"),
                    "rag_verdict": rag.get("verdict"),
                    "rag_score": rag.get("score"),
                    "rag_support_ratio": rag.get("support_ratio"),
                    "rag_claimed": ";".join(rag.get("claimed_concepts", [])),
                    "rag_supported": ";".join(rag.get("supported_concepts", [])),
                    "direct_verdict": direct.get("verdict"),
                    "direct_score": direct.get("score"),
                    "direct_support_ratio": direct.get("support_ratio"),
                    "direct_claimed": ";".join(direct.get("claimed_concepts", [])),
                    "direct_supported": ";".join(direct.get("supported_concepts", [])),
                    "winner_by_strict_support": row.get("winner_by_strict_support"),
                }
            )

    summary = _build_summary(eval_rows)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input:   {input_path}")
    print(f"rows:    {len(rows)}")
    print(f"jsonl:   {out_jsonl}")
    print(f"csv:     {out_csv}")
    print(f"summary: {out_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
