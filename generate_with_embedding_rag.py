import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from anes_pipeline import (
    build_miller_retriever,
    create_embedding_client,
    create_openai_client,
    generate_single_qa,
    retrieve_miller_context,
)


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


def _build_cfg(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        enable_miller_rag=True,
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
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        api_key_env=args.api_key_env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate anesthesia QA with Miller embedding retrieval context."
    )
    parser.add_argument("--input", required=True, help="Input JSONL/JSON records containing `snapshot`.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to run; 0 means all.")
    parser.add_argument("--output-field", default="llm_output_embedding_rag")

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")

    parser.add_argument("--miller-corpus-path", required=True)
    parser.add_argument("--miller-index-path", required=True)
    parser.add_argument("--miller-top-k", type=int, default=3)
    parser.add_argument("--miller-chunk-chars", type=int, default=1200)
    parser.add_argument("--miller-chunk-overlap-chars", type=int, default=200)
    parser.add_argument("--miller-max-passage-chars", type=int, default=800)

    parser.add_argument("--embedding-backend", default="local", choices=["auto", "api", "local"])
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-api-key", default="")
    args = parser.parse_args()

    records = _load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No input records loaded.")

    cfg = _build_cfg(args)
    llm_client = create_openai_client(cfg)
    embed_client = create_embedding_client(cfg)
    retriever = build_miller_retriever(embed_client, cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            snapshot = _snapshot_from_record(record)
            retrieval = retrieve_miller_context(snapshot, retriever, embed_client, cfg)
            qa = generate_single_qa(llm_client, args.llm_model, snapshot, retrieval=retrieval)
            out = dict(record)
            out["generation_mode"] = "embedding_rag"
            out["miller_retrieval"] = retrieval
            out[args.output_field] = qa
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  - embedding RAG generated {idx}/{len(records)}")

    print(f"Done: wrote {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
