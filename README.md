# Anes VitalDB QA Pipeline

Build a training dataset for anesthesia decision reasoning:
- Stage 1: clinical filtering + surgery grouping + remove cases with no medication data
- Stage 2: detect intervention anchors from medication tracks + extract pre-anchor vital windows
- Stage 3: transform structured snapshot JSON into AnesSuite-style QA with LLM

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Input

Place `clinical_information.csv` in project root, or provide a custom path:

```bash
python anes_pipeline.py --clinical-csv /path/to/clinical_information.csv
```

If you want Miller retrieval, first convert the licensed PDF into a passage JSONL:

```bash
python prepare_miller_corpus.py ^
  --input-pdf "C:\path\to\Miller's Anesthesia.pdf" ^
  --output-jsonl "Anes_Dataset\datasets\miller_corpus.jsonl"
```

## 3) Run

### A) Preprocessing only (no LLM)

```bash
python anes_pipeline.py ^
  --clinical-csv clinical_information.csv ^
  --max-cases 0 ^
  --window-sec 300 ^
  --overwrite-jsonl
```

`--max-cases 0` means process all cases (e.g., 6,388).

### B) Full pipeline with vLLM (default settings in code)

No OpenAI API key is required if your local vLLM server is running at `http://127.0.0.1:8000/v1`.

```bash
python anes_pipeline.py ^
  --clinical-csv clinical_information.csv ^
  --max-cases 0 ^
  --window-sec 300 ^
  --enable-llm ^
  --overwrite-jsonl
```

The code default now is:
- `--llm-model Qwen/Qwen2.5-14B-Instruct`
- `--llm-base-url http://127.0.0.1:8000/v1`
- `--llm-api-key local`

### C) Full pipeline with local deployed model (no OpenAI API required)

#### Option 1: vLLM (OpenAI-compatible API)

Start server:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --port 8000
```

Run pipeline:

```bash
python anes_pipeline.py ^
  --clinical-csv clinical_information.csv ^
  --max-cases 0 ^
  --window-sec 300 ^
  --enable-llm ^
  --llm-base-url http://127.0.0.1:8000/v1 ^
  --llm-model Qwen/Qwen2.5-14B-Instruct ^
  --llm-api-key local ^
  --overwrite-jsonl
```

#### Option 2: Ollama (OpenAI-compatible API)

Start and pull model:

```bash
ollama serve
ollama pull qwen2.5:14b
```

Run pipeline:

```bash
python anes_pipeline.py ^
  --clinical-csv clinical_information.csv ^
  --max-cases 0 ^
  --window-sec 300 ^
  --enable-llm ^
  --llm-base-url http://127.0.0.1:11434/v1 ^
  --llm-model qwen2.5:14b ^
  --llm-api-key ollama ^
  --overwrite-jsonl
```

### D) Full pipeline with Miller embedding retrieval

Provide a licensed Miller corpus first. Supported formats:
- `.txt` / `.md`: one full text file, chunked automatically
- `.pdf`: extracted page-by-page with `pypdf`, then chunked automatically
- `.jsonl`: one passage per line with `text`, `content`, `passage`, or `chunk`

```bash
python anes_pipeline.py ^
  --clinical-csv clinical_information.csv ^
  --max-cases 0 ^
  --window-sec 300 ^
  --enable-llm ^
  --enable-miller-rag ^
  --miller-corpus-path Anes_Dataset\datasets\miller_corpus.jsonl ^
  --miller-index-path Anes_Dataset\datasets\miller_index.npz ^
  --embedding-model text-embedding-3-small ^
  --miller-top-k 3 ^
  --overwrite-jsonl
```

If the LLM endpoint does not provide embeddings, point retrieval to a separate embedding endpoint:

```bash
python anes_pipeline.py ^
  --enable-llm ^
  --enable-miller-rag ^
  --miller-corpus-path C:\path\to\miller_corpus.txt ^
  --embedding-base-url https://api.openai.com/v1 ^
  --embedding-api-key-env OPENAI_API_KEY
```

If you prefer local CPU embeddings, use a local `sentence-transformers` model path:

```bash
python anes_pipeline.py ^
  --enable-llm ^
  --enable-miller-rag ^
  --miller-corpus-path Anes_Dataset\datasets\miller_corpus.jsonl ^
  --embedding-backend local ^
  --embedding-model C:\models\bge-large-en-v1.5 ^
  --embedding-device cpu
```

## 4) Output

- `Anes_Dataset/Data/<Surgery_Group>/caseids.csv`
- `Anes_Dataset/Data/<Surgery_Group>/clinical_subset.csv`
- `Anes_Dataset/images/*.png`
- `Anes_Dataset/datasets/snapshots.json`
- `Anes_Dataset/datasets/llm_outputs.jsonl` (only if `--enable-llm`)
- `Anes_Dataset/datasets/anes_qa_dataset.jsonl`
- `Anes_Dataset/datasets/vitaldb_miller_alignment_report.json` (VitalDB vs Miller consistency summary)
- `Anes_Dataset/datasets/miller_index.npz` (optional cached embedding index)

## 5) Important arguments

- `--window-sec`: 300 for 5 min, 600 for 10 min
- `--case-fetch-timeout-sec`: per-case vital fetch hard timeout in stage2 (default `12`)
- `--case-fetch-retries`: retries when case fetch timeout/fails (default `1`)
- `--case-fetch-backoff-sec`: linear backoff seconds between retries (default `1.0`)
- `--rate-delta-threshold`: intervention threshold for rate tracks (default `0.5`)
- `--vol-delta-threshold`: intervention threshold for cumulative volume tracks (default `0.03`)
- `--vol-rate-lookback-sec`: local lookback window to smooth cumulative-volume inferred infusion rate (default `60`)
- `--propofol-bolus-rate-threshold-ml-h`: if smoothed Propofol rate exceeds this threshold, classify as bolus-like event (default `50`)
- `--propofol-bolus-min-delta-ml`: minimum smoothed window volume delta for bolus-like classification (default `1.0`)
- `--anchor-selection-mode`: `diverse` (default) to diversify medication anchors, or `time` for purely chronological anchors
- `--anchor-class-priority`: targeted anchor class priority, default `vasoactive,analgesic,neuromuscular,sedative,other`
- `--disable-decision-anchor-filter`: keep all threshold-triggered anchors (default is filtered decision anchors only)
- `--vol-decision-min-delta-ml`: minimum smoothed volume delta to keep non-bolus `*_VOL` anchors (default `0.08`)
- `--vol-decision-min-smoothed-rate-ml-h`: minimum smoothed rate to keep non-bolus `*_VOL` anchors (default `4.0`)
- `--vol-decision-min-rate-change-ml-h`: minimum level-shift against previous smoothed rate for `*_VOL` anchors (default `2.0`)
- `--vol-start-min-after-ml`: minimum `after` value for `infusion_start` on `*_VOL` tracks (default `0.05`)
- `--require-mbp-for-qa`: drop snapshots with low MBP coverage (recommended for stronger hemodynamic reasoning)
- `--min-mbp-valid-coverage-ratio`: MBP minimum valid ratio when MBP is required (default `0.10`)
- `--require-bis-for-qa`: drop snapshots with low BIS coverage
- `--min-bis-valid-coverage-ratio`: BIS minimum valid ratio when BIS is required (default `0.10`)
- `--bis-invalid-floor`: BIS values <= this threshold are treated as invalid sensor readings (default `1.0`)
- `--max-bis-zero-ratio`: if BIS invalid-zero ratio exceeds this threshold, mark snapshot as sensor-outlier (default `0.30`)
- `--drop-snapshot-if-outlier`: drop snapshots with severe sensor outlier signatures
- `--min-non-propofol-records`: for demo generators, continue scanning until at least N non-propofol anchors are collected (default `2`)
- `--label-policy`: single supervision target policy:
  - `qa_normative` (default): train on LLM-generated QA target
  - `doctor_action`: train as behavior-cloning on doctor actual intervention only
  - when `doctor_action` is selected, stage-3 LLM generation is skipped automatically
- `--max-anchors-per-case`: keep N anchors per case (default `3`)
- `--min-anchor-gap-sec`: dedup nearby anchor events (default `30`)
- `--skip-medication-filter`: skip stage-1 medication validity filtering
- `--llm-base-url`: local OpenAI-compatible endpoint (vLLM/Ollama)
- `--llm-api-key`: explicit key/token, for local service can be any non-empty string
- `--llm-max-workers`: parallel LLM workers (default `1`, set >1 for concurrent generation)
- `--enable-miller-rag`: retrieve Miller evidence before prompt construction
- `--miller-corpus-path`: local licensed Miller corpus path
- `--miller-index-path`: optional cached embedding index path
- `--embedding-backend`: `auto`, `api`, or `local`
- `--embedding-model`: embedding model for Miller retrieval
- `--embedding-device`: device for local embedding backend, e.g. `cpu`
- `--embedding-base-url`: optional embedding endpoint; empty means reuse `--llm-base-url`
- `--miller-top-k`: number of Miller passages injected into prompt (clamped to `1-5`)
- `--llm-max-retries`: per-request retries with exponential backoff (default `3`)
- `--llm-request-timeout-sec`: hard timeout per LLM API call (default `120`)
- `--llm-max-tokens`: cap output length to prevent very long generations (default `700`)
- `--llm-progress-every`: print Stage-3 progress every N records (default `1`)
- `--disable-mbp-unit-fix`: disable automatic MBP kPa->mmHg conversion
- `--mbp-art-min-coverage-ratio`: if ART+NIBP coexist, fallback to NIBP only when ART coverage is too sparse
- `--disable-artifact-filter`: disable vital artifact filtering before trend/assessment
- `--artifact-median-window`: median filter window size for artifact suppression
- `--auto-clean-after-generation`: run QA cleaner automatically right after JSONL generation
- `--auto-clean-drop-invalid`: when auto-clean is on, drop records whose QA still fails strict format check
- `--auto-clean-field`: target field to clean (default `llm_output`)

## 6) Notes

- VitalDB track names can vary by dataset/hospital setup.  
  Update candidates in `MEDICATION_TRACK_CANDIDATES` and `VITAL_TRACK_CANDIDATES` if needed.
- Medication candidates currently cover:
  - `vasoactive`: Norepinephrine / Phenylephrine / Ephedrine / Epinephrine / Esmolol / Nicardipine / Nitroprusside
  - `vasodilator/inodilator`: Nitroglycerin / Milrinone
  - `chronotropic`: Atropine
  - `analgesic`: Remifentanil (`REMI_*` and `RFTN*` aliases)
  - `neuromuscular`: Rocuronium / Vecuronium
  - `sedative`: Propofol variants (`PPF*` volume/rate aliases)
  - `volatile`: Sevoflurane / Desflurane / Isoflurane ET/FI concentration and MAC anchors
  - anchor selection now prioritizes high-impact hemodynamic triggers: `NOR_RATE` > `PHE_RATE` > `NICA_RATE` > `ESMO_RATE`
- For vLLM, prefer `--llm-model` as served model name (e.g. `Qwen3.5-35B-A3B-FP8`), not local filesystem path.
- Stage-3 now tries one-time auto-resolve from `/v1/models`:
  - exact model id match
  - basename fallback for path-like input
  - single-model server fallback
- MBP selection now prefers invasive ART when present and only falls back to NIBP if ART is too sparse in the same window.
- Snapshot now includes position context, inferred intraop phase (`Induction` / `Maintenance` / `Emergence`), and cumulative dose context for propofol concentration variants.
- Snapshot now includes structured `state` / `action` / `quality` blocks for train-ready supervision.
- `action` now keeps dual representation:
  - `action_raw` (device-side mL / mL/h)
  - `action_clinical` (converted clinical units when concentration/weight is available, plus `action_text`)
- Stage-2 now enforces hard removal for `quality.signal_quality == C` samples.
- Clinical risk features now include:
  - `double_low_state` (MAP < 75 and BIS < 45)
  - MBP hypotension burden `AUC<65` (mmHg*min) in trend/stat and observations
- Snapshot now includes `outcome_ground_truth.vital_trend_next_3min` (post-intervention future trend for evaluation/reward, not mandatory as prompt input).
- Anchor detection now:
  - applies tiny median smoothing on `*_RATE` tracks before diff-triggering (noise-robust)
  - de-duplicates within each medication key (not global), and keeps strongest event in each local time cluster
  - preserves simultaneous cross-drug interventions occurring in the same short window
- Stage-3 now prefers `qa_ready_snapshot` (reduced noisy fields) instead of dumping full engineering snapshot into the LLM prompt.
- Optional Miller RAG converts each VitalDB snapshot into a natural-language query, embeds it, retrieves top-k Miller passages, and injects those excerpts into the prompt as evidence context.
- Local Miller retrieval can run without an embedding API by loading a `sentence-transformers` model directly from a local path.
- For large-scale runs, start with preprocessing-only mode, verify snapshots, then enable LLM generation.
- QA answer format now supports dual intervention fields:
  - `【决策干预（Miller）】`: guideline-oriented recommendation
  - `【决策干预（VitalDB）】`: must align with logged action in dataset

Drug-specific VitalDB-vs-Miller audit checks include:
- phenylephrine escalation in severe bradycardia
- ephedrine escalation in tachycardia
- nitroglycerin / milrinone escalation while MAP < 65
- atropine use while already tachycardic
- propofol/remifentanil escalation in brady-hypotension states

## 7) Post-cleaning existing JSONL

If existing outputs contain `<think>` or draft CoT leakage, run:

```bash
python clean_qa_jsonl.py --input Anes_Dataset_5Demo_Local/datasets/demo_5_records_local.jsonl
```

## 8) Separate Generation Modes

Generate with Miller embedding retrieval:

```bash
python generate_with_embedding_rag.py \
  --input Anes_Dataset_10Demo_MillerRAG/datasets/demo_10_records_local.jsonl \
  --output Anes_Dataset_10Demo_MillerRAG/datasets/demo_10_embedding_rag.jsonl \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen3.5-35B-A3B-FP8 \
  --llm-api-key local \
  --miller-corpus-path Anes_Dataset/datasets/miller_corpus.jsonl \
  --miller-index-path Anes_Dataset/datasets/miller_index_bge_large_cpu.npz \
  --embedding-backend local \
  --embedding-model /root/autodl-tmp/models/bge-large-en-v1.5 \
  --embedding-device cpu
```

Generate directly with GPT API without Miller retrieval:

```bash
python generate_with_gpt_api_direct.py \
  --input Anes_Dataset_10Demo_MillerRAG/datasets/demo_10_records_local.jsonl \
  --output Anes_Dataset_10Demo_MillerRAG/datasets/demo_10_gpt_direct.jsonl \
  --gpt-model gpt-4.1 \
  --gpt-api-key-env OPENAI_API_KEY
```

This writes `*.cleaned.jsonl` and keeps only cleaned QA text in `llm_output`.

## 8) Generate + auto-clean in one run

```bash
python generate_5_examples_local.py ^
  --clinical-csv clinical_information.csv ^
  --target-n 10 ^
  --min-non-propofol-records 2 ^
  --anchor-selection-mode diverse ^
  --anchor-class-priority vasoactive,analgesic,neuromuscular,sedative,other ^
  --llm-base-url http://127.0.0.1:8000/v1 ^
  --llm-model Qwen3.5-35B-A3B-FP8 ^
  --llm-api-key local ^
  --skip-medication-filter ^
  --require-mbp-for-qa ^
  --drop-snapshot-if-outlier ^
  --auto-clean-after-generation
```

To strictly keep only valid QA format lines:

```bash
python generate_5_examples_local.py ^
  --clinical-csv clinical_information.csv ^
  --target-n 10 ^
  --llm-base-url http://127.0.0.1:8000/v1 ^
  --llm-model Qwen3.5-35B-A3B-FP8 ^
  --llm-api-key local ^
  --skip-medication-filter ^
  --auto-clean-after-generation ^
  --auto-clean-drop-invalid
```

## 9) Export train-ready dataset (state-action-target)

```bash
python export_train_dataset.py ^
  --input Anes_Dataset_10Demo_Local/datasets/demo_10_records_local.cleaned.jsonl ^
  --mode qa_normative ^
  --drop-non-decision ^
  --drop-missing-qa ^
  --drop-low-confidence
```

## 10) Audit VitalDB Strategy Against Miller Policy

```bash
python audit_vitaldb_alignment.py ^
  --input downloaded_results/Dataset_Thoracic/datasets/anes_qa_dataset.jsonl
```
