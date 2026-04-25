# Replication Package — "Flow Graph Based Code Generation"

SOEN 691 — Topics in Software Engineering  
Concordia University, 2026

---

## Overview

This package contains the code, raw results, and figures for **RQ1** and **RQ3** of the paper. The study evaluates whether a two-stage LLM pipeline — code → Mermaid flowchart → regenerated code — produces more correct and structurally faithful implementations than direct prompting from docstrings alone.

**Benchmark:** HumanEval (164 tasks)  
**Models evaluated:** GPT-OSS 120B, Llama 3 70B, Llama 4 Scout, Qwen3 32B  
**API provider:** [Groq](https://groq.com)

---

## Contents

```
replication_package/
├── humaneval_roundtrip_fix.py        # Main pipeline script
├── plots_multimodel.py               # Figure and table generator
├── analysis.py                       # Quick aggregated analysis
├── requirements.txt                  # Python dependencies
│
├── results_multimodel/
│   ├── results_gpt-oss-120b.jsonl    # Raw results — GPT-OSS 120B (492 records)
│   ├── results_llama3-70b.jsonl      # Raw results — Llama 3 70B (492 records)
│   ├── results_llama4-scout.jsonl    # Raw results — Llama 4 Scout (492 records)
│   └── results_qwen3-32b.jsonl       # Raw results — Qwen3 32B (492 records)
│
└── figures_multimodel/
    ├── fig_pass_at_1.pdf/png         # RQ1 — Pass@1 grouped bar chart
    ├── fig_codebleu.pdf/png          # RQ1 — CodeBLEU grouped bar chart
    ├── fig_tokens.pdf/png            # RQ3 — Token overhead per model
    ├── table_multimodel.tex          # LaTeX results table (booktabs)
    ├── table_multimodel.pdf/png      # Visual preview of the table
    └── summary_multimodel.csv        # Aggregated numbers per model × approach
```

Each `.jsonl` file contains 492 records = 164 tasks × 3 approaches (`direct`, `mermaid`, `identity`).

---

## How to Use This Package

There are two main scripts:

**`humaneval_roundtrip_fix.py`** — the experiment pipeline. It downloads HumanEval from Hugging Face, runs each task through one or more approaches (direct prompting, Mermaid round-trip, or identity), and writes one JSON record per task per approach to a `.jsonl` file. Requires a Groq API key.

**`plots_multimodel.py`** — the analysis and plotting script. It reads the `.jsonl` result files, computes aggregated metrics and statistical tests, and writes all figures, the LaTeX table, and the summary CSV to an output directory. Does **not** require an API key.

If you only want to regenerate the figures from our pre-computed results, skip directly to [Step 5](#step-5-regenerate-figures-and-tables) below. Steps 1–4 are only needed if you want to re-run the experiments from scratch.

---

## Step-by-Step: Reproducing Our Results

### Step 1 — Set up the environment

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Set your Groq API key

Create a free API key at [console.groq.com](https://console.groq.com), then export it:

```bash
export GROQ_API_KEY=your_key_here
```

> **Note:** Steps 3 and 4 make live API calls across 164 tasks × 3 approaches × 4 models = 1,968 total requests. Estimated cost: ~$5–15 per model (prices vary). Pre-computed results are already included in `results_multimodel/` — you can skip to Step 5 if you do not want to re-run.

### Step 3 — Run the experiment for each model

Run the pipeline once per model. Each command produces a `.jsonl` file with 492 records (164 tasks × 3 approaches). The pipeline appends to the output file, so **make sure the output file does not already exist** (or delete it first) before running.

```bash
# GPT-OSS 120B
python humaneval_roundtrip_fix.py \
    --model gpt-oss-120b \
    --n 164 \
    --approach all \
    --output results_multimodel/results_gpt-oss-120b.jsonl

# Llama 3 70B
python humaneval_roundtrip_fix.py \
    --model llama3-70b \
    --n 164 \
    --approach all \
    --output results_multimodel/results_llama3-70b.jsonl

# Llama 4 Scout
python humaneval_roundtrip_fix.py \
    --model llama4-scout \
    --n 164 \
    --approach all \
    --output results_multimodel/results_llama4-scout.jsonl

# Qwen3 32B
python humaneval_roundtrip_fix.py \
    --model qwen3-32b \
    --n 164 \
    --approach all \
    --output results_multimodel/results_qwen3-32b.jsonl
```

Each run prints a live progress table and a summary report when complete.

**Experimental settings used in the paper:**
- Temperature: `0.1` (near-deterministic; hardcoded default in `call_llm`)
- Max retries on API failure: 3 (exponential backoff)
- Test timeout per task: 10 seconds
- `<think>` blocks from Qwen3 32B are stripped automatically before evaluation

### Step 4 — Verify record counts

Each output file should contain exactly 492 lines:

```bash
wc -l results_multimodel/results_*.jsonl
```

Expected output:
```
492 results_multimodel/results_gpt-oss-120b.jsonl
492 results_multimodel/results_llama3-70b.jsonl
492 results_multimodel/results_llama4-scout.jsonl
492 results_multimodel/results_qwen3-32b.jsonl
```

### Step 5 — Regenerate figures and tables {#step-5-regenerate-figures-and-tables}

This step requires no API key. It reads all four `.jsonl` files and writes every figure, the LaTeX table, and the summary CSV:

```bash
python plots_multimodel.py --out figures_multimodel/
```

Outputs written to `figures_multimodel/`:

| File | Used in |
|------|---------|
| `fig_pass_at_1.pdf` / `.png` | Section IV-A (RQ1 correctness) |
| `fig_codebleu.pdf` / `.png` | Section IV-A (RQ1 structural fidelity) |
| `fig_tokens.pdf` / `.png` | Section IV-C (RQ3 overhead) |
| `table_multimodel.tex` | Paste directly into Overleaf |
| `table_multimodel.pdf` / `.png` | Visual preview |
| `summary_multimodel.csv` | Raw aggregated numbers |

### Step 6 — (Optional) Quick summary printout

To print a per-approach summary from any single model result file:

```bash
python analysis.py
```

By default this reads `results_multimodel/results_gpt-oss-120b.jsonl`. Edit the path at the top of the file to switch models.

---

## A Note on Reproducibility

LLM outputs are stochastic even at low temperature. Re-running the pipeline will produce slightly different generated code on each run, which may shift individual task pass/fail outcomes. The **aggregate metrics** (Pass@1, mean CodeBLEU, token counts) should be close to our reported numbers but will not match exactly. The pre-computed `.jsonl` files in `results_multimodel/` reflect the exact outputs used in the paper.

---

## Additional Notes

**Dataset:** HumanEval is downloaded automatically from Hugging Face (`openai/openai_humaneval`) on first run. No manual download needed.

**Qwen3 32B:** This model emits chain-of-thought `<think>` blocks. The pipeline strips them automatically before evaluation.

**Identity baseline:** The `identity` approach passes the canonical solution through unchanged (164/164 pass). It validates the test harness and is excluded from the main results table.

**Security:** The test runner uses `exec()` to evaluate LLM-generated code. When re-running experiments, do so inside a sandboxed environment (VM or container).

**LaTeX preamble:** `table_multimodel.tex` requires the following packages:
```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
```
