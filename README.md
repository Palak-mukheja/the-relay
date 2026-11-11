# The Relay — Research Paper Assistant and Reliability Experiments

The Relay is a retrieval-augmented assistant for answering engineers' questions from a fixed library of three research papers. The project combines page-aware document ingestion, BM25 retrieval, grounded answer generation with citations, answer validation/refusal, a Groq-backed model client, and a controllable OpenAI-compatible stand-in service for repeatable reliability experiments.

## Source library

The evaluation corpus is pinned to these versions:

1. *Attention Is All You Need* — arXiv:1706.03762v7
2. *Moshi* — arXiv:2410.00037v1
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* — arXiv:2507.07955v2

The question set contains 24 fixed questions: 9 about the Transformer paper, 6 about Moshi, 5 about Dynamic Chunking, 2 cross-paper questions, and 2 additional unanswerable questions. The set also includes a source-contradiction test and refusal tests. Printed page numbers are used in expected sources.

## Architecture

1. **Ingestion:** extract page text from the pinned PDFs and retain page metadata.
2. **Chunking and retrieval:** create overlapping chunks and retrieve candidate passages with BM25. Retrieval settings were compared using a fixed 24-question set.
3. **Answer generation:** build a question plus retrieved-context prompt and call either the real Groq API or the local stand-in.
4. **Grounding and validation:** check citation references and selected answer constraints; refuse when retrieval/context is insufficient or the question is unsupported.
5. **Gateway:** place requests behind bounded workers, a token budget, deadlines, retry limits, and retry/backoff rules.
6. **Observability:** log one row per request and retain scenario manifests and calibration evidence.

## Project layout

The local source project should contain the following folders/files (the evidence bundle here does not duplicate the Python source files or the original paper PDFs):

```text
src/                 application, RAG, stand-in, gateway, logging
probes/              real-provider probes, tests, scenario runner, checker
data/papers/         the three pinned PDF versions
data/                evaluation_questions.json and extracted text
results.csv          combined request log (real + stand-in)
results/             scenario summaries and evaluation summaries
calibration.json     real Groq calibration evidence
DECISIONS.md         project decisions and experiment interpretation
README.md            project setup and reproduction guide
```

## Setup

Use Python 3.11+ and a virtual environment. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies pinned by the project (keep a `requirements.txt` in the repository so the environment can be recreated):

```powershell
python -m pip install -r requirements.txt
```

For live Groq requests, provide an API key through an environment variable. Do not commit the real `.env` file or key:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

An `.env.example` is included as a placeholder only.

## Running the local stand-in

Start the stand-in server in one terminal:

```powershell
uvicorn src.standin:app --host 127.0.0.1 --port 8000
```

Check its health endpoint in another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The stand-in exposes administrative state/configuration endpoints and an OpenAI-compatible chat-completions endpoint. It can simulate calibrated latency, slow responses, failures, RPM/TPM constraints, daily-budget exhaustion, and streaming. Its response content is deliberately synthetic; it is not a substitute for measuring answer quality with the real model.

The repository also contains smoke-test entry points used during development:

```powershell
python -m src.run_standin
python -m src.run_gateway_standin
```

If the stand-in is already running, reuse that server rather than launching a second process on the same port.

## Running experiments

Preview one scenario before sending requests:

```powershell
python -m probes.run_scenarios --plan --scenario steady
```

Execute only that scenario:

```powershell
python -m probes.run_scenarios --execute --scenario steady
```

Supported scenario names are `steady`, `burst`, `overload_naive_retry`, `overload`, `limit_halved`, `daily_exhausted`, and `all`. During reproduction, select scenarios individually to avoid overwriting or rerunning earlier experiments unintentionally.

Validate the generated stand-in results:

```powershell
python -m probes.check_results
```

Run the test suite and compile check before committing:

```powershell
python -m pytest
$files = Get-ChildItem .\src, .\probes -Recurse -File -Filter "*.py" | ForEach-Object { $_.FullName }
python -m py_compile $files
```

## Results included in this bundle

- `results.csv`: 3,959 unique request records: 120 real Groq probe records and 3,839 stand-in scenario records. The required 12 columns are first; additional diagnostic fields follow.
- `evidence/day3_scenarios_raw.csv`: original 3,839-row stand-in load-test CSV.
- `evidence/probe_raw.csv`: original 120-row real-provider probe CSV.
- `evidence/scenario_runs.jsonl`: one manifest per stand-in scenario.
- `evidence/calibration.json`: real-provider latency and limit calibration.
- `results/scenario_summary.csv`: scenario-wise outcomes and latency percentiles.
- `results/groq_calibration_summary.csv`: real-provider model calibration (p50/p95 only because the samples are small).
- `results/rag_evaluation_status_summary.csv`: answerable/unanswerable status counts.
- `results/manual_rag_assessment.csv`: a separate, provisional manual assessment with notes.
- `results/retrieval_config_summary.csv`: Day 2 retrieval figures previously reported during the project; the original retrieval-metrics export was not part of this upload.

### Important timing definitions

- For **real** rows, `first_token_ms` is the first-token latency measured from the direct Groq streaming probe.
- For **stand-in** rows, `first_token_ms` records when the validated response is released to the caller. The pipeline holds the answer until validation completes, so it equals `total_ms` in this dataset.
- `model_first_token_ms` is the provider/model first-token measurement for real probes and the simulated model first-token measurement for stand-in runs. Use this field—not the mixed `first_token_ms` column—for model TTFT comparisons across sources.
- Do not compute one pooled latency percentile across `real` and `stand-in` rows: they are different measurement scopes.

### Status and percentile rules

Infrastructure `status` values are `ok`, `rate_limited`, `timeout`, and `error`. Stand-in reply behavior is recorded separately in `reply_kind`. Latency percentiles in the scenario summary include successful (`ok`) requests only. Percentiles use NumPy's linear method; p99 is omitted when the successful sample count is below 300.

## Observed Day 3 outcomes

| Scenario | Requests | Successful | Rate limited | Timeouts | Success rate |
|---|---:|---:|---:|---:|---:|
| Steady | 306 | 306 | 0 | 0 | 100.00% |
| Burst | 424 | 346 | 78 | 0 | 81.60% |
| Sustained overload | 1,151 | 577 | 572 | 2 | 50.13% |
| Naive retries | 1,151 | 575 | 576 | 0 | 49.96% |
| Limits halved | 461 | 353 | 108 | 0 | 76.57% |
| Daily budget exhausted | 346 | 172 | 174 | 0 | 49.71% |

The daily-budget scenario's 174 rejected requests were all confirmed as `daily_budget_exhausted`. The retry comparison produced 115 retry attempts under the budgeted policy versus 1,931 under naive immediate retries, with virtually identical success rates.

## Known limitations and interpretation

- Real Groq model calibration used only 13–15 successful samples per model. Those measurements are useful for initial calibration, not population-level performance claims; do not report p99 for these small samples.
- The tested concurrency levels (1, 2, 4, 8, 16) all succeeded; the actual provider concurrency ceiling remains unknown.
- The daily token limit and reduced RPM/TPM limits were exercised on the stand-in, not by exhausting or changing the real Groq account quota.
- The stand-in uses synthetic response content. Its load-test outcomes say nothing about RAG answer correctness.
- The scenario manifest reports `queue_peak = 1` for all runs, while request-level queue-wait measurements reach 70.63 ms. Treat `queue_peak` as a limited diagnostic rather than proof that no queuing occurred.
- The response-release time is not the same as model TTFT in the stand-in RAG path. Both fields are retained to make this distinction explicit.
- The uploaded grading review contains machine/status outcomes but no native human-grade column. The manual assessment file records the earlier human review and flags the Q1 validator discrepancy and Q4 truncation; it should be treated as provisional.
- The uploaded retrieval summary numbers were previously recorded during Day 2; the underlying retrieval-metrics export was not included in this upload.

## GitHub hygiene

Before pushing the project:

1. Keep `.env`, API keys, `.venv/`, `__pycache__/`, logs, and temporary files out of Git.
2. Commit source code, tests, the README, the decisions log, reproducible evaluation data, and the evidence/results files needed to support the report.
3. Check the final staged files before committing:

```powershell
git status --short
git diff --cached --stat
git diff --cached --name-only
```

4. Confirm `.env` is ignored and no key appears in staged files. Do not push until the staged-file review is clean.

The Day 4 evidence bundle does not include `src/`, `probes/`, `data/papers/`, or a requirements lockfile; add those from the working project before treating the GitHub repository as a fully reproducible source repository.
