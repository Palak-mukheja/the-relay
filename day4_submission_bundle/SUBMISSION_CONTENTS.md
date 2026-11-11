# Bundle contents and integration notes

## Included

- `The_Relay_Day4_Report.docx` — two-page experiment report.
- `README.md` — project overview, reproduction steps, data definitions, limitations, Git hygiene.
- `DECISIONS_FINAL.md` — updated decisions, observed results, and caveats.
- `results.csv` — merged 3,959-row request log, combining the 120 real Groq probes and 3,839 stand-in scenario rows. The required 12 fields remain first.
- `.gitignore` and `.env.example` — safe repository defaults and a placeholder environment file.
- `results/` — derived scenario, Groq calibration, retrieval, and RAG assessment summaries.
- `evidence/` — unchanged copies of the uploaded raw CSVs, calibration JSON, scenario manifest, question set, grading review, and original decisions note.

## Not included

This is a Day 4 analysis/evidence bundle, not a complete source-code repository. It does not include `src/`, `probes/`, the original paper PDFs, extracted paper text, or a requirements lockfile. Bring those over from the working project before pushing the complete repository to GitHub. The README lists the expected layout and reproduction commands.

## Important data notes

- The merged `results.csv` contains both `real` and `stand-in` rows. Do not pool latency percentiles across these sources.
- For stand-in requests, `first_token_ms` is the time until the validated response is released; `model_first_token_ms` is the model's simulated TTFT.
- `answer_grade` is blank in request logs because the load-test rows are synthetic. RAG answer assessment is kept separately in `results/manual_rag_assessment.csv`.
- Retrieval summary values in `results/retrieval_config_summary.csv` were recorded earlier during Day 2; the source metrics export was not included with the current upload.
