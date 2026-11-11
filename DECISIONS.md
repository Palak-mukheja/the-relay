# Project Decisions — The Relay

## 1. Product priorities and response targets

Programming language: Python.

Priority order:
1. Correctness and reliability
2. Availability
3. Response speed
4. Cost efficiency

Provisional performance targets:
- First model token: p95 below 2,000 ms.
- Total response time: p95 below 12,000 ms.
- User-abandon deadline: 15,000 ms.

These are project targets, not guarantees. In the validated RAG path, the answer is released only after validation, so response-release latency and model TTFT are distinct measurements.

## 2. Source library and evaluation

The corpus is pinned to:
- Attention Is All You Need — arXiv:1706.03762v7
- Moshi — arXiv:2410.00037v1
- Dynamic Chunking for End-to-End Hierarchical Sequence Modeling — arXiv:2507.07955v2

The fixed evaluation set contains 24 questions, including cross-paper questions, source-contradiction handling, and unanswerable questions. Printed paper page numbers are used for expected sources.

## 3. Retrieval decisions

The medium chunking configuration (400 words, 80-word overlap; four retrieved chunks on average) was retained as the provisional baseline because the previously recorded 24-question retrieval-only run had the highest page recall (0.7540), source recall (0.7698), and full-source coverage (66.67%) among the tested configurations. Large chunks had the best MRR (0.8095) and precision proxy (0.5952), so medium is a recall/coverage trade-off rather than a universal winner.

The retrieval-only figures are copied from the earlier Day 2 discussion; the raw metrics export was not included in the Day 4 upload.

## 4. Real Groq model calibration

GPT-OSS 120B was selected as the provisional main model. In the calibration sample it completed 15/15 requests, with model TTFT p50/p95 of 477.50/1,161.87 ms and total latency p50/p95 of 485.66/1,168.37 ms. GPT-OSS 20B also completed 15/15 but was slower in the sample. Qwen 3.8 27B was faster in the 13 successful samples, but one of 14 attempts failed. Samples are too small to make strong comparative claims.

Real-provider limits recorded from the probe/console evidence:
- RPM: a controlled probe reached 30 successful requests followed by two HTTP 429 responses; recovery was observed after the supplied 2-second Retry-After interval. Treat 30 RPM as the observed threshold in this probe, not a universal or permanent account limit.
- TPM: 8,000, based on console/response-header evidence. Controlled output-cap tests did not trigger a TPM 429; full output reservation was not established.
- RPD: 1,000, based on console/response-header evidence.
- TPD: 200,000, console-reported and not independently exhausted.
- Concurrency: requests succeeded at tested levels 1, 2, 4, 8, and 16. Actual provider concurrency limit remains unknown.

The real model calibration samples were only 13–15 per model. p99 is not used for those small samples.

## 5. Stand-in calibration and scenario design

The stand-in uses latency distributions calibrated from GPT-OSS 120B measurements. To make repeatable traffic tests practical, its nominal limits were scaled to 40x the observed real-provider RPM/TPM values: 1,200 RPM and 320,000 TPM. The baseline daily budget was 8,000,000 tokens except in the daily-exhaustion scenario.

Baseline request capacity was estimated at 115.09 requests/minute using 320,000 tokens/minute divided by approximately 2,780.38 tokens per successful answer.

The six open-loop scenarios were:
- Steady: approximately 0.5x baseline for 300 seconds.
- Burst: baseline traffic with a short 4x burst segment.
- Sustained overload: approximately 2x baseline for 300 seconds.
- Naive retries: same sustained-overload arrival schedule with immediate retries, up to three retries.
- Limits halved: approximately 0.8x baseline, with RPM and TPM halved after 120 seconds.
- Daily budget exhausted: approximately 0.5x baseline for about six minutes with a deliberately constrained daily token budget.

All 3,839 scenario rows have source `stand-in`. These load tests do not represent live Groq traffic or real-model answer quality.

## 6. Retry policies

Budgeted policy: honor Retry-After where present, use exponential backoff with jitter, permit at most one retry per request, and enforce a shared retry budget of 10% of admitted requests.

Naive policy: immediate retries, no Retry-After honoring, and up to three retries per request.

Observed sustained-overload comparison:
- Budgeted: 115 retry attempts across 115 requests; 577/1,151 successful (50.13%); 2 timeouts.
- Naive: 1,931 retry attempts across 690 requests; 575/1,151 successful (49.96%); no timeouts.

Naive retries created about 16.8x as many retry attempts without a meaningful success-rate improvement. Budgeted retries imposed greater waiting time; this trade-off must be stated rather than describing the budgeted policy as uniformly faster.

## 7. Logging and analysis rules

Required first 12 CSV columns, in order:
`request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`.

Extra diagnostic columns follow the required columns. One row is stored per request. `source` is `real` or `stand-in`.

- `model_first_token_ms` records model/provider TTFT.
- In the stand-in RAG path, `first_token_ms` means time until the validated response is released. It equals `total_ms` because the pipeline waits for validation before releasing the answer.
- `queue_wait_ms + work_ms + retry_wait_ms` equals `total_ms` within floating-point precision in the uploaded scenario CSV.
- Infrastructure status values: `ok`, `rate_limited`, `timeout`, `error`. Reply behavior is stored separately as `reply_kind`.
- Latency percentiles use successful requests only and NumPy's linear percentile method. p99 is reported only when the successful sample count is at least 300.
- Do not pool real-provider and stand-in latency percentiles: their measurement scopes differ.

## 8. Data-quality findings and caveats

The uploaded stand-in scenario CSV has 3,839 unique request IDs, correct required-column order, no missing timing components, and no timing-sum mismatches above 2 ms. The six manifests each report the scheduled number of rows as logged. The combined `results.csv` adds 120 unique real probe rows, for 3,959 unique requests total.

The manifest's `queue_peak` is 1 for every scenario, while row-level queue-wait values reach 70.63 ms. The peak counter should be treated as a limited diagnostic, not proof that no queueing occurred.

The daily-budget scenario recorded 174 rejected requests; all 174 error bodies were confirmed as `daily_budget_exhausted`. The final manifest reports 477,869 daily tokens consumed against a 480,108-token test budget; the remaining balance was too small for another request.

The Windows Proactor `WinError 10054` connection-reset warning appeared during some scenario runs. The runner nevertheless logged all scheduled rows, and the CSV integrity checker reported no data errors. Preserve the warning in project notes; do not claim the underlying connection reset was fixed.

## 9. Current answer-quality assessment

The uploaded evaluation status file contains 15 `answered`, 7 `refused`, and 2 `unverified` outputs. Among 21 answerable questions: 15 were labeled answered, 4 were refused, and 2 were unverified. All 3 unanswerable questions were refused.

The earlier manual review recorded 8 correct, 8 partial, 4 incorrect refusals, and 1 ungraded/truncated answer among the 21 answerable questions. That is 57.1% weighted credit when partial answers receive half credit (12/21). This is provisional: the uploaded CSV does not contain a dedicated manual-grade field, and Q1's earlier draft assessment differs from its final `unverified` status. Q4 remains ungraded because its response was truncated.

## 10. Submission rule

Preserve raw evidence. Keep API keys and `.env` files out of Git. Include source code, tests, pinned paper versions where license permits, evaluation data, `results.csv`, calibration evidence, scenario manifests, the README, this decisions log, and the final report in the repository.
