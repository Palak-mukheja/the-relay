# Groq API Research

## Account Information

Plan: Free
Selected Model: openai/gpt-oss-20b

## Rate Limits

RPM: 30
RPD: 1000
TPM: 8000
TPD: 200000

## Other Restrictions

Concurrency limit: Not specified yet.
To be investigated through experiments.

## Observations

- Limits are model-specific.
- The project inherits organization limits.
- HTTP 429 indicates a rate-limit response.
- Need to investigate actual recovery time.

## First Real Groq API Request

Model: openai/gpt-oss-20b
HTTP Status: 200

TTFT: 773.55 ms
Total Time: 781.65 ms

Input Tokens: 80
Output Tokens: 20
Reasoning Tokens: 6

Rate Limit Headers:

Limit Requests: 1000
Remaining Requests: 999
Limit Tokens: 8000
Remaining Tokens: 7792

Reset Requests: 1m26.4s
Reset Tokens: 1.559s

Observations:

1. Streaming worked successfully.
2. First answer token arrived in approximately 774 ms.
3. The request completed in approximately 782 ms.
4. TPM counter decreased by 208.
5. Input tokens + requested maximum completion tokens
   also equals 208.
6. Need further experiments to verify token reservation.
7. Need to investigate reset-request timing.

## Probe A: Model Comparison

### GPT-OSS 20B
Requests: 15
Successful: 15
Failed: 0

TTFT p50: 698.59 ms
TTFT p95: 1406.35 ms

Total p50: 707.93 ms
Total p95: 1411.35 ms

### GPT-OSS 120B
Requests: 15
Successful: 15
Failed: 0

TTFT p50: 477.50 ms
TTFT p95: 1161.87 ms

Total p50: 485.66 ms
Total p95: 1168.37 ms

### Qwen 27B
Requests attempted: 14
Successful: 13
Failed: 1

Failure: HTTP 503
Reason: Model over capacity.

TTFT p50: 449.44 ms
TTFT p95: 670.95 ms

Total p50: 460.35 ms
Total p95: 682.76 ms

### Decision

GPT-OSS 120B is the provisional primary
candidate because it was faster than GPT-OSS 20B
in this experiment and completed all requests.

Qwen was fastest but encountered one 503 error.

More testing and answer-quality evaluation are required.

## Probe B: Concurrency Testing

Model: openai/gpt-oss-120b

Concurrency levels tested:
1, 2, 4, 8, 16

Total requests: 31
Successful: 31
Failed: 0

Maximum successful concurrency: 16

### Observations

1. All requests succeeded.
2. No HTTP 429 errors observed.
3. No HTTP 503 errors observed.
4. Latency did not increase monotonically.
5. No separate concurrency restriction was observed
   up to 16 simultaneous requests.

### Conclusion

The service successfully handled 16 simultaneous
requests during this experiment.

The actual maximum concurrency limit remains unknown.
## Probe C: RPM Testing

Model: openai/gpt-oss-120b

Requests attempted: 32
Successful: 30
Rate-limited: 2
Initial probe duration: 4.77 seconds

HTTP status: 429
Retry-After: 2 seconds

Recovery:
First poll: 429
Second poll: 200

Observed polling recovery interval: 2.38 seconds
Exact time from first 429: Not measured

Observations:
- Concurrent requests can complete even after
  another request receives HTTP 429.
- The service recovered after the retry window.
- Results are consistent with the console's
  30 RPM limit.
- Error body must be inspected to identify
  the exact exhausted limit.
  ### Probe C: Verified RPM Limit

Model: openai/gpt-oss-120b

RPM: 30
Confidence: High

Evidence:
HTTP 429 error explicitly reported:
Limit 30, Used 30, Requested 1.

Requests attempted: 32
Successful: 30
Rate limited: 2

Retry-After: 2 seconds

Recovery:
First poll: HTTP 429
Second poll: HTTP 200

Observed polling recovery interval: 2.38 seconds.
Exact recovery time from first 429: Not measured.

Daily remaining requests at error:
939

Important:
x-ratelimit-reset-requests refers to the
daily request window, not the RPM window.

## Probe D: TPM Accounting

Model: openai/gpt-oss-120b

### Test 1: Low output limit
Maximum completion tokens: 10
Input tokens: 76
Output tokens: 10
Remaining TPM: 7914
Header-implied usage: 86

### Test 2: High output limit
Maximum completion tokens: 2000
Input tokens: 76
Output tokens: 17
Remaining TPM: 7910
Header-implied usage: 90

### Findings

- Both tests used isolated TPM windows.
- The 2000-token maximum did not result in
  2000 tokens being deducted from the TPM allowance.
- Header-implied usage was close to actual usage.
- The high-limit test had a 3-token discrepancy.
- An earlier observation suggesting full reservation
  was not reproduced.

Conclusion:
No evidence of full maximum-output reservation
in this controlled experiment.


# Table 1: Limits Found in the Real Service

| Limit | Value | How Found | Confidence |
|---|---|---|---|
| Requests per minute | 30 | Groq console and HTTP 429 error explicitly reporting Limit 30 | High |
| Tokens per minute | 8000 | Console and response headers | Medium |
| Requests per day | 1000 | Console and response headers | Medium |
| Tokens per day | 200000 | Groq console | Medium |
| Requests in flight at once | At least 16 | Successfully tested 16 concurrent requests; actual limit unknown | Medium |
| What it returns when a limit is hit | HTTP 429, rate_limit_exceeded, retry-after: 2 seconds | Actual RPM limit experiment | High |
| Time until recovery | Approximately 2.38 seconds after recovery polling began; exact time from first 429 not measured | Recovery polling experiment | Medium |

## Model Budget Independence Experiment

Models:
- openai/gpt-oss-20b
- openai/gpt-oss-120b

Requests sent: 3 per model

### Observations

GPT-OSS 20B:
999 -> 998 -> 997

GPT-OSS 120B:
950 -> 949 -> 948

### Finding

Each model's daily request counter decreased
only when that model was called.

Calls to the other model did not appear to
decrease its counter.

Conclusion:
Evidence supports model-specific RPD counters.

Confidence: Medium.

TPM independence: Not established.
## Unanswerable Question Verification

Question:
How many dollars did it cost to train the big model,
and how much energy did it use?

Documents checked:
1. Attention Is All You Need v7
2. Moshi v1
3. Dynamic Chunking v2

Search terms:
dollars, USD, $, energy, kWh, power, watt,
electricity, training cost.

Finding:
The papers discuss FLOPs, computational cost,
hardware and training time, but do not provide
the requested monetary training cost or energy use.

Expected behaviour:
Assistant should refuse to provide numerical values
and explain that the information is not available.

## Day 2: Retrieval-Only Evaluation

Questions: 24
Configurations: 4
Total retrieval tests: 96

### Setting S
Mean page recall: 72.22%
Precision proxy: 38.10%
MRR: 0.6865
Full page coverage: 61.90%
Wrong-paper rate: 18.25%

### Setting M
Mean page recall: 75.40%
Precision proxy: 47.62%
MRR: 0.6905
Full page coverage: 66.67%
Wrong-paper rate: 14.29%

### Setting L
Mean page recall: 71.43%
Precision proxy: 59.52%
MRR: 0.8095
Full page coverage: 61.90%
Wrong-paper rate: 11.90%

### Setting M+R
Mean page recall: 73.02%
Precision proxy: 46.43%
MRR: 0.7143
Full page coverage: 61.90%
Wrong-paper rate: 14.29%

### Observations

1. Medium has the highest mean page recall.
2. Medium has the highest complete source-group coverage.
3. Large has the highest MRR.
4. Large has the lowest wrong-paper chunk rate.
5. M+R did not consistently improve retrieval.
6. All three unanswerable questions returned candidates.

### Provisional Decision

Medium is the preferred baseline because it provides
the strongest evidence coverage.

Large will remain a comparison setting because
it provides better ranking and fewer unrelated
paper results.

Precision values are page-overlap proxies, not
manually judged semantic precision.


## Day 3 — Experiment definitions

### Latency
- User-abandon deadline: 15 seconds total.
- Record model_first_token_ms separately from user-visible first_token_ms.
- User-visible first token is measured after validation if the answer is held until validation completes.
- Only status=ok requests are included in latency percentiles.
- Report p50 and p95 with sample size n.
- Report p99 only when n >= 300.
- Percentile calculation: numpy percentile method="linear".

### Infrastructure status
- ok: request completed successfully.
- rate_limited: provider HTTP 429 is the final outcome.
- timeout: deadline exceeded, user abandoned, or request shed while queued.
- error: all other failures.
- Record the specific cause in the extra column detail.

### User-visible outcome
- reply_kind: answered, refused, unverified, degraded, unavailable.

### Data source
- real: actual Groq service.
- stand-in: controllable local service.

### CSV
- Required 12 columns must appear first and in the assignment's exact order.
- Additional measurement columns follow the required columns.
- One CSV row per request, including failed and shed requests.

### Time accounting
- queue_wait_ms: time waiting for gateway admission.
- work_ms: actual pipeline/model processing time.
- retry_wait_ms: time spent waiting between attempts.
- These three values must sum to total_ms within measurement tolerance.

### Retry policy — provisional
- Respect provider retry-after.
- Use exponential backoff with jitter.
- Maximum one retry per request.
- Do not retry if the remaining deadline cannot accommodate the wait and another attempt.
- Use a global retry budget of 10% of admitted requests per run.
