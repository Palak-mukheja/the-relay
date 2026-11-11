
import json
import re
import time
import uuid
from datetime import datetime

from src.rag_core import answer


def classify_status(result):
    pipeline_status = result.get("status")

    if pipeline_status in ("answered", "refused", "unverified"):
        return "ok"

    error = result.get("error_message", "").lower()

    if "429" in error or "rate limit" in error:
        return "rate_limited"

    if "timeout" in error or "timed out" in error:
        return "timeout"

    return "error"


def classify_reply(result):
    mapping = {
        "answered": "answered",
        "refused": "refused",
        "unverified": "unverified",
        "error": "unavailable"
    }

    return mapping.get(result.get("status"), "degraded")


def run_and_log(
    question,
    retriever,
    llm,
    logger,
    scenario,
    model,
    retrieval_setting,
    source="real",
    allow_retry=True,
    answer_grade="",
    queue_wait_ms=0.0,
    retry_wait_ms=0.0,
    seed=None,
    queue_peak=None
):
    if source not in ("real", "stand-in"):
        raise ValueError("source must be 'real' or 'stand-in'")

    request_id = str(uuid.uuid4())

    start_time = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )

    # Measure actual time spent in the pipeline.
    work_start = time.perf_counter()

    result = answer(
        question=question,
        retriever=retriever,
        llm=llm,
        allow_retry=allow_retry
    )

    work_ms = (time.perf_counter() - work_start) * 1000

    status = classify_status(result)
    reply_kind = classify_reply(result)

    total_ms = queue_wait_ms + work_ms + retry_wait_ms

    # The answer is held until validation finishes.
    # Therefore, user-visible first token is when the
    # validated response becomes available.
    visible_first_token_ms = (
        total_ms if status == "ok" else None
    )

    # This is the pipeline's measured time to the first
    # model token, before the response is validated.
    model_first_token_ms = result.get(
        "model_first_token_ms",
        result.get("first_token_ms")
    )

    error_message = result.get("error_message", "")

    status_code_match = re.search(
        r"HTTP\s+(\d{3})",
        error_message,
        re.IGNORECASE
    )

    status_code = (
        int(status_code_match.group(1))
        if status_code_match else None
    )

    citations = json.dumps(
        result.get("retrieved_pages", []),
        ensure_ascii=False
    )

    logger.log(
        request_id=request_id,
        scenario=scenario,
        source=source,
        start_time=start_time,
        first_token_ms=visible_first_token_ms,
        total_ms=total_ms,
        status=status,
        retries=max(0, result.get("attempts", 1) - 1),
        tokens_in=result.get("tokens_in") or 0,
        tokens_out=result.get("tokens_out") or 0,
        retrieval_setting=retrieval_setting,
        answer_grade=answer_grade,

        model=model,
        model_first_token_ms=model_first_token_ms,
        status_code=status_code,
        detail=error_message,
        reply_kind=reply_kind,
        reasoning_tokens=result.get("reasoning_tokens"),
        rate_limit_headers=json.dumps(
            result.get("rate_limit_headers", {}),
            ensure_ascii=False
        ),
        error_message=error_message,
        queue_wait_ms=queue_wait_ms,
        work_ms=work_ms,
        retry_wait_ms=retry_wait_ms,
        attempts=result.get("attempts", 1),
        retrieved_pages=citations,
        reply_text=result.get("reply", ""),
        seed=seed,
        queue_peak=queue_peak
    )

    # Retain the pipeline's original answer status for the caller.
    result["infra_status"] = status
    result["reply_kind"] = reply_kind
    result["first_token_ms"] = visible_first_token_ms
    result["model_first_token_ms"] = model_first_token_ms
    result["total_ms"] = total_ms
    result["work_ms"] = work_ms

    return result
