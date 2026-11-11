
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from src.gateway import Gateway
from src.rag_core import (
    load_pages,
    make_chunks,
    Retriever,
    make_groq_llm,
    answer
)
from src.logger import Logger


ROOT = Path(__file__).resolve().parents[1]

PAPER_FILES = {
    "Attention Is All You Need": ROOT / "data/papers/attention_v7.pdf",
    "Moshi": ROOT / "data/papers/moshi_v1.pdf",
    "Dynamic Chunking": ROOT / "data/papers/dynamic_chunking_v2.pdf"
}

STANDIN_URL = "http://127.0.0.1:8000/v1"
MODEL = "openai/gpt-oss-120b"


def classify_reply(result):

    mapping = {
        "answered": "answered",
        "refused": "refused",
        "unverified": "unverified",
        "error": "unavailable"
    }

    return mapping.get(result.get("status"), "unavailable")


async def main():

    # 1. Load papers.
    papers = {
        name: load_pages(str(path))
        for name, path in PAPER_FILES.items()
    }

    # 2. Create medium-sized chunks.
    chunks = make_chunks(
        papers,
        size_words=400,
        overlap_words=80
    )

    retriever = Retriever(chunks)

    print(f"Loaded {len(papers)} papers.")
    print(f"Created {len(chunks)} chunks.")

    # 3. Configure the local stand-in client.
    llm = make_groq_llm(
        model=MODEL,
        max_completion_tokens=160,
        base_url=STANDIN_URL,
        use_env_key=False
    )

    # 4. Create the gateway.
    gateway = Gateway(
        max_concurrency=4,
        queue_size=50,
        tpm=160000,
        deadline_seconds=15,
        retry_budget_fraction=0.10,
        max_retries=1
    )

    logger = Logger(ROOT / "results/gateway_smoke_test.csv")

    await gateway.start()

    try:

        question = input("\nAsk a question about the papers: ").strip()

        if not question:
            print("No question entered.")
            return

        start_time = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )

        # This function is executed by a gateway worker.
        # Inner RAG retries are disabled; gateway owns retries.
        def process_request():

            return answer(
                question=question,
                retriever=retriever,
                llm=llm,
                allow_retry=False
            )

        # 5. Submit through the gateway.
        result = await gateway.submit(
            fn=process_request,
            estimated_tokens=3000,
            request_timeout=15
        )

        infra_status = result.get("infra_status", "error")
        reply_kind = classify_reply(result)

        queue_wait = result.get("queue_wait_ms", 0.0)
        work = result.get("work_ms", 0.0)
        retry_wait = result.get("retry_wait_ms", 0.0)
        total = result.get("total_ms", 0.0)

        # We hold the response until validation has finished.
        visible_ttft = total if infra_status == "ok" else None

        model_ttft = result.get("model_first_token_ms")

        error_message = (
            result.get("error_message")
            or result.get("detail")
            or ""
        )

        match = re.search(r"HTTP\s+(\d{3})", error_message)

        status_code = int(match.group(1)) if match else None

        # 6. Log exactly one row.
        logger.log(
            request_id=__import__("uuid").uuid4().hex,
            scenario="gateway_smoke_test",
            source="stand-in",
            start_time=start_time,
            first_token_ms=visible_ttft,
            total_ms=total,
            status=infra_status,
            retries=result.get("gateway_retries", 0),
            tokens_in=result.get("tokens_in") or 0,
            tokens_out=result.get("tokens_out") or 0,
            retrieval_setting="M",
            answer_grade="",

            model="stand-in",
            model_first_token_ms=model_ttft,
            status_code=status_code,
            detail=error_message,
            reply_kind=reply_kind,
            reasoning_tokens=result.get("reasoning_tokens"),
            rate_limit_headers="{}",
            error_message=error_message,
            queue_wait_ms=queue_wait,
            work_ms=work,
            retry_wait_ms=retry_wait,
            attempts=result.get("gateway_attempts", 0),
            retrieved_pages=json.dumps(
                result.get("retrieved_pages", [])
            ),
            reply_text=result.get("reply", ""),
            seed=42,
            queue_peak=result.get("queue_peak", 0)
        )

        print("\n--- ANSWER ---")
        print(result.get("reply", error_message))

        print("\n--- STATUS ---")
        print("Infrastructure:", infra_status)
        print("User response:", reply_kind)

        print("\n--- LATENCY ---")
        print(f"Queue wait:       {queue_wait:.2f} ms")
        print(f"Model TTFT:       {model_ttft}")
        print(f"Work:             {work:.2f} ms")
        print(f"Retry wait:       {retry_wait:.2f} ms")
        print(f"User-visible TTFT:{visible_ttft}")
        print(f"Total:            {total:.2f} ms")

        print("\n--- GATEWAY ---")
        print("Attempts:", result.get("gateway_attempts"))
        print("Retries:", result.get("gateway_retries"))
        print("Peak queue:", result.get("queue_peak"))

        print("\n--- TOKENS ---")
        print("Input:", result.get("tokens_in"))
        print("Output:", result.get("tokens_out"))

        print("\nSaved to results/gateway_smoke_test.csv")

    finally:
        await gateway.close()
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())
