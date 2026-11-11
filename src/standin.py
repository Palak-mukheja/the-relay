
import asyncio
import json
import time
import uuid

from datetime import datetime, timezone

import tiktoken

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from src.standin_state import StandinState


app = FastAPI(title="The Relay - Controllable Stand-in")

state = StandinState()

try:
    tokenizer = tiktoken.get_encoding("o200k_base")
except Exception:
    tokenizer = None


def count_tokens(text):

    if tokenizer:
        return len(
            tokenizer.encode(
                text,
                disallowed_special=()
            )
        )

    return max(1, len(text) // 4)


def error_response(status_code, code, message, retry_after=None):

    headers = state.rate_headers()

    if retry_after is not None:
        headers["Retry-After"] = str(
            max(1, int(retry_after + 0.999))
        )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": code,
                "code": code,
                "message": message
            }
        },
        headers=headers
    )


@app.get("/health")
def health():

    return {"status": "running", "service": "relay-standin"}


@app.get("/admin/state")
def admin_state():

    return state.snapshot()


@app.patch("/admin/config")
def admin_config(updates: dict):

    try:
        return state.update_config(updates)

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )


@app.post("/admin/reset")
def admin_reset(payload: dict = None):

    seed = (payload or {}).get("seed", 42)

    return state.reset(seed)


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):

    request_id = str(uuid.uuid4())

    messages = payload.get("messages", [])

    if not messages:
        return error_response(
            400, "invalid_request", "Messages cannot be empty."
        )

    model = payload.get("model", "openai/gpt-oss-120b")

    max_tokens = payload.get(
        "max_completion_tokens",
        payload.get("max_tokens", 256)
    )

    if not isinstance(max_tokens, int) or max_tokens < 1:
        return error_response(
            400, "invalid_request", "Invalid completion token limit."
        )

    prompt = "\n".join(
        str(message.get("content", ""))
        for message in messages
    )

    tokens_in = count_tokens(prompt)



    # The stand-in uses a fixed simulated answer.
    answer = (
        "The supplied passages contain information relevant "
        "to this question [C1]."
    )

    tokens_out = count_tokens(answer)

    # Charge actual simulated usage, not the maximum allowed output.
    reserved_tokens = tokens_in + tokens_out

    admission = state.admit(reserved_tokens)

    if not admission["allowed"]:

        status_code = (
            503 if admission["code"] == "overloaded"
            else 429
        )

        return error_response(
            status_code,
            admission["code"],
            admission["message"],
            admission["retry_after"]
        )

    if state.should_fail():

        state.release()

        return error_response(
            503,
            "simulated_failure",
            "The stand-in deliberately simulated a server failure."
        )

    ttft_ms, total_ms = state.sample_latency()




    usage = {
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out
    }

    response_id = f"chatcmpl-{request_id}"
    created = int(time.time())

    headers = state.rate_headers()

    # -------------------------------------------------
    # STREAMING RESPONSE
    # -------------------------------------------------

    if payload.get("stream", False):

        async def generate():

            try:

                # Delay before the first visible answer token.
                await asyncio.sleep(ttft_ms / 1000)

                words = answer.split(" ")

                interval = (
                    max(0, total_ms - ttft_ms)
                    / max(1, len(words) - 1)
                )

                for index, word in enumerate(words):

                    if index > 0:
                        await asyncio.sleep(interval / 1000)

                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None
                        }]
                    }

                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage
                }

                yield f"data: {json.dumps(final_chunk)}\n\n"

                yield "data: [DONE]\n\n"

            finally:
                state.release()

        headers["x-request-id"] = request_id

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=headers
        )

    # -------------------------------------------------
    # NON-STREAMING RESPONSE
    # -------------------------------------------------

    try:

        await asyncio.sleep(total_ms / 1000)

        return JSONResponse(
            content={
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
                "usage": usage
            },
            headers=headers
        )

    finally:
        state.release()
