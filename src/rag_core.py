"""rag_core.py - the Day 2 / Step 3 pipeline for The Relay.

Steps (each timed separately so section 4 of the write-up can be built from the log):
  1 RETRIEVE   BM25 over page-aware chunks             (local tool call, zero tokens)
  2 DECIDE     is the evidence good enough?            (branch: score threshold)
  3 DRAFT      one model call, answer cites [C#] ids   (the only token cost)
  4 CHECK      deterministic citation + number check   (zero tokens)
  loop         at most ONE re-retrieve with larger k   (bounded; skipped under pressure)
  fallback     never silently drop: show the cited passages instead

The model never writes page numbers. It cites chunk ids like [C2]; code maps ids to
(paper, pages). That removes a whole class of invented-citation errors.
"""
import json, os, re, time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

REFUSAL = "NOT_IN_PAPERS"

# ----------------------------------------------------------------- ingest --
@dataclass
class Chunk:
    cid: int
    paper: str
    page_start: int
    page_end: int
    text: str

    def pages(self):
        return (f"p.{self.page_start}" if self.page_start == self.page_end
                else f"pp.{self.page_start}-{self.page_end}")


def load_pages(pdf_path):
    """Return [(page_number, text)] using printed order. PDF index == printed number
    for these three arXiv PDFs; verify once by eye."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    return [(i, page.get_text()) for i, page in enumerate(doc, start=1)]


# Pages that are bibliography only. I first tried a regex "looks like references" heuristic:
# it wrongly dropped Moshi p4 (related work) and missed most of Moshi's reference pages, so
# the pages are listed by hand. CHECK THEM BY EYE (Attention v7: references start around p10).
SKIP_PAGES = {
    "Moshi": set(range(43, 53)),
    "Dynamic Chunking": set(range(25, 31)),
    # "Attention Is All You Need": set(range(10, 13)),   # fill in after looking at your PDF
}


def make_chunks(papers, size_words, overlap_words, skip_pages=SKIP_PAGES):
    """Create chunks without crossing papers or excluded page ranges."""
    chunks, cid = [], 0
    step = max(1, size_words - overlap_words)

    for name, pages in papers.items():
        skip = skip_pages.get(name, set())

        # Split each paper into continuous sections.
        # An excluded page or a page-number gap ends the current section.
        segments = []
        current = []
        previous_page = None

        for page, text in pages:
            if page in skip:
                if current:
                    segments.append(current)
                    current = []
                previous_page = None
                continue

            if (
                current
                and previous_page is not None
                and page != previous_page + 1
            ):
                segments.append(current)
                current = []

            current.extend((word, page) for word in text.split())
            previous_page = page

        if current:
            segments.append(current)

        # Create overlapping chunks separately inside each section.
        for words in segments:
            for start in range(0, len(words), step):
                win = words[start:start + size_words]

                if len(win) < size_words // 4 and start > 0:
                    break

                chunks.append(
                    Chunk(
                        cid,
                        name,
                        win[0][1],
                        win[-1][1],
                        " ".join(word for word, _ in win)
                    )
                )
                cid += 1

    return chunks


# -------------------------------------------------------------- retrieval --
_tok = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

def tokenize(s):
    return _tok.findall(s.lower())


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query, k=4):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# ----------------------------------------------------------------- prompt --
SYSTEM = (
    "You are a research assistant answering questions about three supplied research papers.\n"
    "You must use ONLY the passages provided in the user's message.\n\n"

    "STRICT RULES:\n"
    "1. Answer the question directly in 1-3 short sentences.\n"
    "2. Every sentence containing a factual claim MUST end with a citation "
    "such as [C1] or [C1][C2].\n"
    "3. Only use citation IDs that appear in the supplied passages.\n"
    "4. Do not invent facts, numbers, page numbers, or citations.\n"
    "5. Do not add an uncited introduction or conclusion.\n"
    "6. If the passages do not contain enough evidence, reply exactly in this format: "
    "NOT_IN_PAPERS: <what information is missing>.\n"
    "7. If passages disagree, report both values and explicitly mention the conflict, "
    "with citations for both.\n\n"

    "EXAMPLE OF THE REQUIRED FORMAT:\n"
    "Each attention head uses the dimensions stated in the supplied passage [C1].\n\n"

    "Remember: citations are mandatory for every factual claim."
)

def build_messages(question, hits):
    ctx = "\n\n".join(f"[C{n}] ({c.paper}, {c.pages()})\n{c.text}"
                      for n, (c, _s) in enumerate(hits, start=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]


# -------------------------------------------------------------- validator --
_cite = re.compile(r"[\[【]\s*C(\d+)\s*[\]】]")
_num = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

def _norm(n):
    return n.replace(",", "")


def validate(answer, hits, strict_numbers=True):
    """Check citation IDs and whether numerical claims are supported."""
    a = answer.strip()

    if a.startswith(REFUSAL):
        return "refused", [], []

    reasons, used = [], []

    ids = [int(x) for x in _cite.findall(a)]

    if not ids:
        reasons.append("no citation")

    for i in ids:
        if not (1 <= i <= len(hits)):
            reasons.append(f"cites unknown passage C{i}")
        elif i not in used:
            used.append(i)

    if strict_numbers and used and not reasons:
        cited_text = " ".join(
            hits[i - 1][0].text for i in used
        )

        cited_nums = {
            _norm(n) for n in _num.findall(cited_text)
        }

        cited_values = []

        for number in cited_nums:
            try:
                cited_values.append(Decimal(number))
            except InvalidOperation:
                pass

        body = _cite.sub("", a)
        missing = []

        for number in _num.findall(body):
            normalized = _norm(number)

            if normalized in cited_nums:
                continue

            try:
                value = Decimal(normalized)
            except InvalidOperation:
                missing.append(normalized)
                continue

            # Allow simple differences between two cited numbers.
            derived = any(
                abs(cited_values[i] - cited_values[j]) == value
                for i in range(len(cited_values))
                for j in range(i + 1, len(cited_values))
            )

            if not derived:
                missing.append(normalized)

        if missing:
            reasons.append(
                f"numbers not found or derived from cited passages: "
                f"{sorted(set(missing))}"
            )

    cites = [
        (hits[i - 1][0].paper, hits[i - 1][0].pages())
        for i in used
        if 1 <= i <= len(hits)
    ]

    # IMPORTANT: This return must remain outside the numerical-check block.
    return ("ok" if not reasons else "bad"), cites, reasons


# --------------------------------------------------------------- pipeline --
def answer(question, retriever, llm, k=4, min_top_score=0.0, allow_retry=True,
           strict_numbers=True, retry_on_refusal=False):
    """llm(messages) -> dict(text, tokens_in, tokens_out, first_token_ms, total_ms).
    Returns a dict ready to become one results.csv row (plus the visible reply)."""
    t = {"retrieve_ms": 0.0, "llm_ms": 0.0, "validate_ms": 0.0}
    tok_in = tok_out = attempts = 0
    first_token_ms = None
    trace = []
    cur_k = k
    while True:
        attempts += 1
        t0 = time.perf_counter()
        hits = retriever.search(question, cur_k)
        t["retrieve_ms"] += (time.perf_counter() - t0) * 1000

        # DECIDE (branch): no evidence -> refuse without spending tokens.
        if not hits:
            trace.append(f"attempt{attempts}: no passages retrieved")
            return _result(
                "refused",
                "I could not find relevant evidence in the supplied papers.",
                [],
                hits,
                attempts,
                tok_in,
                tok_out,
                first_token_ms,
                t,
                trace,
                spent_llm=False
            )

        # DECIDE (branch): weak evidence -> refuse without spending tokens.
        if hits[0][1] <= min_top_score:
            trace.append(
                f"attempt{attempts}: top score {hits[0][1]:.2f} < {min_top_score}"
            )
            return _result(
                "refused",
                "The papers do not appear to cover this.",
                [],
                hits,
                attempts,
                tok_in,
                tok_out,
                first_token_ms,
                t,
                trace,
                spent_llm=False
            )

        llm_start = time.perf_counter()

        try:
            r = llm(build_messages(question, hits))

        except Exception as exc:
            t["llm_ms"] += (time.perf_counter() - llm_start) * 1000

            error_message = f"{type(exc).__name__}: {exc}"
            trace.append(
                f"attempt{attempts}: model request failed: {error_message}"
            )

            result = _result(
                "error",
                "The model service could not complete this request. Please try again.",
                [],
                hits,
                attempts,
                tok_in,
                tok_out,
                first_token_ms,
                t,
                trace
            )

            result["error_message"] = error_message
            return result

        previous_llm_ms = t["llm_ms"]

        t["llm_ms"] += r["total_ms"]
        tok_in += r["tokens_in"] or 0
        tok_out += r["tokens_out"] or 0
        if (
            "model_first_token_ms" not in t
            and r.get("first_token_ms") is not None
        ):
            t["model_first_token_ms"] = r["first_token_ms"]
        if first_token_ms is None and r.get("first_token_ms") is not None:
            first_token_ms = (
                t["retrieve_ms"]
                + previous_llm_ms
                + r["first_token_ms"]
            )

        t1 = time.perf_counter()
        status, cites, reasons = validate(r["text"], hits, strict_numbers)
        t["validate_ms"] += (time.perf_counter() - t1) * 1000
        trace.append(f"attempt{attempts}: k={cur_k} -> {status} {reasons}")

        # Keep a preview of the model's response for debugging.
        trace.append(f"Draft preview: {r['text'][:500]!r}")

        if status == "ok":
            return _result("answered", r["text"], cites, hits, attempts, tok_in, tok_out,
                           first_token_ms, t, trace)

        # LOOP (bounded): one retry with more evidence, only if allowed.
        # Retrying a REFUSAL doubles the cost of every unanswerable question, so it is off by
        # default. Turn it on only if your eval shows k*2 rescues answerable-but-refused ones.
        retry_ok = allow_retry and attempts < 2 and (status == "bad" or retry_on_refusal)
        if retry_ok:
            cur_k = k + 2
            continue
        if status == "refused":
            return _result("refused", r["text"], [], hits, attempts, tok_in, tok_out,
                           first_token_ms, t, trace)

        # bad twice: do not send the unverified text; show the evidence instead.
        shown = [(c.paper, c.pages()) for c, _ in hits[:3]]
        msg = ("I could not verify an answer against the papers. The most relevant passages "
               "are cited below so you can read them directly.")
        return _result("unverified", msg, shown, hits, attempts, tok_in, tok_out,
                       first_token_ms, t, trace)


def _result(status, text, cites, hits, attempts, tin, tout, ftm, t, trace, spent_llm=True):
    return {
        "status": status,
        "reply": text,
        "citations": cites,
        "retrieved_pages": [(c.paper, c.pages()) for c, _ in hits],
        "attempts": attempts,
        "tokens_in": tin,
        "tokens_out": tout,
        "first_token_ms": ftm,
        "total_ms": t["retrieve_ms"] + t["llm_ms"] + t["validate_ms"],
        "model_first_token_ms": t.get("model_first_token_ms"),
        "retrieve_ms": t["retrieve_ms"],
        "llm_ms": t["llm_ms"],
        "validate_ms": t["validate_ms"],
        "trace": trace
    }


# ------------------------------------------------------------ groq client --

def make_groq_llm(
    model="openai/gpt-oss-120b",
    max_completion_tokens=300,
    reasoning_effort="low",
    base_url="https://api.groq.com/openai/v1",
    timeout=60,
    api_key=None,
    use_env_key=True
):
    """Streaming client for Groq or the local OpenAI-compatible stand-in."""

    import httpx

    if use_env_key:
        from pathlib import Path
        from dotenv import load_dotenv

        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")

        api_key = api_key or os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")

    def llm(messages):

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0
        }

        if "api.groq.com" in base_url:
            body["reasoning_effort"] = reasoning_effort
            body["include_reasoning"] = False

        headers = {}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.perf_counter()
        first = None
        parts = []
        usage = {}
        rate_headers = {}

        with httpx.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout
        ) as response:

            rate_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower().startswith("x-ratelimit")
                or key.lower() == "retry-after"
            }

            if response.status_code != 200:
                response.read()

                raise RuntimeError(
                f"HTTP {response.status_code} "
                f"retry-after={response.headers.get('retry-after')} "
                f"rate_headers={rate_headers} "
                f"body={response.text[:500]}"
            )

            for line in response.iter_lines():

                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()

                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                usage = (
                    chunk.get("usage")
                    or (chunk.get("x_groq") or {}).get("usage")
                    or usage
                )

                choices = chunk.get("choices") or []

                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:

                    if first is None:
                        first = (
                            time.perf_counter() - t0
                        ) * 1000

                    parts.append(content)

        generated_text = "".join(parts)
        total = (time.perf_counter() - t0) * 1000

        details = usage.get("completion_tokens_details") or {}

        return {
            "text": generated_text,
            "first_token_ms": first,
            "total_ms": total,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "rate_limit_headers": rate_headers
        }

    return llm
