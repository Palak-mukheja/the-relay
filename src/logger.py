
import csv
import threading
from pathlib import Path


class Logger:

    REQUIRED_FIELDS = [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade"
    ]

    EXTRA_FIELDS = [
        "model",
        "model_first_token_ms",
        "status_code",
        "detail",
        "reply_kind",
        "reasoning_tokens",
        "rate_limit_headers",
        "error_message",
        "queue_wait_ms",
        "work_ms",
        "retry_wait_ms",
        "attempts",
        "retrieved_pages",
        "reply_text",
        "seed",
        "queue_peak",
        "run_id",
        "scheduled_offset_ms",
        "experiment_variant",
    ]

    FIELDS = REQUIRED_FIELDS + EXTRA_FIELDS

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()

        write_header = (
            not self.path.exists()
            or self.path.stat().st_size == 0
        )

        self.file = open(
            self.path,
            "a",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDS,
            extrasaction="raise"
        )

        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def log(self, **row):
        missing = [
            field for field in self.REQUIRED_FIELDS
            if field not in row
        ]

        if missing:
            raise ValueError(
                f"Missing required CSV fields: {missing}"
            )

        with self.lock:
            self.writer.writerow(row)
            self.file.flush()

    def close(self):
        with self.lock:
            if not self.file.closed:
                self.file.close()
