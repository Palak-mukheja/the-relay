
from pathlib import Path
from datetime import datetime, timezone

from src.logger import Logger


# Store test data separately from actual experiment results
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "results" / "logger_test.csv"

# Create the logger
logger = Logger(OUTPUT_FILE)

# Simulate one request
logger.log(
    request_id=1,
    scenario="test",
    source="real",
    model="openai/gpt-oss-20b",
    start_time=datetime.now(timezone.utc).isoformat(),
    first_token_ms=850,
    total_ms=2400,
    status="ok",
    status_code=200,
    retries=0,
    tokens_in=1200,
    tokens_out=250,
    reasoning_tokens=0,
    retrieval_setting="test",
    answer_grade="correct"
)

# Close the file
logger.close()

print("Logger test completed!")
print(f"CSV saved at: {OUTPUT_FILE}")
