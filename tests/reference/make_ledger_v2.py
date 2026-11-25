"""Write a ledger with reliacode as it was at ledger version 2, before version 3 added the cache
counts and the batch ids.

The output, ``tests/fixtures/ledger_v2.jsonl``, was made once with the source of commit c69b4e8,
and must never be made again with a later version. To check it, run the script on that commit's
source in a scratch folder (nothing is installed and no request leaves the machine):

    git archive c69b4e8 src | tar -x -C /tmp/reliacode-c69b4e8
    PYTHONPATH=/tmp/reliacode-c69b4e8/src uv run python tests/reference/make_ledger_v2.py

It codes two invented texts with the news-frames example codebook through
``OpenAICompatibleBackend`` against a mock transport that reports reasoning tokens, with one
extra request field. The first answer for the second variable cannot be used, so a second ask
is recorded, and the last call gets a server error, so a failed line is recorded too.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from reliacode import BackendError, Ledger, OpenAICompatibleBackend, code, load_codebook
from reliacode.ledger import LEDGER_VERSION

ROOT = Path(__file__).parent.parent.parent
OUTPUT = ROOT / "tests" / "fixtures" / "ledger_v2.jsonl"
TEXTS = {
    "v2-a": "The council voted to extend the night buses to the river district.",
    "v2-b": "A storm closed the harbor, and the ferry company blamed the forecast.",
}
ANSWERS = [
    '{"code": "economic"}',
    "not json",
    '{"code": "government"}',
    '{"code": "conflict"}',
]


def main() -> None:
    assert LEDGER_VERSION == 2, "run this on the source of commit c69b4e8 only"
    answers = iter(ANSWERS)

    def answer(request: httpx.Request) -> httpx.Response:
        try:
            text = next(answers)
        except StopIteration:
            return httpx.Response(500, json={"error": {"message": "invented failure"}})
        usage = {
            "prompt_tokens": 610,
            "completion_tokens": 40,
            "total_tokens": 650,
            "completion_tokens_details": {"reasoning_tokens": 32},
        }
        body = {
            "id": "chatcmpl-v2",
            "model": "example-reasoning-model-served",
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }
        return httpx.Response(200, json=body, headers={"x-request-id": "req-v2"})

    start = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
    times = (start + timedelta(seconds=n) for n in range(100))
    OUTPUT.unlink(missing_ok=True)
    ledger = Ledger(OUTPUT, clock=lambda: next(times))
    backend = OpenAICompatibleBackend(
        "http://localhost:8000/v1",
        "example-reasoning-model",
        temperature=None,
        max_retries=0,
        extra_body={"reasoning_effort": "low"},
        transport=httpx.MockTransport(answer),
        clock=iter(float(n) for n in range(1000)).__next__,
    )
    codebook = load_codebook(ROOT / "examples" / "codebooks" / "news_frames.yaml")
    try:
        code(codebook, TEXTS, backend=backend, ledger=ledger)
    except BackendError as error:
        assert error.reason == "server_error"
    lines = [json.loads(line) for line in OUTPUT.read_text(encoding="utf-8").splitlines()]
    print(f"{len(lines)} lines, versions {sorted({line['schema_version'] for line in lines})}")


if __name__ == "__main__":
    main()
