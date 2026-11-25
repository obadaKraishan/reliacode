"""Record the HTTP requests the default recipe sends, byte for byte, with reliacode as it was
before the one-call-per-unit recipe and the extra request fields were added.

The output, ``tests/fixtures/default_requests.json``, was made once with the source of commit
60410f5, and must never be made again with a later version. To check it, run the script on that
commit's source in a scratch folder (nothing is installed and no request leaves the machine):

    git archive 60410f5 src | tar -x -C /tmp/reliacode-60410f5
    PYTHONPATH=/tmp/reliacode-60410f5/src uv run python tests/reference/make_request_reference.py

For both example codebooks it runs ``code()`` through ``OpenAICompatibleBackend`` against a mock
transport, once with the default settings and once with a seed, a token limit and structured
output turned off. The first answer for each variable cannot be used, so a second ask is
recorded too. ``tests/test_default_prompts.py`` sends the same requests and compares the bytes.
The texts and answers are invented.
"""

import inspect
import json
import subprocess
from pathlib import Path

import httpx

from reliacode import OpenAICompatibleBackend, code, load_codebook

ROOT = Path(__file__).parent.parent.parent
OUTPUT = ROOT / "tests" / "fixtures" / "default_requests.json"
FILES = ["examples/codebooks/incivility.yaml", "examples/codebooks/news_frames.yaml"]
TEXTS = {
    "c1": "Thanks for the clear summary of the budget vote.",
    "c2": 'The mayor said "no comment" again, and the ferry is still late.',
}
SETTINGS = {
    "default": {},
    "seed, token limit, no structured output": {
        "seed": 7,
        "max_tokens": 64,
        "structured_output": False,
    },
}


def first_code(body: dict) -> str:
    system = body["messages"][0]["content"]
    return system.split("where the code is one of: ")[-1].split(",")[0].split(".")[0]


def answer(request: httpx.Request) -> httpx.Response:
    """Unusable at first, then the variable's first code."""
    body = json.loads(request.content)
    unusable = len(body["messages"]) == 2
    text = "I cannot tell." if unusable else json.dumps({"code": first_code(body)})
    message = {"role": "assistant", "content": text}
    completion = {
        "id": "chatcmpl-reference",
        "object": "chat.completion",
        "model": "reference-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }
    return httpx.Response(200, json=completion)


def main() -> None:
    if "extra_body" in inspect.signature(OpenAICompatibleBackend).parameters:
        raise SystemExit(
            "Run this on the source of 60410f5 (see the docstring), not on a later one."
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    runs = []
    for path in FILES:
        codebook = load_codebook(ROOT / path)
        for label, settings in SETTINGS.items():
            sent: list[str] = []

            def server(request: httpx.Request, sent: list[str] = sent) -> httpx.Response:
                sent.append(request.content.decode("utf-8"))
                return answer(request)

            backend = OpenAICompatibleBackend(
                "http://localhost:11434/v1",
                "reference-model",
                transport=httpx.MockTransport(server),
                **settings,
            )
            run = code(codebook, TEXTS, backend=backend)
            assert run.missing() == (), path
            runs.append({"file": path, "settings": label, "requests": sent})
    data = {
        "about": (
            "The HTTP request bodies of the default recipe, sent by reliacode before the "
            "one-call-per-unit recipe and the extra request fields were added. Never regenerate."
        ),
        "made_with": {
            "source_commit": "60410f5e66993eab5de986f812dc679e88ce5c22",
            "checked_out_commit": commit,
            "httpx": httpx.__version__,
        },
        "texts": TEXTS,
        "settings": SETTINGS,
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {sum(len(r['requests']) for r in runs)} requests")


if __name__ == "__main__":
    main()
