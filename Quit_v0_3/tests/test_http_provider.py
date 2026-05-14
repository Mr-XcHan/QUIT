from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from quit_agent.providers import http


class _Response:
    def __init__(self, body: bytes, *, content_type: str = "") -> None:
        self.body = body
        self.headers = {"Content-Type": content_type} if content_type else {}
        self.status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_post_json_retries_retryable_http_status(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                url="https://example.test/v1/chat/completions",
                code=524,
                msg="status code 524",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"openai_error"}}'),
            )
        return _Response(b'{"ok": true}')

    monkeypatch.setattr(http, "urlopen", fake_urlopen)
    monkeypatch.setattr(http.time, "sleep", lambda _seconds: None)

    result = http.post_json(url="https://example.test", payload={"prompt": "x"})

    assert result == {"ok": True}
    assert calls["count"] == 2


def test_post_json_parses_openai_event_stream(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    def fake_urlopen(_request, timeout):
        return _Response(body, content_type="text/event-stream")

    monkeypatch.setattr(http, "urlopen", fake_urlopen)

    result = http.post_json(url="https://example.test", payload={"prompt": "x"})

    assert result["choices"][0]["message"]["content"] == "hello world"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"total_tokens": 3}
