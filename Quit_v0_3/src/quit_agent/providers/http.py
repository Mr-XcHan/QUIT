from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504, 524}


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 60,
    max_retries: int = 2,
) -> dict[str, Any]:
    """POST JSON using the Python standard library."""

    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", "unknown")
                headers_obj = getattr(response, "headers", None)
                content_type = headers_obj.get("Content-Type", "") if headers_obj else ""
                raw = response.read().decode("utf-8")
            if "text/event-stream" in content_type.lower():
                return _parse_chat_completion_stream(raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                preview = raw[:500] if raw else "<empty response>"
                raise RuntimeError(
                    "HTTP provider returned non-JSON response: "
                    f"status={status}, content_type={content_type}, body={preview}"
                ) from exc
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP provider request failed: {exc.code} {detail}")
            if exc.code not in _RETRYABLE_HTTP_STATUS or attempt >= max_retries:
                raise last_error from exc
        except URLError as exc:
            last_error = RuntimeError(f"HTTP provider request failed: {exc.reason}")
            if attempt >= max_retries:
                raise last_error from exc
        time.sleep(min(2**attempt, 8))
    if last_error is not None:
        raise last_error
    raise RuntimeError("HTTP provider request failed")


def _parse_chat_completion_stream(raw: str) -> dict[str, Any]:
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    last_event: dict[str, Any] = {}
    finish_reason: str | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HTTP provider returned malformed event-stream data: {payload[:500]}") from exc
        last_event = event
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            piece = delta.get("content") or message.get("content") or choice.get("text")
            if piece:
                content_parts.append(piece)

    if not content_parts and not last_event:
        raise RuntimeError("HTTP provider returned empty event-stream response")

    response = dict(last_event)
    response["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "".join(content_parts)},
            "finish_reason": finish_reason,
        }
    ]
    if usage is not None:
        response["usage"] = usage
    return response
