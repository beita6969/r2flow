from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from skillev.evaluation.external_judge_policy import (
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MODEL,
)


def _judge_key() -> str:
    key = os.environ.get(EXTERNAL_JUDGE_KEY_ENV, "").strip()
    if not key and (filename := os.environ.get(f"{EXTERNAL_JUDGE_KEY_ENV}_FILE")):
        key = Path(filename).expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            "Set R2FLOW_JUDGE_API_KEY_FILE (or R2FLOW_JUDGE_API_KEY) for the judge endpoint"
        )
    return key


def _output_text(response: Any) -> str | None:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(raw, dict):
        return None
    pieces: list[str] = []
    for item in raw.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                value = content.get("text")
                if isinstance(value, str):
                    pieces.append(value)
    return "".join(pieces) or None


class ResponsesCompletionsAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def create(self, **kwargs: Any) -> Any:
        if (kwargs.get("model"), kwargs.get("reasoning_effort")) != (
            EXTERNAL_JUDGE_MODEL,
            EXTERNAL_JUDGE_EFFORT,
        ):
            raise ValueError("the Judge requires the declared medium model")
        if kwargs.get("stream") is not False or "tools" in kwargs:
            raise ValueError("external Judge requires synchronous tool-free Responses calls")
        request: dict[str, Any] = {
            "model": kwargs["model"],
            "input": kwargs["messages"],
            "max_output_tokens": kwargs["max_completion_tokens"],
            "reasoning": {"effort": kwargs["reasoning_effort"]},
            "stream": False,
            "store": False,
        }
        if "timeout" in kwargs:
            request["timeout"] = kwargs["timeout"]
        if "response_format" in kwargs:
            request["text"] = {"format": kwargs["response_format"]}
        response = self.client.responses.create(**request)
        content = _output_text(response)
        status = getattr(response, "status", None)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        normalized_usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model_dump=lambda: {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        return SimpleNamespace(
            id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, refusal=None),
                    finish_reason="stop" if status == "completed" else "length",
                )
            ],
            usage=normalized_usage if usage is not None else None,
            _request_id=getattr(response, "_request_id", None),
            model_dump=lambda **options: response.model_dump(**options)
            if callable(getattr(response, "model_dump", None))
            else None,
        )


class ResponsesCompatibilityClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = SimpleNamespace(completions=ResponsesCompletionsAdapter(client))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ResponsesCompatibilityClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def make_external_judge_client(*, recovery: Any = None) -> Any:
    from openai import DefaultHttpxClient, OpenAI

    if spool := os.environ.get("SKILLEV_JUDGE_SPOOL_DIR"):
        from skillev.evaluation.judge_spool import JudgeSpoolClient

        return JudgeSpoolClient(Path(spool), recovery=recovery)

    if not EXTERNAL_JUDGE_API_BASE or not EXTERNAL_JUDGE_MODEL:
        raise RuntimeError(
            "Set R2FLOW_JUDGE_API_BASE and R2FLOW_JUDGE_MODEL for the judge endpoint"
        )
    raw_client = OpenAI(
        api_key=_judge_key(),
        base_url=EXTERNAL_JUDGE_API_BASE,
        max_retries=0,
        timeout=120.0,
        default_headers={"User-Agent": "r2flow-judge/1.0"},
        http_client=DefaultHttpxClient(follow_redirects=False),
    )
    return ResponsesCompatibilityClient(raw_client)
