from __future__ import annotations

import re


def action_payload(text: str) -> str:
    payload = text.strip()
    if payload.startswith("Action:"):
        payload = payload.removeprefix("Action:").strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", payload, re.I | re.S)
    return fence[1] if fence is not None else payload
