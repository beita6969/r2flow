from __future__ import annotations

from collections.abc import Mapping


def canonical_source_key(
    source: tuple[str, str], aliases: Mapping[str, Mapping[str, str]] | None = None
) -> tuple[str, str]:
    domain, identity = source
    return domain, (aliases or {}).get(domain, {}).get(identity, identity)
