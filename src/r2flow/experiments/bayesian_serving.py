from __future__ import annotations

import json
from pathlib import Path

from skillev.contracts import JsonValue
from skillev.runtime.service_topology import ServiceTopology
from skillev.runtime.serving_profile import require_same_profile, require_training_service
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig
from skillev.runtime.sglang_pool import SGLangActorPool
from skillev.training.inflight import durable_json


def register_serving(
    gateway: SGLangGateway,
    topology: ServiceTopology,
    *,
    root: Path,
    model_path: str,
    tokenizer_path: str,
    minimum_context: int,
    event_grammar: bool = False,
) -> dict[str, JsonValue]:
    actors = gateway.members if isinstance(gateway, SGLangActorPool) else (gateway,)
    actor_map = {v.config.api_root: v for v in actors}
    profiles: dict[str, JsonValue] = {}
    actor_profile: dict[str, JsonValue] | None = None
    for service in topology.services:
        endpoint = service.endpoint.rstrip("/").removesuffix("/v1")
        member = actor_map.get(endpoint)
        if member is None:
            member = SGLangGateway(
                SGLangGatewayConfig(
                    endpoint,
                    gateway.config.base_model,
                    gateway.config.supervisor_adapter,
                    control_retries=0,
                )
            )
        actor_grammar = event_grammar and endpoint in actor_map
        actual = (
            member.read_serving_profile(event_grammar=True)
            if actor_grammar
            else member.read_serving_profile()
        )
        require_training_service(
            actual,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            base_model=gateway.config.base_model,
            minimum_context=minimum_context,
            actor=endpoint in actor_map,
            event_grammar=actor_grammar,
        )
        if endpoint in actor_map:
            if actor_profile is not None:
                require_same_profile(actor_profile, actual)
            actor_profile = actual
        profiles[endpoint] = actual
    path = root / "serving-runtime.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    for endpoint, profile in profiles.items():
        if endpoint in previous and previous[endpoint] != profile:
            raise ValueError("same-run serving profile changed; declare and qualify the transition")
    durable_json(path, {**previous, **profiles})
    for endpoint, member in actor_map.items():
        profile = profiles[endpoint]
        if not isinstance(profile, dict):
            raise TypeError("registered serving profile must be an object")
        member.bind_serving_profile(profile)
    return profiles
