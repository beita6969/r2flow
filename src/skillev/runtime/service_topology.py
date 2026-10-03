from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from skillev.contracts import JsonValue, normalize_json

from .serving_cost import ACTOR_ROUTING_POLICIES, FIXED_BENCHMARK


@dataclass(frozen=True, slots=True)
class InferenceService:
    service_id: str
    endpoint: str
    gpu_uuid: str
    request_capacity: int = 8
    token_capacity: int | None = None

    def __post_init__(self) -> None:
        if type(self.request_capacity) is not int or self.request_capacity < 1:
            raise ValueError("request capacity must be positive")
        if self.token_capacity is not None and (
            type(self.token_capacity) is not int or self.token_capacity < 1
        ):
            raise ValueError("token capacity must be positive")
        parsed = urlsplit(self.endpoint)
        if not self.service_id or not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("service identity and physical GPU UUID are required")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("service endpoint must be an HTTP URL without credentials")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "service_id": self.service_id,
            "endpoint": self.endpoint,
            "gpu_uuid": self.gpu_uuid,
            "request_capacity": self.request_capacity,
            "token_capacity": self.token_capacity,
        }

    @classmethod
    def from_value(cls, value: object) -> InferenceService:
        if (
            not isinstance(value, dict)
            or not {"service_id", "endpoint", "gpu_uuid"} <= set(value)
            or set(value)
            - {"service_id", "endpoint", "gpu_uuid", "request_capacity", "token_capacity"}
        ):
            raise ValueError("incompatible service placement")
        if any(not isinstance(value[k], str) for k in ("service_id", "endpoint", "gpu_uuid")):
            raise TypeError("service placement values must be text")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ServiceTopology:
    services: tuple[InferenceService, ...]
    actor_pool: tuple[str, ...]
    judge_pool: tuple[str, ...]
    author_pool: tuple[str, ...]
    gradient_workers: tuple[str, ...]
    actor_benchmark_routes: tuple[tuple[str, str], ...] = ()
    actor_benchmark_pools: tuple[tuple[str, tuple[str, ...]], ...] = ()
    actor_routing_policy: str = FIXED_BENCHMARK
    actor_transport_isolation: str = "shared"
    actor_balanced_benchmarks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.actor_routing_policy not in ACTOR_ROUTING_POLICIES:
            raise ValueError("unknown actor routing policy")
        if self.actor_transport_isolation not in ("shared", "endpoint-partitioned"):
            raise ValueError("unknown actor transport isolation")
        if len(set(self.actor_balanced_benchmarks)) != len(self.actor_balanced_benchmarks) or any(
            not v.strip() for v in self.actor_balanced_benchmarks
        ):
            raise ValueError("balanced benchmarks must be distinct nonempty domains")
        ids = [v.service_id for v in self.services]
        endpoints = [v.endpoint.rstrip("/").removesuffix("/v1") for v in self.services]
        devices = [v.gpu_uuid for v in self.services] + list(self.gradient_workers)
        if not ids or len(ids) != len(set(ids)) or len(endpoints) != len(set(endpoints)):
            raise ValueError("service identities and endpoints must be unique")
        if len(devices) != len(set(devices)):
            raise ValueError("different physical workers cannot own the same GPU")
        if len(self.gradient_workers) < 2 or any(
            not v.startswith("GPU-") for v in self.gradient_workers
        ):
            raise ValueError("formal execution requires at least two participating gradient GPUs")
        used: set[str] = set()
        for pool in (self.actor_pool, self.judge_pool, self.author_pool):
            if not pool or len(pool) != len(set(pool)) or not set(pool) <= set(ids):
                raise ValueError("each role requires distinct registered service references")
            used.update(pool)
        if used != set(ids):
            raise ValueError("every allocated service must have an explicit role")
        routes = dict(self.actor_benchmark_routes)
        benchmark_pools = dict(self.actor_benchmark_pools)
        if len(routes) != len(self.actor_benchmark_routes) or any(
            not domain.strip() or service not in self.actor_pool
            for domain, service in self.actor_benchmark_routes
        ):
            raise ValueError("benchmark routes require unique domains and registered actors")
        if len(benchmark_pools) != len(self.actor_benchmark_pools) or any(
            not domain.strip()
            or not services
            or len(services) != len(set(services))
            or any(service not in self.actor_pool for service in services)
            for domain, services in self.actor_benchmark_pools
        ):
            raise ValueError("benchmark pools require unique domains and registered actors")
        if any(
            domain in benchmark_pools and service not in benchmark_pools[domain]
            for domain, service in routes.items()
        ):
            raise ValueError("preferred benchmark route must belong to its actor pool")

    def members(self, role: str) -> tuple[InferenceService, ...]:
        pools = {"actor": self.actor_pool, "judge": self.judge_pool, "author": self.author_pool}
        catalog = {v.service_id: v for v in self.services}
        return tuple(catalog[key] for key in pools[role])

    def require_device_mapping(self, visible: str, world_size: int) -> None:
        if visible.split(",") != list(self.gradient_workers) or world_size != len(
            self.gradient_workers
        ):
            raise ValueError(
                "torchrun visibility/order must match the participating gradient ranks"
            )

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "format": "skillev-service-topology@1",
            "services": [v.to_value() for v in self.services],
            "actor_pool": list(self.actor_pool),
            "judge_pool": list(self.judge_pool),
            "author_pool": list(self.author_pool),
            "gradient_workers": list(self.gradient_workers),
        }
        if self.actor_benchmark_routes:
            value["actor_benchmark_routes"] = dict(self.actor_benchmark_routes)
        if self.actor_benchmark_pools:
            value["actor_benchmark_pools"] = {
                domain: list(services) for domain, services in self.actor_benchmark_pools
            }
        if self.actor_routing_policy != FIXED_BENCHMARK:
            value["actor_routing_policy"] = self.actor_routing_policy
        if self.actor_transport_isolation != "shared":
            value["actor_transport_isolation"] = self.actor_transport_isolation
        if self.actor_balanced_benchmarks:
            value["actor_balanced_benchmarks"] = list(self.actor_balanced_benchmarks)
        return value

    @classmethod
    def from_value(cls, value: object) -> ServiceTopology:
        data = normalize_json(value)
        if (
            not isinstance(data, dict)
            or set(data)
            - {
                "actor_benchmark_routes",
                "actor_benchmark_pools",
                "actor_routing_policy",
                "actor_transport_isolation",
                "actor_balanced_benchmarks",
            }
            != {"format", "services", "actor_pool", "judge_pool", "author_pool", "gradient_workers"}
            or data["format"] != "skillev-service-topology@1"
        ):
            raise ValueError("incompatible service topology")
        services = data["services"]
        if not isinstance(services, list):
            raise TypeError("service catalog must be a list")
        pools = []
        for name in ("actor_pool", "judge_pool", "author_pool", "gradient_workers"):
            pool = data[name]
            if not isinstance(pool, list) or any(not isinstance(v, str) for v in pool):
                raise TypeError("role membership must be a list of identifiers")
            pools.append(tuple(pool))
        routes = data.get("actor_benchmark_routes", {})
        if not isinstance(routes, dict) or any(not isinstance(v, str) for v in routes.values()):
            raise TypeError("benchmark routes must map domains to service identifiers")
        benchmark_pools = data.get("actor_benchmark_pools", {})
        if not isinstance(benchmark_pools, dict) or any(
            not isinstance(domain, str)
            or not isinstance(services, list)
            or any(not isinstance(service, str) for service in services)
            for domain, services in benchmark_pools.items()
        ):
            raise TypeError("benchmark pools must map domains to service identifier lists")
        routing = data.get("actor_routing_policy", FIXED_BENCHMARK)
        isolation = data.get("actor_transport_isolation", "shared")
        balanced = data.get("actor_balanced_benchmarks", [])
        if not isinstance(balanced, list) or any(not isinstance(v, str) for v in balanced):
            raise TypeError("balanced benchmarks must be domain names")
        if not isinstance(routing, str) or not isinstance(isolation, str):
            raise TypeError("actor execution modes must be text")
        return cls(
            tuple(InferenceService.from_value(v) for v in services),
            actor_pool=pools[0],
            judge_pool=pools[1],
            author_pool=pools[2],
            gradient_workers=pools[3],
            actor_benchmark_routes=tuple(routes.items()),
            actor_benchmark_pools=tuple(
                (domain, tuple(services)) for domain, services in benchmark_pools.items()
            ),
            actor_routing_policy=routing,
            actor_transport_isolation=isolation,
            actor_balanced_benchmarks=tuple(balanced),
        )
