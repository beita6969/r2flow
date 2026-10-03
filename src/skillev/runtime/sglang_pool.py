from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

from .serving_cost import (
    ACTOR_ROUTING_POLICIES,
    FIXED_BENCHMARK,
    PREFERRED_WORK_CONSERVING,
    EpisodeWork,
    ServingCostModel,
)
from .sglang_gateway import (
    AdapterGeneration,
    PreparedAdapterSwap,
    SGLangGateway,
    SGLangGatewayError,
    SGLangRole,
)


class SGLangActorPool(SGLangGateway):
    def __init__(
        self,
        members: Sequence[SGLangGateway],
        *,
        benchmark_routes: Mapping[str, str] | None = None,
        benchmark_pools: Mapping[str, Sequence[str]] | None = None,
        routing_policy: str = FIXED_BENCHMARK,
        balanced_benchmarks: tuple[str, ...] = (),
    ) -> None:
        if not members:
            raise ValueError("actor pool cannot be empty")
        super().__init__(members[0].config)
        self.members = tuple(members)
        for member in self.members:
            if (member.config.base_model, member.config.supervisor_adapter) != (
                self.config.base_model,
                self.config.supervisor_adapter,
            ):
                raise ValueError("actor replicas require the same model and adapter namespace")
        if len({v.config.api_root for v in self.members}) != len(self.members):
            raise ValueError("actor endpoints must be distinct")
        self.benchmark_routes = dict(benchmark_routes or {})
        self.benchmark_pools = {
            domain: tuple(endpoints) for domain, endpoints in (benchmark_pools or {}).items()
        }
        if routing_policy not in ACTOR_ROUTING_POLICIES:
            raise ValueError("unknown actor routing policy")
        self.routing_policy = routing_policy
        if len(set(balanced_benchmarks)) != len(balanced_benchmarks) or any(
            not name.strip() for name in balanced_benchmarks
        ):
            raise ValueError("balanced benchmarks must be distinct nonempty domains")
        self.balanced_benchmarks = balanced_benchmarks
        actor_endpoints = {v.config.api_root for v in self.members}
        if any(
            not domain.strip() or endpoint not in actor_endpoints
            for domain, endpoint in self.benchmark_routes.items()
        ):
            raise ValueError("benchmark route must name a registered actor endpoint")
        if any(
            not domain.strip()
            or not endpoints
            or len(endpoints) != len(set(endpoints))
            or any(endpoint not in actor_endpoints for endpoint in endpoints)
            for domain, endpoints in self.benchmark_pools.items()
        ):
            raise ValueError("benchmark pools require distinct registered actor endpoints")
        if any(
            domain in self.benchmark_pools and endpoint not in self.benchmark_pools[domain]
            for domain, endpoint in self.benchmark_routes.items()
        ):
            raise ValueError("preferred benchmark route must belong to its actor pool")
        self._pool_lock = threading.RLock()
        self._sessions: dict[str, SGLangGateway] = {}
        self._remaining_work: dict[str, int] = {}
        self._work_contexts: dict[str, EpisodeWork] = {}
        self._routing_basis: dict[str, str] = {}
        self.cost_model = ServingCostModel()
        self._pool_swaps: tuple[PreparedAdapterSwap, ...] = ()
        self._available = False
        self._publishing = False
        self._policy_snapshot_id: str | None = None
        self._previous_policy_snapshot_id: str | None = None

    @property
    def adapter_generation(self) -> AdapterGeneration:
        return self.members[0].adapter_generation

    def _require_consistent(self) -> None:
        revisions = {
            (v.adapter_generation.adapter_name, v.adapter_generation.adapter_revision)
            for v in self.members
        }
        if len(revisions) != 1 or self.adapter_generation.adapter_revision == "not-loaded":
            raise SGLangGatewayError("actor replicas do not have one published policy")

    def bind_policy_snapshot(self, policy_snapshot_id: str) -> None:
        with self._pool_lock:
            if self._publishing or self._sessions or not self._available:
                raise SGLangGatewayError("policy identity requires a fully published idle pool")
            self._require_consistent()
            self._policy_snapshot_id = policy_snapshot_id

    def acquire_episode(
        self,
        episode_id: str,
        expected_policy_snapshot_id: str | None = None,
        *,
        endpoint: str | None = None,
        estimated_work: int = 1,
        benchmark: str | None = None,
        horizon: int = 1,
        benchmark_episode_index: int | None = None,
    ) -> SGLangGateway:
        if type(estimated_work) is not int or estimated_work < 1:
            raise ValueError("episode work estimate must be positive")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("episode horizon estimate must be positive")
        with self._pool_lock:
            if not self._available or self._publishing:
                raise SGLangGatewayError("actor publication is incomplete")
            self._require_consistent()
            if (
                expected_policy_snapshot_id is not None
                and expected_policy_snapshot_id != self._policy_snapshot_id
            ):
                raise SGLangGatewayError("episode policy differs from the published actor pool")
            if episode_id in self._sessions:
                raise ValueError("episode already has an actor lease")
            restored_endpoint = endpoint is not None
            preferred = None
            balanced = benchmark in self.balanced_benchmarks
            eligible = self._eligible_members(benchmark)
            if balanced and endpoint is None:
                if type(benchmark_episode_index) is not int or benchmark_episode_index < 0:
                    raise ValueError(
                        "balanced admission needs its canonical domain episode ordinal"
                    )
                endpoint = eligible[benchmark_episode_index % len(eligible)].config.api_root
            if self.benchmark_routes:
                if benchmark not in self.benchmark_routes:
                    raise SGLangGatewayError("benchmark has no declared actor route")
                assigned = self.benchmark_routes[benchmark]
                preferred = assigned
                if self.routing_policy == FIXED_BENCHMARK and not balanced:
                    if endpoint is not None and endpoint != assigned:
                        raise SGLangGatewayError(
                            "saved episode requires an explicit route migration"
                        )
                    endpoint = assigned
            work = EpisodeWork(benchmark, horizon)
            loads, basis = self._placement_costs(work, estimated_work, members=eligible)
            member = eligible[
                min(
                    range(len(loads)),
                    key=lambda i: (
                        loads[i],
                        eligible[i].config.api_root != preferred,
                        i,
                    ),
                )
            ]
            if endpoint is not None:
                matches = [m for m in eligible if m.config.api_root == endpoint]
                if len(matches) != 1:
                    raise SGLangGatewayError(
                        "episode replica is not in the benchmark's eligible actor pool; "
                        "an explicit route migration is required"
                    )
                member = matches[0]
            member.begin_supervisor_rollout()
            self._sessions[episode_id] = member
            self._remaining_work[episode_id] = estimated_work
            self._work_contexts[episode_id] = work
            self._routing_basis[episode_id] = (
                "saved-endpoint"
                if restored_endpoint
                else "canonical-domain-round-robin"
                if balanced
                else FIXED_BENCHMARK
                if preferred and self.routing_policy == FIXED_BENCHMARK
                else basis
            )
            return member

    def _eligible_members(self, benchmark: str | None) -> tuple[SGLangGateway, ...]:
        if benchmark is None:
            return self.members
        endpoints = self.benchmark_pools.get(benchmark)
        if endpoints is None:
            return self.members
        catalog = {member.config.api_root: member for member in self.members}
        return tuple(catalog[endpoint] for endpoint in endpoints)

    def _placement_costs(
        self,
        incoming: EpisodeWork,
        proxy: int,
        *,
        members: Sequence[SGLangGateway] | None = None,
    ) -> tuple[list[float], str]:
        candidates = self.members if members is None else tuple(members)
        if self.routing_policy == PREFERRED_WORK_CONSERVING:
            estimated = []
            for member in candidates:
                endpoint = member.config.api_root
                estimates = [
                    self.cost_model.estimate(endpoint, work)
                    for episode, work in self._work_contexts.items()
                    if self._sessions[episode] is member
                ] + [self.cost_model.estimate(endpoint, incoming)]
                if any(value is None for value in estimates):
                    break
                estimated.append(sum(value for value in estimates if value is not None))
            if len(estimated) == len(candidates):
                return estimated, "observed-domain-phase-length-service-seconds"
        return [
            float(sum(self._remaining_work[k] for k, v in self._sessions.items() if v is m) + proxy)
            for m in candidates
        ], "cold-budget-token-proxy"

    def update_episode_work(
        self, episode_id: str, remaining_work: int, *, context: EpisodeWork | None = None
    ) -> None:
        if type(remaining_work) is not int or remaining_work < 1:
            raise ValueError("remaining work must be positive")
        with self._pool_lock:
            if episode_id not in self._sessions:
                raise ValueError("episode has no serving lease")
            if context is not None:
                if context.benchmark != self._work_contexts[episode_id].benchmark:
                    raise ValueError("episode benchmark cannot change")
                self._work_contexts[episode_id] = context
            self._remaining_work[episode_id] = remaining_work

    def observe_episode_phase(
        self,
        episode_id: str,
        phase: str,
        *,
        input_tokens: int,
        output_tokens: int,
        response_seconds: float,
        metrics: Mapping[str, object],
    ) -> None:
        with self._pool_lock:
            self.cost_model.observe(
                self._sessions[episode_id].config.api_root,
                self._work_contexts[episode_id].benchmark,
                phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_seconds=response_seconds,
                metrics=metrics,
            )

    def routing_observation(self, episode_id: str) -> dict[str, object]:
        with self._pool_lock:
            return {
                "actor_routing_policy": self.routing_policy,
                "actor_routing_cost_basis": self._routing_basis[episode_id],
                "actor_balanced_benchmarks": list(self.balanced_benchmarks),
                "actor_benchmark_pool": [
                    member.config.api_root
                    for member in self._eligible_members(self._work_contexts[episode_id].benchmark)
                ],
            }

    def release_episode(self, episode_id: str) -> None:
        with self._pool_lock:
            member = self._sessions.pop(episode_id)
            del self._remaining_work[episode_id]
            del self._work_contexts[episode_id]
            del self._routing_basis[episode_id]
            member.end_supervisor_rollout()

    def _begin_publication(self) -> None:
        with self._pool_lock:
            if self._publishing or self._sessions:
                raise SGLangGatewayError("cannot publish across an active episode or transaction")
            self._publishing = True
            self._available = False
            self._previous_policy_snapshot_id = self._policy_snapshot_id
            self._policy_snapshot_id = None

    def prepare_supervisor_adapter(
        self, *, adapter_path: str, adapter_revision: str
    ) -> PreparedAdapterSwap:
        self._begin_publication()
        swaps = []
        try:
            for member in self.members:
                swaps.append(
                    member.prepare_supervisor_adapter(
                        adapter_path=adapter_path, adapter_revision=adapter_revision
                    )
                )
        except Exception:
            for member, swap in reversed(list(zip(self.members, swaps, strict=False))):
                member.rollback_supervisor_adapter(swap)
            self._publishing = False
            raise
        self._pool_swaps = tuple(swaps)
        return swaps[0]

    def _require_pool_swap(self, prepared: PreparedAdapterSwap) -> None:
        if not self._pool_swaps or prepared is not self._pool_swaps[0]:
            raise ValueError("publication does not belong to this pool transaction")

    def commit_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> AdapterGeneration:
        self._require_pool_swap(prepared)
        for member, swap in zip(self.members, self._pool_swaps, strict=True):
            member.commit_supervisor_adapter(swap)
        self._require_consistent()
        self._pool_swaps = ()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    def rollback_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> None:
        self._require_pool_swap(prepared)
        for member, swap in reversed(list(zip(self.members, self._pool_swaps, strict=True))):
            member.rollback_supervisor_adapter(swap)
        self._pool_swaps = ()
        self._publishing = False
        self._policy_snapshot_id = self._previous_policy_snapshot_id
        self._available = self.adapter_generation.adapter_revision != "not-loaded"
        if self._available:
            self._require_consistent()

    def restore_supervisor_adapter(
        self, *, adapter_path: str, adapter_revision: str
    ) -> AdapterGeneration:
        self._begin_publication()
        for member in self.members:
            member.restore_supervisor_adapter(
                adapter_path=adapter_path, adapter_revision=adapter_revision
            )
        self._require_consistent()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    def bind_existing_supervisor_adapter(self, *, adapter_revision: str) -> AdapterGeneration:
        self._begin_publication()
        for member in self.members:
            member.bind_existing_supervisor_adapter(adapter_revision=adapter_revision)
        self._require_consistent()
        self._publishing = False
        self._available = True
        return self.adapter_generation

    async def health(self) -> tuple[str, ...]:
        models = [await member.health() for member in self.members]
        return tuple(sorted(set.intersection(*(set(v) for v in models))))

    def _begin_request(self, role: SGLangRole) -> None:
        if role is SGLangRole.SUPERVISOR:
            raise SGLangGatewayError("actor-pool requests require an episode lease")
        super()._begin_request(role)

    def begin_supervisor_rollout(self) -> AdapterGeneration:
        raise SGLangGatewayError("actor-pool requests require an episode lease")
