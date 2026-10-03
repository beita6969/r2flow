from skillev.rollout import ExternalSGLangRolloutConfig


def formal_actor_transport(endpoint: str, *, worker_threads: int) -> ExternalSGLangRolloutConfig:
    return ExternalSGLangRolloutConfig(
        endpoint_base=endpoint,
        request_timeout_seconds=1800.0,
        transport_worker_threads=worker_threads,
    )
