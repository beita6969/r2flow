import argparse
import json
import os
import sys

import yaml

PLACEMENT_KEYS = (
    "endpoint",
    "serving_gpu_uuid",
    "training_gpu_uuids",
    "topology",
    "evidence_mirror_root",
)
REQUIRED_KEYS = (
    "preparation",
    "dataset",
    "deployments",
    "evalplus_python",
    "evalplus_source_root",
    "base_model",
    "adapter_namespace",
    "data_condition",
)


def main():
    parser = argparse.ArgumentParser(
        description="Write the training bindings: the template's inputs plus this placement."
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--gradient", nargs="+", required=True, metavar="GPU_UUID")
    parser.add_argument("--executor-record", action="append", required=True)
    parser.add_argument("--request-capacity", type=int, default=32)
    args = parser.parse_args()
    with open(args.template, encoding="utf-8") as handle:
        value = json.load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in value]
    if missing:
        sys.exit(f"the bindings template lacks {missing}")
    with open(args.config, encoding="utf-8") as handle:
        domains = list(yaml.safe_load(handle)["domains"])
    records = []
    for path in args.executor_record:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    if not 1 <= len(records) <= 3:
        sys.exit("the training topology takes 1 to 3 executor replicas")
    if len(args.gradient) != 2:
        sys.exit("training takes exactly two gradient GPUs (configs/r2flow/execution_2gpu.yaml)")
    names = [f"actor{i}" for i in range(len(records))]
    services = [
        {
            "service_id": name,
            "endpoint": record["endpoint"],
            "gpu_uuid": record["gpu_uuid"],
            "request_capacity": args.request_capacity,
            "token_capacity": None,
        }
        for name, record in zip(names, records, strict=True)
    ]
    topology = {
        "format": "skillev-service-topology@1",
        "services": services,
        "actor_pool": names,
        "judge_pool": names[:1],
        "author_pool": names[:1],
        "gradient_workers": list(args.gradient),
        "actor_benchmark_routes": {domain: names[0] for domain in domains},
        "actor_routing_policy": "preferred-benchmark-work-conserving",
        "actor_transport_isolation": "endpoint-partitioned",
        "actor_balanced_benchmarks": ["alfworld"] if "alfworld" in domains else [],
        "actor_benchmark_pools": {domain: list(names) for domain in domains},
    }
    bindings = {key: item for key, item in value.items() if key not in PLACEMENT_KEYS}
    bindings.update(
        endpoint=services[0]["endpoint"],
        serving_gpu_uuid=services[0]["gpu_uuid"],
        training_gpu_uuids=list(args.gradient),
        topology=topology,
        evidence_mirror_root=args.evidence_root,
    )
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(bindings, handle, indent=2)
        handle.write("\n")
    print(
        f"wrote {args.out}: {len(records)} executor replica(s), {len(args.gradient)} gradient GPUs"
    )


if __name__ == "__main__":
    main()
