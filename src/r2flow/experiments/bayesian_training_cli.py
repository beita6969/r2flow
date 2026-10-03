from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .training_sources import require_training_sources, training_source_condition


def main() -> None:
    from . import bayesian_improve_training as entry

    parser = argparse.ArgumentParser(
        description="Train R2 Flow from one declared run configuration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--training-sources",
        type=Path,
        required=True,
        help="Declared training-source allowlist and exclusions; repeat unchanged on resume",
    )
    parser.add_argument(
        "--vq-heldout",
        type=Path,
        help="R2 Flow V_q held-out set (vq-heldout.json); required by R2 Flow runs",
    )
    parser.add_argument(
        "--validation-pool",
        type=Path,
        help=(
            "The dedicated validation pool (validation-pool.json); required by "
            "dedicated-validation-pool@1"
        ),
    )
    parser.add_argument(
        "--pause-at-step",
        type=int,
        help="Save and pause at this committed step without changing the full run plan",
    )
    parser.add_argument("--resume", type=Path, help="Complete method snapshot from this same run")
    args = parser.parse_args()
    from .fresh_restart import load_fresh_config

    config = (
        entry.BayesianFormalConfig.load(args.config)
        if args.resume
        else load_fresh_config(args.config)
    )
    from .validation_pool import require_validation_pool_argument

    try:
        require_validation_pool_argument(config, args.validation_pool)
    except ValueError as error:
        parser.error(str(error))
    training_source_condition(
        config,
        sources=args.training_sources,
        root=args.run_root.resolve(),
        resume=args.resume,
    )
    bindings = entry.FormalTrainingBindings.load(args.bindings)
    from .autonomous_ttb import autonomous_training_sources
    from .training_domain_schedule import training_schedule

    records = autonomous_training_sources(
        config,
        entry.load_training_sources(bindings.dataset),
        bindings.data_condition,
    )
    selected = training_schedule(config, records, root=args.run_root, resume=args.resume)
    require_training_sources(
        args.training_sources,
        selected,
        expected_trajectories=len(selected),
    )
    profile = entry.TrainingPerformanceConfig.load(Path(config.performance_profile))
    entry.require_formal_execution(profile)
    bindings.require_device_mapping(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), int(os.environ.get("WORLD_SIZE", "1"))
    )
    profile.configure_process()
    from r2flow.benchmarks.healthbench_api import configure_api_capacity

    configure_api_capacity(profile.healthbench_judge_capacity)
    topology = entry.initialize_distributed_ttb(timeout_minutes=180)
    try:
        if topology.rank == 0:
            asyncio.run(
                entry.run_coordinator(
                    config=config,
                    bindings=bindings,
                    profile=profile,
                    root=args.run_root.resolve(),
                    resume=None if args.resume is None else args.resume.resolve(),
                    topology=topology,
                    training_sources=args.training_sources,
                    pause_at_step=args.pause_at_step,
                    vq_heldout=args.vq_heldout,
                    validation_pool=args.validation_pool,
                )
            )
        else:
            backbone_config, checkpoint = entry._read_preparation(bindings.preparation)
            config.require_backbone(backbone_config)
            backbone = entry.build_qwen_policy_backbone(backbone_config, performance=profile)
            backbone.load_checkpoint(checkpoint.directory)
            backbone.bind_initial_trainable_state(checkpoint.trainable_state)
            from skillev.training.flow_offsets import flow_offset_domains

            backbone.enable_flow_offsets(flow_offset_domains(config.sampling_config))
            entry.serve_distributed_ttb_worker(topology=topology, backbone=backbone)
    except BaseException:
        raise
    else:
        if entry.dist.is_initialized():
            entry.dist.destroy_process_group()
