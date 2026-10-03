import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ.get("R2FLOW_HOME", HERE.parents[1])).resolve()
sys.path.insert(0, str(HOME / "src"))

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TOKENIZER_ID = "Qwen/Qwen3.5-9B"
REVISION = "local"


def env_path(name, default=None):
    value = os.environ.get(name)
    return Path(os.path.abspath(value)) if value else default


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hidden_size(model):
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    return int((config.get("text_config") or config)["hidden_size"])


def prepare_policy(out, model, config):
    preparation = out / "preparation.json"
    if preparation.exists():
        print(f"keep {preparation}")
        return preparation
    import torch
    from transformers import AutoTokenizer

    from r2flow.experiments.bayesian_training_setup import _PREPARATION_FORMAT
    from skillev.policy.config import QwenMultimodalBackboneConfig
    from skillev.policy.flow_head import FlowHeadSpec
    from skillev.policy.hf_backbone import build_qwen_policy_backbone
    from skillev.policy.tokenizer import qwen_tokenizer_artifact_identity
    from skillev.policy.trainable_state import PrivateInitialCheckpointBinding
    from skillev.policy.z_initialization import ZInitializationSpec
    from skillev.training.performance_config import TrainingPerformanceConfig

    method = config["r2flow"]["method"]
    tokenizer = AutoTokenizer.from_pretrained(str(model), use_fast=True, local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    backbone_config = QwenMultimodalBackboneConfig(
        base_model_path=str(model),
        revision=REVISION,
        tokenizer_id=TOKENIZER_ID,
        tokenizer_content_hash=qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer, tokenizer_id=TOKENIZER_ID, revision=REVISION
        ).content_hash,
        hidden_size=hidden_size(model),
        device="cuda",
        torch_dtype=config["base_dtype"],
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,
        lora_target_modules=LORA_TARGETS,
        z_hidden_width=32,
        eos_token_ids=(eos,),
        teacher_forced_gradient_checkpointing=True,
        z_initialization=ZInitializationSpec(
            mode="output-bias-log-epsilon@1", epsilon=method["epsilon_min"]
        ),
        flow_head=FlowHeadSpec(
            hidden_width=32, eta=method["temperature_beta"], epsilon=method["epsilon_min"]
        ),
    )
    profile = TrainingPerformanceConfig.load(Path(config["performance_profile"]).resolve())
    backbone = build_qwen_policy_backbone(backbone_config, performance=profile)
    directory = out / "initial-policy"
    backbone.save_checkpoint(str(directory))
    identity = backbone.trainable_state_identity
    torch.save(
        {
            name: value.detach().cpu().clone()
            for name, value in backbone.named_trainable_parameters().items()
        },
        out / "initial_named_parameters.pt",
    )
    backbone.load_checkpoint(str(directory))
    if backbone.trainable_state_identity != identity:
        raise SystemExit("the saved initial policy does not reload to the same state")
    write_new(
        preparation,
        {
            "backbone": backbone_config.to_value(),
            "format": _PREPARATION_FORMAT,
            "initial_checkpoint": PrivateInitialCheckpointBinding(
                directory=str(directory), trainable_state=identity
            ).to_value(),
        },
    )
    print(f"wrote {preparation}")
    return preparation


def corpus_pin(path):
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "passages": int(metadata["passages"]),
        "corpus_id": metadata["corpus_id"],
    }


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def module_root(interpreter, module):
    code = (
        f"import pathlib, {module}; print(pathlib.Path({module}.__file__).resolve().parent.parent)"
    )
    return Path(subprocess.check_output([str(interpreter), "-c", code], text=True).strip())


def prepare_deployments(out, config):
    deployments = out / "deployments.json"
    if deployments.exists():
        print(f"keep {deployments}")
        return deployments
    alfworld_python = env_path("R2FLOW_ALFWORLD_PY")
    alfworld_source = env_path("R2FLOW_ALFWORLD_SRC")
    alfworld_data = env_path("ALFWORLD_DATA")
    if not (alfworld_python and alfworld_source and alfworld_data):
        raise SystemExit(
            "set R2FLOW_ALFWORLD_PY (interpreter with alfworld), R2FLOW_ALFWORLD_SRC (git checkout "
            "of alfworld) and ALFWORLD_DATA (extracted ALFWorld data)"
        )
    simple_evals = env_path("R2FLOW_SIMPLE_EVALS")
    if simple_evals is None:
        raise SystemExit("set R2FLOW_SIMPLE_EVALS to the openai/simple-evals checkout")
    from r2flow.benchmarks.session_deployments import SESSION_DEPLOYMENTS_FORMAT

    value = {
        "format": SESSION_DEPLOYMENTS_FORMAT,
        "alfworld": {
            "interpreter": str(alfworld_python),
            "source_root": str(alfworld_source),
            "source_revision": git_head(alfworld_source),
            "config_path": str(HOME / "configs" / "alfworld" / "base_config.yaml"),
            "dataset_root": str(alfworld_data),
            "timeout_seconds": 180,
        },
        "healthbench": {"source_root": str(simple_evals)},
    }
    if "triviaqa" in config["domains"]:
        database = env_path("R2FLOW_WIKIPEDIA_DB", out / "wikipedia" / "psgs_w100.sqlite")
        passages = env_path("R2FLOW_WIKIPEDIA_PASSAGES")
        if not database.exists():
            if passages is None:
                raise SystemExit(
                    "set R2FLOW_WIKIPEDIA_PASSAGES to psgs_w100.tsv.gz (DPR Wikipedia passages)"
                )
            from r2flow.evaluation.wikipedia_corpus import build_index

            database.parent.mkdir(parents=True, exist_ok=True)
            print(f"indexed {build_index(passages, database)} passages")
        value["triviaqa_wikipedia"] = corpus_pin(database)
    write_new(deployments, value)
    print(f"wrote {deployments}")
    return deployments


def prepare_bindings(out, preparation, deployments, data):
    template = out / "bindings-valpool.json"
    if template.exists():
        print(f"keep {template}")
        return template
    evalplus_python = env_path("R2FLOW_EVALPLUS_PY")
    if evalplus_python is None:
        raise SystemExit("set R2FLOW_EVALPLUS_PY (interpreter with evalplus)")
    write_new(
        template,
        {
            "preparation": str(preparation),
            "dataset": str(data / "training.jsonl"),
            "deployments": str(deployments),
            "evalplus_python": str(evalplus_python),
            "evalplus_source_root": str(module_root(evalplus_python, "evalplus")),
            "base_model": os.environ.get("R2FLOW_SERVED_MODEL", "qwen35-direct-base"),
            "adapter_namespace": "r2flow",
            "data_condition": json.loads(
                (data / "data-condition.json").read_text(encoding="utf-8")
            ),
        },
    )
    print(f"wrote {template}")
    return template


def main():
    parser = argparse.ArgumentParser(
        description="Create the initial policy, the deployments document and the bindings template."
    )
    parser.add_argument("--config", type=Path, default=env_path("R2FLOW_CONFIG"))
    args = parser.parse_args()
    out = env_path("R2FLOW_INPUTS", HOME / "data" / "inputs")
    data = env_path("R2FLOW_TRAIN_DATA", HOME / "data" / "r2flow" / "train")
    model = env_path("R2FLOW_MODEL", HOME / "models" / "Qwen3.5-9B")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["performance_profile"] = str(HOME / config["performance_profile"])
    out.mkdir(parents=True, exist_ok=True)
    deployments = prepare_deployments(out, config)
    preparation = prepare_policy(out, model, config)
    prepare_bindings(out, preparation, deployments, data)


if __name__ == "__main__":
    main()
