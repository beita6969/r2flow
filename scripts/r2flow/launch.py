import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ.get("R2FLOW_HOME", HERE.parents[1])).resolve()
SECRET_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "R2FLOW_JUDGE_API_KEY",
    "R2FLOW_AUTHOR_API_BASE",
)

stop_requested = False


def say(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def write_private(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} differs from the value this run started with")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def gpu_uuids(indices):
    out = []
    for index in indices:
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", index, "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if not uuid.startswith("GPU-"):
            raise SystemExit(f"no GPU {index}")
        out.append(uuid)
    return out


def run_config(source, run_dir):
    text = source.read_text(encoding="utf-8")
    values = {
        "author_base_file": str(run_dir / "author" / "api-base"),
        "author_key_file": os.environ["R2FLOW_AUTHOR_API_KEY_FILE"],
        "author_model": os.environ["R2FLOW_AUTHOR_API_MODEL"],
        "reference_verifier_model": os.environ["R2FLOW_AUTHOR_API_MODEL"],
    }
    for key, value in values.items():
        pattern = re.compile(rf"(?m)^(\s*{key}:).*$")
        if len(pattern.findall(text)) != 1:
            raise SystemExit(f"{source}: expected one {key} entry")
        text = pattern.sub(lambda m, v=value: f"{m.group(1)} {v}", text)
    return text


def checkpoints(run_root):
    found = []
    directory = run_root / "checkpoints"
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        match = re.search(r"(?:^|-)step-(\d{8})$", path.name)
        if match and not path.name.startswith(".") and (path / "COMPLETE").exists():
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: (item[0], "paused" in item[1].name))


def resume_point(run_root):
    if not run_root.exists():
        return None
    complete = checkpoints(run_root)
    paused_file = run_root / "paused.json"
    paused = json.loads(paused_file.read_text()) if paused_file.exists() else None
    pick = Path(paused["checkpoint"]) if paused else None
    if complete and (pick is None or complete[-1][0] > paused["optimizer_step"]):
        pick = complete[-1][1]
    if pick is None or not (pick / "COMPLETE").exists():
        raise SystemExit(f"{run_root} exists but holds no complete checkpoint to resume from")
    stop = run_root / "STOP_AFTER_CHECKPOINT"
    if stop.exists():
        stop.rename(run_root / f"STOP_AFTER_CHECKPOINT.acknowledged-{int(time.time())}")
    return pick


def request_stop(run_root):
    global stop_requested
    stop_requested = True
    if run_root.exists():
        (run_root / "STOP_AFTER_CHECKPOINT").touch()


def main():
    parser = argparse.ArgumentParser(description="Launch (or resume) one R2 Flow training run.")
    parser.add_argument("--run-dir", type=Path, help="default: $R2FLOW_RUNS/r2flow_<stamp>")
    parser.add_argument("--pause-at-step", type=int)
    args = parser.parse_args()
    runs = Path(os.environ.get("R2FLOW_RUNS", HOME / "runs"))
    run_dir = (args.run_dir or runs / time.strftime("r2flow_%Y%m%d-%H%M%S")).resolve()
    inputs = Path(os.environ.get("R2FLOW_INPUTS", HOME / "data" / "inputs"))
    template = Path(os.environ.get("R2FLOW_BINDINGS_TEMPLATE", inputs / "bindings-valpool.json"))
    data = Path(os.environ.get("R2FLOW_TRAIN_DATA", HOME / "data" / "r2flow" / "train"))
    sources = Path(os.environ.get("R2FLOW_TRAINING_SOURCES", data / "training-sources.json"))
    heldout = Path(os.environ.get("R2FLOW_VQ_HELDOUT", data / "vq-heldout.json"))
    pool = Path(os.environ.get("R2FLOW_VALIDATION_POOL", data / "validation-pool.json"))
    for path in (template, sources, heldout, pool):
        if not path.is_file():
            raise SystemExit(f"missing input {path}")
    config_source = Path(os.environ["R2FLOW_CONFIG"]).resolve()
    run_root = run_dir / "run"
    os.umask(0o077)
    write_private(
        run_dir / "author" / "api-base", os.environ["R2FLOW_AUTHOR_API_BASE"].strip() + "\n"
    )
    config = run_dir / "config" / config_source.name
    write_private(config, run_config(config_source, run_dir))
    gradient = gpu_uuids(os.environ["R2FLOW_TRAIN_GPUS"].split(","))
    records = [r for r in os.environ["R2FLOW_EXECUTOR_RECORDS"].split(",") if r]
    segment = run_dir / "segments" / time.strftime("%Y%m%d-%H%M%S")
    segment.mkdir(parents=True)
    bindings = segment / "bindings.json"
    subprocess.check_call(
        [
            sys.executable,
            str(HERE / "make_bindings.py"),
            "--template",
            str(template),
            "--config",
            str(config),
            "--out",
            str(bindings),
            "--evidence-root",
            str(run_dir / "evidence-mirror"),
            "--gradient",
            *gradient,
        ]
        + [item for record in records for item in ("--executor-record", record)]
    )
    environment = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    environment.update(
        PYTHONPATH=str(HOME / "src"),
        PYTHONFAULTHANDLER="1",
        CUDA_VISIBLE_DEVICES=",".join(gradient),
    )
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(HOME), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        environment["SKILLEV_CODE_REVISION"] = f"r2flow-{revision}"
    except (OSError, subprocess.CalledProcessError):
        pass
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(name, lambda *_: request_stop(run_root))
    attempt = 0
    while not stop_requested:
        attempt += 1
        resume = resume_point(run_root)
        command = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={len(gradient)}",
            str(HERE / "train_rank.py"),
            "--config",
            str(config),
            "--bindings",
            str(bindings),
            "--run-root",
            str(run_root),
            "--training-sources",
            str(sources),
            "--vq-heldout",
            str(heldout),
            "--validation-pool",
            str(pool),
        ]
        if args.pause_at_step:
            command += ["--pause-at-step", str(args.pause_at_step)]
        if resume is not None:
            command += ["--resume", str(resume)]
        started = time.time()
        log = segment / f"training-{attempt:02d}.log"
        say(f"attempt {attempt}: {'resume ' + str(resume) if resume else 'fresh'}; log {log}")
        with open(log, "x") as handle:
            child = subprocess.Popen(
                command,
                cwd=HOME,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while child.poll() is None:
                time.sleep(10)
        if child.returncode != 0:
            say(f"training exited with {child.returncode}; see {log}")
            return child.returncode
        summary_file = run_root / "summary.json"
        if summary_file.exists() and summary_file.stat().st_mtime >= started:
            say(f"training ended: {json.loads(summary_file.read_text()).get('status')}")
            return 0
        paused_file = run_root / "paused.json"
        if not paused_file.exists() or paused_file.stat().st_mtime < started:
            say("training ended without a summary or a pause marker")
            return 0
        reason = json.loads(paused_file.read_text()).get("reason")
        say(f"paused ({reason})")
        if reason != "vq-phase-boundary" or stop_requested:
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
