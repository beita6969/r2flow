#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
export R2FLOW_CONFIG=${R2FLOW_CONFIG:-$R2FLOW_HOME/configs/r2flow/r2flow_9b_paper.yaml}

missing=()
for name in R2FLOW_AUTHOR_API_BASE R2FLOW_AUTHOR_API_KEY_FILE R2FLOW_AUTHOR_API_MODEL R2FLOW_TRAIN_GPUS; do
  [ -n "${!name:-}" ] || missing+=("$name")
done
[ -n "${R2FLOW_EXECUTOR_GPUS:-}${R2FLOW_EXECUTOR_RECORDS:-}" ] || missing+=("R2FLOW_EXECUTOR_GPUS")
for name in R2FLOW_JUDGE_API_BASE R2FLOW_JUDGE_API_KEY_FILE R2FLOW_JUDGE_MODEL R2FLOW_SIMPLE_EVALS; do
  [ -n "${!name:-}" ] || missing+=("$name")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Not configured: ${missing[*]}" >&2
  echo "  R2FLOW_TRAIN_GPUS           two GPU indices for the gradient ranks, e.g. 0,1" >&2
  echo "  R2FLOW_EXECUTOR_GPUS        one to three GPU indices for the executor replicas, e.g. 2,3" >&2
  echo "  R2FLOW_AUTHOR_API_BASE      base URL of an OpenAI Responses-compatible endpoint (skill author and reference verifier)" >&2
  echo "  R2FLOW_AUTHOR_API_KEY_FILE  absolute path of the file that contains its API key" >&2
  echo "  R2FLOW_AUTHOR_API_MODEL     model name of the skill author" >&2
  echo "  R2FLOW_JUDGE_API_BASE / R2FLOW_JUDGE_API_KEY_FILE / R2FLOW_JUDGE_MODEL   HealthBench judge" >&2
  echo "  R2FLOW_SIMPLE_EVALS         openai/simple-evals checkout (HealthBench source)" >&2
  exit 1
fi
for file in "$R2FLOW_AUTHOR_API_KEY_FILE" ${R2FLOW_JUDGE_API_KEY_FILE:+"$R2FLOW_JUDGE_API_KEY_FILE"}; do
  case "$file" in /*) ;; *) echo "key files must be absolute paths: $file" >&2; exit 1;; esac
  [ -s "$file" ] || { echo "key file is empty or missing: $file" >&2; exit 1; }
done
cd "$R2FLOW_HOME"

if [ ! -f "$R2FLOW_INPUTS/bindings-valpool.json" ]; then
  FIRST_GPU=${R2FLOW_TRAIN_GPUS%%,*}
  CUDA_VISIBLE_DEVICES=$(gpu_uuid "$FIRST_GPU") "$R2FLOW_TRAIN_PY" -u "$HERE/prepare.py"
fi

if [ -z "${R2FLOW_EXECUTOR_RECORDS:-}" ]; then
  port=18601
  records=()
  for gpu in ${R2FLOW_EXECUTOR_GPUS//,/ }; do
    R2FLOW_EXECUTOR_GPU=$gpu R2FLOW_EXECUTOR_PORT=$port "$HERE/serve_executor.sh" start
    records+=("$R2FLOW_RUNS/executors/executor-$port.json")
    port=$((port + 1))
  done
  R2FLOW_EXECUTOR_RECORDS=$(IFS=,; echo "${records[*]}")
  export R2FLOW_EXECUTOR_RECORDS
fi
for record in ${R2FLOW_EXECUTOR_RECORDS//,/ }; do
  url=$("$R2FLOW_TRAIN_PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["endpoint"])' "$record")
  for _ in $(seq 360); do curl -sf -m 10 "$url/health" > /dev/null && break || sleep 5; done
  curl -sf -m 10 "$url/health" > /dev/null || { echo "executor not healthy at $url; see $R2FLOW_LOGS" >&2; exit 1; }
done
exec "$R2FLOW_TRAIN_PY" -u "$HERE/launch.py" "$@"
