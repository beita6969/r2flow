#!/bin/bash
source "$(dirname "$0")/env.sh"
PORT=${R2FLOW_EXECUTOR_PORT:-18601}
GPU=${R2FLOW_EXECUTOR_GPU:-0}
FRACTION=${MEM_FRACTION:-0.78}
ATTENTION=${R2FLOW_ATTENTION_BACKEND:-flashinfer}
if [ "${R2FLOW_EXECUTOR_HOST:-loopback}" = node ]; then BIND=0.0.0.0; ENDPOINT_HOST=$(hostname -s); else BIND=127.0.0.1; ENDPOINT_HOST=127.0.0.1; fi
mkdir -p "$R2FLOW_RUNS/executors"
LOG=$R2FLOW_LOGS/executor_$PORT.log
PIDF=$R2FLOW_LOGS/executor_$PORT.pid
RECORD=$R2FLOW_RUNS/executors/executor-$PORT.json
case "${1:-status}" in
  start)
    UUID=$(gpu_uuid "$GPU")
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      printf '{"endpoint": "http://%s:%s", "gpu_uuid": "%s", "port": %s, "pid": %s}\n' "$ENDPOINT_HOST" "$PORT" "$UUID" "$PORT" "$(cat "$PIDF")" > "$RECORD"
      echo "executor already running: $(cat "$PIDF")"; exit 0
    fi
    [ -n "$UUID" ] || { echo "no GPU $GPU" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES=$UUID setsid nohup "$R2FLOW_SGLANG_PY" -u -m skillev.runtime.sglang_server \
      --model-path "$R2FLOW_MODEL" --tokenizer-path "$R2FLOW_TOKENIZER" --served-model-name "$R2FLOW_SERVED_MODEL" \
      --host "$BIND" --port "$PORT" --context-length 81920 --tp-size 1 --dtype bfloat16 --random-seed 0 \
      --sampling-backend pytorch --enable-deterministic-inference --trust-remote-code \
      --enable-lora --max-lora-rank 4 --lora-target-modules all --max-loras-per-batch 2 --max-loaded-loras 3 \
      --max-running-requests 32 --cuda-graph-backend-decode full --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 32 \
      --chunked-prefill-size 4096 --max-prefill-tokens 8192 --mem-fraction-static "$FRACTION" \
      --attention-backend "$ATTENTION" --linear-attn-backend triton --grammar-backend xgrammar --reasoning-parser qwen3 \
      --mamba-radix-cache-strategy extra_buffer --page-size 64 --skillev-fp32-mamba-checkpoints --enable-metrics \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    printf '{"endpoint": "http://%s:%s", "gpu_uuid": "%s", "port": %s, "pid": %s}\n' "$ENDPOINT_HOST" "$PORT" "$UUID" "$PORT" "$(cat "$PIDF")" > "$RECORD"
    echo "started $(cat "$PIDF") on GPU $GPU ($UUID), port $PORT; log $LOG; record $RECORD";;
  stop)
    P=$(cat "$PIDF" 2>/dev/null); [ -n "$P" ] || { echo "no pid file"; exit 0; }
    tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "skillev.runtime.sglang_server" && { kill "$P"; echo "stopped $P"; } || echo "pid $P is not this executor"
    rm -f "$PIDF" "$RECORD";;
  status)
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && echo "healthy" || echo "not healthy";;
  *) echo "usage: $0 start|stop|status" >&2; exit 1;;
esac
