#!/bin/bash
export R2FLOW_HOME=${R2FLOW_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export R2FLOW_MODELS=${R2FLOW_MODELS:-$R2FLOW_HOME/models}
export R2FLOW_DATA=${R2FLOW_DATA:-$R2FLOW_HOME/data}
export R2FLOW_TRAIN_DATA=${R2FLOW_TRAIN_DATA:-$R2FLOW_DATA/r2flow/train}
export R2FLOW_INPUTS=${R2FLOW_INPUTS:-$R2FLOW_DATA/inputs}
export R2FLOW_RUNS=${R2FLOW_RUNS:-$R2FLOW_HOME/runs}
export R2FLOW_LOGS=${R2FLOW_LOGS:-$R2FLOW_HOME/logs}
export R2FLOW_TRAIN_PY=${R2FLOW_TRAIN_PY:-python}
export R2FLOW_SGLANG_PY=${R2FLOW_SGLANG_PY:-python}
export R2FLOW_MODEL=${R2FLOW_MODEL:-$R2FLOW_MODELS/Qwen3.5-9B}
export R2FLOW_TOKENIZER=${R2FLOW_TOKENIZER:-$R2FLOW_MODEL}
export R2FLOW_SERVED_MODEL=${R2FLOW_SERVED_MODEL:-qwen35-direct-base}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export PYTHONPATH="$R2FLOW_HOME/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$R2FLOW_RUNS" "$R2FLOW_LOGS"
gpu_uuid() { nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | head -1; }
