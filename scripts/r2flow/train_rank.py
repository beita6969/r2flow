import os
import sys

import torch


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != int(os.environ["WORLD_SIZE"]):
        raise SystemExit("CUDA_VISIBLE_DEVICES must name one GPU per training rank")
    torch.cuda.set_device(local_rank)
    fraction = os.environ.get("R2FLOW_GPU_MEMORY_FRACTION")
    if fraction:
        torch.cuda.set_per_process_memory_fraction(float(fraction), local_rank)
    from r2flow.experiments.bayesian_training_cli import main as train

    train()


if __name__ == "__main__":
    sys.exit(main())
