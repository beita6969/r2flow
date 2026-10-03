from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MARKER = "Your task is to:"
MODES = {"train": "data_path", "eval_out_of_distribution": "eval_ood_data_path"}


def goal(job: tuple[dict, str, str, int]) -> tuple[str, str]:
    config, game_file, mode, seed = job
    from alfworld.agents.environment import get_environment

    config = copy.deepcopy(config)
    config["dataset"][MODES[mode]] = os.path.dirname(game_file)
    config["dataset"]["num_train_games"] = -1
    config["dataset"]["num_eval_games"] = -1
    config["general"]["random_seed"] = seed
    builder = get_environment("AlfredTWEnv")(config, train_eval=mode)
    if [os.path.realpath(p) for p in builder.game_files] != [os.path.realpath(game_file)]:
        raise ValueError(f"{game_file}: ALFWorld bound another game")
    env = builder.init_env(batch_size=1)
    observations, _ = env.reset()
    env.close()
    lines = [
        line.strip() for line in observations[0].splitlines() if line.strip().startswith(MARKER)
    ]
    if len(lines) != 1:
        raise ValueError(f"{game_file}: no unique task goal")
    return game_file[game_file.index("json_2.1.1/") :], lines[0][len(MARKER) :].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    jobs = []
    for line in args.games.read_text(encoding="utf-8").splitlines():
        if line.strip():
            game_file, mode, seed = line.split("\t")
            jobs.append((config, game_file, mode, int(seed)))
    with ProcessPoolExecutor(args.workers) as pool:
        goals = dict(pool.map(goal, jobs))
    args.out.write_text(json.dumps(goals, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"goals\t{len(goals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
