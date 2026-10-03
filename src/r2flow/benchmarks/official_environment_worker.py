#!/usr/bin/env python3

import argparse
import copy
import importlib
import json
import os
import sys

if __package__:
    from .alfworld_public_goal import reset_public_goal
else:
    from alfworld_public_goal import reset_public_goal

PROTOCOL_VERSION = "skillev-official-environment-worker@1"
MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def require_object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("{} has an incompatible field set".format(label))
    return value


def require_text(value, label, allow_empty=False):
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("{} must be text without NUL".format(label))
    if not allow_empty and not value.strip():
        raise ValueError("{} cannot be empty".format(label))
    return value


def require_int(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} has an invalid integer".format(label))
    return value


def require_bool(value, label):
    if not isinstance(value, bool):
        raise TypeError("{} must be boolean".format(label))
    return value


def require_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be numeric".format(label))
    return float(value)


def require_sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise TypeError("{} must be a sequence".format(label))
    return value


def require_singleton(value, label):
    sequence = require_sequence(value, label)
    if len(sequence) != 1:
        raise ValueError("{} must contain exactly one item".format(label))
    return sequence[0]


def require_module_in_source(module, source_root):
    module_file = getattr(module, "__file__", None)
    require_text(module_file, "official module file")
    resolved_module = os.path.realpath(module_file)
    resolved_source = os.path.realpath(source_root)
    if os.path.commonpath([resolved_module, resolved_source]) != resolved_source:
        raise ValueError("official module was not imported from the pinned source root")


def single_alfworld_game(data_directory):
    games = []
    trajectories = []
    for root, _directories, files in os.walk(data_directory):
        if "game.tw-pddl" in files:
            games.append(os.path.join(root, "game.tw-pddl"))
        if "traj_data.json" in files:
            trajectories.append(os.path.join(root, "traj_data.json"))
    if (
        len(games) != 1
        or len(trajectories) != 1
        or os.path.dirname(games[0]) != os.path.dirname(trajectories[0])
    ):
        raise ValueError("ALFWorld data directory must contain exactly one complete game")
    return os.path.realpath(games[0])


class ALFWorldWorker(object):
    def __init__(self, source_root, deployment, task):
        deployment = require_object(
            deployment,
            {
                "config_path",
                "data_directory",
                "seed",
                "train_eval",
            },
            "ALFWorld deployment",
        )
        task = require_object(
            task,
            {"game_id", "max_steps", "simulator_max_steps"},
            "ALFWorld task",
        )
        self.game_id = require_text(task["game_id"], "ALFWorld game_id")
        self.outer_max_steps = require_int(task["max_steps"], "ALFWorld outer max_steps", 1)
        self.simulator_max_steps = require_int(
            task["simulator_max_steps"],
            "ALFWorld simulator max_steps",
            self.outer_max_steps,
        )
        config = load_alfworld_config(deployment["config_path"])
        self.env, self.expected_game = create_alfworld_env(
            source_root,
            config=config,
            data_directory=deployment["data_directory"],
            max_steps=self.simulator_max_steps,
            seed=deployment["seed"],
            train_eval=deployment["train_eval"],
        )

    def reset(self):
        return reset_alfworld_env(
            self.env,
            expected_game=self.expected_game,
        )

    def step(self, action):
        result = require_sequence(self.env.step([action]), "ALFWorld step result")
        if len(result) != 4:
            raise ValueError("ALFWorld step result has an incompatible shape")
        observation = require_text(
            require_singleton(result[0], "ALFWorld observations"),
            "ALFWorld observation",
        )
        terminal = require_bool(
            require_singleton(result[2], "ALFWorld terminal flags"),
            "ALFWorld terminal",
        )
        info = result[3]
        if not isinstance(info, dict):
            raise TypeError("ALFWorld step info must be an object")
        commands = require_singleton(info["admissible_commands"], "ALFWorld command batches")
        won = require_bool(
            require_singleton(info["won"], "ALFWorld won flags"),
            "ALFWorld won flag",
        )
        return {
            "admissible_commands": [
                require_text(command, "ALFWorld admissible command")
                for command in require_sequence(commands, "ALFWorld commands")
            ],
            "observation_text": observation,
            "success": won if terminal else None,
            "terminal": terminal,
        }

    def close(self):
        self.env.close()


def load_alfworld_config(config_path):
    config_path = os.path.realpath(require_text(config_path, "ALFWorld config_path"))
    if not os.path.isfile(config_path):
        raise ValueError("ALFWorld config path does not exist")
    yaml_module = importlib.import_module("yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml_module.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("ALFWorld config must be an object")
    return config


def create_alfworld_env(
    source_root,
    *,
    config,
    data_directory,
    max_steps,
    seed,
    train_eval,
):
    builder, expected_game = create_alfworld_builder(
        source_root,
        config=config,
        data_directory=data_directory,
        max_steps=max_steps,
        seed=seed,
        train_eval=train_eval,
    )
    return builder.init_env(batch_size=1), expected_game


def create_alfworld_builder(
    source_root,
    *,
    config,
    data_directory,
    max_steps,
    seed,
    train_eval,
):
    data_directory = os.path.realpath(require_text(data_directory, "ALFWorld data_directory"))
    if not os.path.isdir(data_directory):
        raise ValueError("ALFWorld data directory does not exist")
    expected_game = single_alfworld_game(data_directory)
    train_eval = require_text(train_eval, "ALFWorld train_eval")
    path_keys = {
        "train": "data_path",
        "eval_in_distribution": "eval_id_data_path",
        "eval_out_of_distribution": "eval_ood_data_path",
    }
    if train_eval not in path_keys:
        raise ValueError("ALFWorld train_eval is unsupported")
    max_steps = require_int(max_steps, "ALFWorld max_steps", 1)
    seed = require_int(seed, "ALFWorld seed")

    config = copy.deepcopy(config)
    dataset = config["dataset"]
    dataset[path_keys[train_eval]] = data_directory
    dataset["num_train_games"] = -1
    dataset["num_eval_games"] = -1
    config["general"]["random_seed"] = seed
    config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
    config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps

    module = importlib.import_module("alfworld.agents.environment")
    require_module_in_source(module, source_root)
    environment_type = module.get_environment("AlfredTWEnv")
    builder = environment_type(config, train_eval=train_eval)
    game_files = require_sequence(builder.game_files, "ALFWorld collected games")
    if len(game_files) != 1 or os.path.realpath(game_files[0]) != expected_game:
        raise ValueError("ALFWorld did not bind the pinned single game")
    return builder, expected_game


def reset_alfworld_env(env, *, expected_game):
    result = require_sequence(env.reset(), "ALFWorld reset result")
    if len(result) != 2:
        raise ValueError("ALFWorld reset result has an incompatible shape")
    observation = require_text(
        require_singleton(result[0], "ALFWorld reset observations"),
        "ALFWorld reset observation",
    )
    info = result[1]
    if not isinstance(info, dict):
        raise TypeError("ALFWorld reset info must be an object")
    commands = require_singleton(info["admissible_commands"], "ALFWorld reset command batches")
    game_file = require_text(
        require_singleton(info["extra.gamefile"], "ALFWorld reset game files"),
        "ALFWorld reset game file",
    )
    if os.path.realpath(game_file) != expected_game:
        raise ValueError("ALFWorld reset selected another game")
    return {
        "admissible_commands": [
            require_text(command, "ALFWorld admissible command")
            for command in require_sequence(commands, "ALFWorld commands")
        ],
        "instruction_text": reset_public_goal(observation),
        "observation_text": observation,
    }


def initialize(payload):
    data = require_object(
        payload,
        {"benchmark", "deployment", "source_revision", "source_root", "task"},
        "worker initialization",
    )
    source_root = os.path.realpath(require_text(data["source_root"], "source_root"))
    if source_root != os.path.realpath(os.getcwd()):
        raise ValueError("worker cwd differs from its pinned source root")
    require_text(data["source_revision"], "source_revision")
    sys.path.insert(0, source_root)
    benchmark = require_text(data["benchmark"], "benchmark")
    if benchmark == "alfworld":
        return ALFWorldWorker(source_root, data["deployment"], data["task"])
    raise ValueError("worker benchmark is unsupported")


def response(request_id, result):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-fd", required=True, type=int)
    args = parser.parse_args()
    response_stream = os.fdopen(args.response_fd, "w", buffering=1, encoding="utf-8")
    worker = None
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_MESSAGE_BYTES:
            raise ValueError("worker request is too large")
        request = require_object(
            json.loads(raw_line.decode("utf-8")),
            {"operation", "payload", "protocol_version", "request_id"},
            "worker request",
        )
        if request["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("worker protocol version differs")
        request_id = require_int(request["request_id"], "request_id", 1)
        operation = require_text(request["operation"], "operation")
        payload = request["payload"]
        if not isinstance(payload, dict):
            raise TypeError("worker request payload must be an object")
        if operation == "initialize":
            if worker is not None:
                raise ValueError("worker is already initialized")
            worker = initialize(payload)
            result = {"ready": True}
        elif operation == "reset":
            require_object(payload, set(), "reset payload")
            if worker is None:
                raise ValueError("worker is not initialized")
            result = worker.reset()
        elif operation == "step":
            step = require_object(payload, {"action"}, "step payload")
            if worker is None:
                raise ValueError("worker is not initialized")
            result = worker.step(require_text(step["action"], "action"))
        elif operation == "close":
            require_object(payload, set(), "close payload")
            if worker is None:
                raise ValueError("worker is not initialized")
            worker.close()
            result = {"closed": True}
        else:
            raise ValueError("worker operation is unsupported")
        response_stream.write(
            json.dumps(
                response(request_id, result),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        if operation == "close":
            return


if __name__ == "__main__":
    main()
