from __future__ import annotations

import importlib
import os
import sys

from skillev.policy.sglang_sampling import (
    install_raw_sampling_compatibility,
    install_unsigned_seed_compatibility,
)
from skillev.runtime.sglang_event_grammar import install_event_grammar
from skillev.runtime.sglang_timing import install_request_timing_transport


def configure_project_options(arguments: list[str]) -> list[str]:
    option = "--skillev-fp32-mamba-checkpoints"
    if option not in arguments:
        return arguments
    from skillev.policy.sglang_mamba_checkpoint import install_fp32_mamba_checkpoints

    os.environ["SKILLEV_SGLANG_FP32_MAMBA_CHECKPOINTS"] = "1"
    install_fp32_mamba_checkpoints()
    return [value for value in arguments if value != option]


def main() -> None:
    arguments = configure_project_options(sys.argv[1:])
    launcher = importlib.import_module("sglang.launch_server")
    importlib.import_module("sglang.srt.plugins").load_plugins()
    arguments = launcher.prepare_server_args(arguments)
    try:
        launcher.run_server(arguments)
    finally:
        launcher.kill_process_tree(os.getpid(), include_parent=False)


if __name__ in {"__main__", "__mp_main__"}:
    install_raw_sampling_compatibility()
    install_unsigned_seed_compatibility()
    install_event_grammar(server_info=__name__ == "__main__")
    install_request_timing_transport()
    if os.environ.get("SKILLEV_SGLANG_FP32_MAMBA_CHECKPOINTS") == "1":
        from skillev.policy.sglang_mamba_checkpoint import install_fp32_mamba_checkpoints

        install_fp32_mamba_checkpoints()
    if __name__ == "__main__":
        main()
