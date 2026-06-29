# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Adapted from sglang.srt.entrypoints.http_server.launch_server
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers. Everything else mirrors the upstream launch_server flow.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys


def areal_launch_server(server_args) -> None:
    from sglang.srt.entrypoints.engine import init_tokenizer_manager
    from sglang.srt.entrypoints import http_server
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        app,
        launch_server,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    # ---- BEGIN AREAL ----
    from areal.experimental.inference_service.sglang.awex import (
        register_awex_endpoints,
    )
    from areal.experimental.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.experimental.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    result_ipc = create_result_ipc()
    rpc_proxy: RpcProxy | None = None
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    original_launch_subprocesses = http_server._launch_subprocesses

    def capture_launch_subprocesses(*args, **kwargs):
        nonlocal rpc_proxy
        result = original_launch_subprocesses(*args, **kwargs)
        # SGLang 0.5.9 returns
        # (tokenizer_manager, template_manager, scheduler_infos, port_args).
        tokenizer_manager = result[0]
        port_args = result[3]
        if tokenizer_manager is not None:
            rpc_proxy = RpcProxy(port_args, result_ipc)
            register_awex_endpoints(app, rpc_proxy)
        return result
    # ---- END AREAL ----

    try:
        # SGLang 0.5.9 removed the private _setup_and_run_http_server helper.
        # Use the public launch_server entrypoint while injecting AReaL's
        # scheduler process and AWEX FastAPI routes.
        http_server._launch_subprocesses = capture_launch_subprocesses
        launch_server(
            server_args=server_args,
            init_tokenizer_manager_func=init_tokenizer_manager,
            run_scheduler_process_func=areal_run_scheduler_process,
            run_detokenizer_process_func=run_detokenizer_process,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        # ---- BEGIN AREAL ----
        http_server._launch_subprocesses = original_launch_subprocesses
        if rpc_proxy is not None:
            rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
