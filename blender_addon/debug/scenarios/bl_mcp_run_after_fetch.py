# File: scenarios/bl_mcp_run_after_fetch.py
# Code: Claude Code
# License: GPL-3.0-or-later
#
# run_simulation over MCP when the scene still holds fetched animation.
#
# The Run button stays grayed out while Clear Local Animation has something to
# remove, and the panel says so. Over MCP the same gate used to answer only
# "operator conditions not met", which leaves an agent that ran, fetched and
# wants to run again with nothing to act on: the status it is shown reads
# "Ready to Run". The refusal now names the blocker and the tool that clears
# it, and the three readers of the gate (SOLVER_OT_Run.poll, the panel line and
# the MCP refusal) share one predicate, run_ready_except_local_animation, so
# they cannot drift apart.
#
# The scene is solved for real, because the state under test is the one a
# fetch leaves behind: a READY server, a connection, and ContactSolverCache
# modifiers on an assigned object.
#
# Assertions:
#   A. ``fetched_animation_is_the_only_blocker`` -- after a solve and a fetch,
#      Clear Local Animation is available, the shared predicate says Run is
#      otherwise ready, and Run's poll is False.
#   B. ``description_names_the_fix`` -- tools/list describes run_simulation
#      with the refusal and names clear_local_animation.
#   C. ``refusal_names_clear_local_animation`` -- run_simulation answers a tool
#      failure (isError, not a protocol error) whose message names
#      clear_local_animation and still carries the status suffix.
#   D. ``refusal_changes_nothing`` -- after the refusal the animation is still
#      there and Run is still gated, so the refusal started nothing.
#   E. ``clear_reopens_run`` -- clear_local_animation succeeds over MCP and
#      Run's poll turns True, so the message names a fix that works.

from __future__ import annotations

from . import _driver_lib as dl
from . import _mcp_lib as ml
from . import _runner as r
from . import REPO_ROOT_POSIX

NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it: a rig run against a CPU
# build on Linux passed it.
BACKENDS = ("real",)

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, as for the other MCP scenarios.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 4

try:
    dh = DriverHelpers(pkg, result)
    solver_ui = __import__(pkg + ".ui.solver", fromlist=["SOLVER_OT_Run"])

    def gate():
        ctx = bpy.context
        return {
            "clear_available": bool(solver_ui.SOLVER_OT_ClearAnimation.poll(ctx)),
            "ready_otherwise": bool(solver_ui.run_ready_except_local_animation()),
            "run_poll": bool(solver_ui.SOLVER_OT_Run.poll(ctx)),
        }

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=1, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "RunAfterFetchSheet"
    root = dh.configure_state(
        project_name="mcp_run_after_fetch", frame_count=FRAME_COUNT,
    )
    dh.api.solver.create_group("Sheet", "SHELL").add(sheet.name)

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _data_hash, _param_hash = (
        encoder_pkg.prepare_upload(bpy.context)
    )
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "run-after-fetch:build",
                      timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetched applied={applied}/{total}")

    before = gate()
    dh.record(
        "A_fetched_animation_is_the_only_blocker",
        before["clear_available"] and before["ready_otherwise"]
        and not before["run_poll"],
        before,
    )

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    try:
        env, _resp = mcp_call(pkg, url, "tools/list", request_id=1)
        served = (env.get("result") or {}).get("tools") or []
        description = ""
        for tool in served:
            if tool.get("name") == "run_simulation":
                description = " ".join((tool.get("description") or "").split())
        dh.record(
            "B_description_names_the_fix",
            "clear_local_animation" in description,
            {"description": description},
        )

        env, _resp = mcp_call(
            pkg, url, "tools/call",
            {"name": "run_simulation", "arguments": {}},
            request_id=2,
        )
        call_result = env.get("result") or {}
        message = mcp_tool_payload(env).get("message") or ""
        dh.record(
            "C_refusal_names_clear_local_animation",
            "error" not in env
            and call_result.get("isError") is True
            and "clear_local_animation" in message
            and "status=" in message,
            {
                "isError": call_result.get("isError"),
                "message": message,
                "protocol_error": env.get("error"),
            },
        )

        after_refusal = gate()
        dh.record(
            "D_refusal_changes_nothing",
            after_refusal["clear_available"] and not after_refusal["run_poll"],
            after_refusal,
        )

        cleared, _raw = mcp_tool(pkg, url, "clear_local_animation", {},
                                 request_id=3)
        after_clear = gate()
        dh.record(
            "E_clear_reopens_run",
            cleared.get("status") == "success"
            and not after_clear["clear_available"]
            and after_clear["run_poll"],
            {"clear_payload": cleared, **after_clear},
        )
    finally:
        mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + ml.MCP_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
