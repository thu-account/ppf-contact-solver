# File: addon_host_tests/_remote_build_selection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Host-side gates for the GPU/CPU choice on a REMOTE connection: SSH, SSH
# Command, Docker, Docker over SSH (``blender_addon/core/connection.py`` and
# ``blender_addon/core/remote_builds.py``).
#
# THE QUESTION IS THE SAME ONE ``_native_device_selection_`` ASKS, AND SO IS
# THE ANSWER. Choosing GPU or CPU is choosing which build DIRECTORY the server
# comes from, because the solver links one backend and ``build.rs`` refuses to
# put a second one in a directory that already holds another. What differs on a
# remote connection is only who can see the disk: the add-on cannot stat the
# solver host, so the host is asked once for a listing and the same rule
# resolves that instead of a filesystem.
#
# THE DRIFT THIS PREVENTS is two rules where there should be one. If the remote
# path grew its own notion of which layouts count or what a marker overrides,
# the same folder would mean different things over SSH and over a native
# connection, and only one of them could be right. The cases below are
# deliberately the same cases as the native file's, driven through the other
# seam.
#
# WHAT WOULD GO WRONG WITHOUT ANY OF IT, measured rather than imagined: before
# this existed the remote launch spelled one path, ``target/release``, so the
# Compute Device reached nothing at all on a remote connection, and an unpacked
# Linux distribution, which ships ``target/<backend>/release`` and no
# ``target/release``, could not be launched or even validated.

from __future__ import annotations

import posixpath
import shutil
import subprocess
import sys

import pytest


ROOT = "/home/u/ppf-contact-solver"


@pytest.fixture(scope="module")
def conn():
    """``blender_addon.core.connection``, loaded from source."""
    from conftest import ADDON_ROOT, _ensure_package, load_addon_module

    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.connection")


@pytest.fixture(scope="module")
def remote_builds():
    """``blender_addon.core.remote_builds``, loaded from source."""
    from conftest import ADDON_ROOT, _ensure_package, load_addon_module

    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.remote_builds")


def _listing(*pairs):
    """A build listing in the shape the probe's output parses into."""
    return {posixpath.join(ROOT, sub): marker for sub, marker in pairs}


# ---------------------------------------------------------------------------
# The probe: what is asked, and how its answer is read
# ---------------------------------------------------------------------------


def test_the_probe_asks_about_every_layout_a_build_can_land_in(remote_builds):
    """Including the per-backend directories, which is what a distribution has.

    A probe that asked only about `target/release` would report an unpacked
    Linux distribution as a host with no solver on it, which is the defect this
    whole module replaced.
    """
    command = remote_builds.probe_command(ROOT)
    for sub in (
        "target/release",
        "target/cuda/release",
        "target/rocm/release",
        "target/cpu/release",
    ):
        # Each path is joined whole and then quoted, so it appears as one word.
        assert f"'{ROOT}/{sub}/ppf-cts-server'" in command, sub
        assert f"'{ROOT}/{sub}'" in command, sub
    assert ".ppf-backend" in command


def test_the_output_ends_in_the_directory_not_the_marker(remote_builds):
    """Because every transport strips the whole output before this reads it.

    `exec_command` returns `stdout.decode().strip().splitlines()` on the SSH,
    Docker and native backends alike. A directory holding a server and NO
    marker prints an empty field, so with the directory first that line would
    END in the separator, the strip would take it, and the directory would
    vanish from the listing. An unmarked build is exactly what a distribution
    from before markers looks like, and it is the LAST line of the output that
    loses its tab, so which directory disappears depends on what else the host
    has.
    """
    command = remote_builds.probe_command(ROOT)
    # The marker is read first and the directory printed last.
    assert command.index("cat ") < command.index(f"'{ROOT}/target/release';")
    stripped = f"\tcuda\t{ROOT}/target/cuda/release\n\t{ROOT}/target/release\n".strip()
    assert remote_builds.parse_listing(stripped.splitlines()) == {
        f"{ROOT}/target/cuda/release": "cuda",
        f"{ROOT}/target/release": "",
    }


def test_a_tilde_root_is_tested_expanded_and_printed_unexpanded(remote_builds):
    """A remote path is whatever the artist typed, and `~/...` is reasonable.

    Quoted, a tilde does not expand, so the `[ -f ... ]` test would be false on
    a host that has the build. The PRINTED path must stay unexpanded, because
    the listing is keyed by it and every later lookup joins onto the root the
    panel and the launch hold, which is still the tilde spelling.
    """
    command = remote_builds.probe_command("~/ppf-contact-solver")
    tested = '"$HOME"' + "'/ppf-contact-solver/target/release/ppf-cts-server'"
    assert tested in command
    assert "'~/ppf-contact-solver/target/release';" in command


def test_the_probe_lists_a_directory_only_with_the_solver_beside_the_server(remote_builds):
    """A release directory holding the server alone is not a build.

    `cargo build --release -p ppf-cts-server` leaves exactly that, and a host
    holding it was listed as having a GPU build: Start Server then launched a
    server with no solver to run. The listing is the only thing the add-on
    knows about that host, so the test for the second file has to be in the
    command, and it has to read a `~` root the way the server test does.
    """
    command = remote_builds.probe_command(ROOT)
    for sub in (
        "target/release",
        "target/cuda/release",
        "target/rocm/release",
        "target/metal/release",
        "target/cpu/release",
    ):
        assert f"'{ROOT}/{sub}/ppf-contact-solver'" in command, sub
    command = remote_builds.probe_command("~/ppf-contact-solver")
    tested = '"$HOME"' + "'/ppf-contact-solver/target/release/ppf-contact-solver'"
    assert tested in command


def _probe_by_a_real_shell(remote_builds, root, env=None):
    """Run the probe the way a solver host does, and return what it printed."""
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("no POSIX shell on this machine to run the probe with")
    done = subprocess.run(
        [shell, "-c", remote_builds.probe_command(root)],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert done.returncode == 0, (done.returncode, done.stderr)
    return remote_builds.parse_listing(done.stdout.splitlines())


def _host_tree(root):
    """A solver host's tree with one of everything the probe has to tell apart.

    THE LAST DIRECTORY THE PROBE VISITS IS ONE IT MUST NOT LIST, deliberately:
    a command built as `test && test && printf` rather than inside `if ...;
    fi` exits non-zero there, and the add-on records a non-zero probe as a
    host that could not be asked.
    """
    def directory(*parts, server=True, solver=True, marker=None):
        d = root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        if server:
            (d / "ppf-cts-server").write_text("")
        if solver:
            (d / "ppf-contact-solver").write_text("")
        if marker is not None:
            (d / ".ppf-backend").write_text(marker + "\n")

    # What `cargo build --release -p ppf-cts-server` leaves: the server alone.
    directory("target", "release", solver=False)
    directory("target", "cuda", "release", marker="cuda")
    # A complete build from before markers were written.
    directory("target", "rocm", "release")
    # A build that failed after `build.rs` wrote its marker.
    directory("target", "cpu", "release", solver=False, marker="cpu")


@pytest.mark.skipif(sys.platform == "win32", reason="the probe runs on a POSIX solver host")
def test_the_probe_run_by_a_real_shell_lists_builds_and_only_builds(
    remote_builds, conn, tmp_path
):
    """The command itself, run by `sh` against a tree on disk.

    Every other test here reads the command as text or hands the resolver a
    listing written by hand, so nothing held the two ends together: what the
    shell prints for a real directory, and what the native resolver says about
    that same directory. This runs both and requires them to agree, which is
    the drift gate for the one rule the two seams cannot share in code.
    """
    _host_tree(tmp_path)
    root = tmp_path.as_posix()
    listing = _probe_by_a_real_shell(remote_builds, root)
    assert listing == {
        f"{root}/target/cuda/release": "cuda",
        f"{root}/target/rocm/release": "",
    }
    for device in (conn.DEVICE_GPU, conn.DEVICE_CPU):
        for gpu_backend in (conn.GPU_BACKEND_AUTO, "CUDA", "ROCM"):
            native = conn.linux_native_server_binary(str(tmp_path), device, gpu_backend)
            remote = conn.remote_server_binary(root, listing, device, gpu_backend)
            assert native == remote, (device, gpu_backend)
    assert conn.remote_holds_any_server(root, listing)
    assert conn.resolve_linux_native_root(str(tmp_path)) == str(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="the probe runs on a POSIX solver host")
def test_a_tilde_root_run_by_a_real_shell_finds_the_build(remote_builds, tmp_path):
    """Both tests read the expanded path, and the listing keeps the tilde."""
    _host_tree(tmp_path / "ppf")
    listing = _probe_by_a_real_shell(
        remote_builds, "~/ppf", env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    )
    assert listing == {
        "~/ppf/target/cuda/release": "cuda",
        "~/ppf/target/rocm/release": "",
    }


@pytest.mark.skipif(sys.platform == "win32", reason="the probe runs on a POSIX solver host")
def test_a_host_holding_only_a_bare_server_lists_nothing(remote_builds, conn, tmp_path):
    """And the refusal that follows names what a build is and how to make one."""
    d = tmp_path / "target" / "release"
    d.mkdir(parents=True)
    (d / "ppf-cts-server").write_text("")
    root = tmp_path.as_posix()
    listing = _probe_by_a_real_shell(remote_builds, root)
    assert listing == {}
    assert not conn.remote_holds_any_server(root, listing)
    text = conn.remote_not_found_message(root, listing, conn.DEVICE_GPU)
    assert "ppf-contact-solver" in text, text
    assert "`cargo build --release`" in text, text
    assert "-p ppf-cts-server" not in text, text
    # The server IS on this host, so the refusal must not say it was not found.
    assert "not found" not in text, text


def test_the_probe_quotes_the_root_it_is_given(remote_builds):
    """The root is interpolated into a shell command, so it is quoted.

    The add-on's own metacharacter gate refuses a path like this before it
    reaches a connection; the quoting is the second of the two rather than the
    only one.
    """
    command = remote_builds.probe_command("/home/u/a b'c")
    assert "'/home/u/a b'\\''c/target/release/ppf-cts-server'" in command


def test_a_banner_in_the_output_does_not_discard_the_listing(remote_builds):
    """The command runs under the user's own login shell on someone's machine.

    A `mesg` warning or a login banner printed ahead of the output is a thing
    that happens, and refusing the whole listing over one would report a host
    with no builds, which reads as a wrong path.
    """
    parsed = remote_builds.parse_listing(
        [
            "mesg: ttyname failed: Inappropriate ioctl for device",
            f"cuda\t{ROOT}/target/cuda/release",
            "",
            f"cpu\t{ROOT}/target/cpu/release",
        ]
    )
    assert parsed == {
        f"{ROOT}/target/cuda/release": "cuda",
        f"{ROOT}/target/cpu/release": "cpu",
    }


def test_a_directory_with_no_marker_is_listed_with_an_empty_one(remote_builds):
    """Which is the normal shape of a distribution built before markers.

    An empty marker is not an error and must not drop the directory: the layout
    is then the only evidence of what it holds, which is the answer that
    shipped before markers were read at all.
    """
    parsed = remote_builds.parse_listing([f"\t{ROOT}/target/release"])
    assert parsed == {f"{ROOT}/target/release": ""}


def test_a_marker_written_by_cmd_exe_reads_the_same(remote_builds):
    """`$(cat ...)` of a CRLF marker leaves the carriage return on the value."""
    parsed = remote_builds.parse_listing([f"cpu\r\t{ROOT}/target/cpu/release"])
    assert parsed == {f"{ROOT}/target/cpu/release": "cpu"}


# ---------------------------------------------------------------------------
# The rule, resolved over a listing
# ---------------------------------------------------------------------------


def test_each_device_resolves_its_own_build(conn):
    """The property the whole feature rests on, asked of a remote host."""
    both = _listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    gpu = conn.remote_server_binary(ROOT, both, conn.DEVICE_GPU)
    cpu = conn.remote_server_binary(ROOT, both, conn.DEVICE_CPU)
    assert gpu == f"{ROOT}/target/cuda/release/ppf-cts-server"
    assert cpu == f"{ROOT}/target/cpu/release/ppf-cts-server"
    assert gpu != cpu


def test_the_default_is_gpu_so_saved_files_keep_their_behavior(conn):
    both = _listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    assert conn.remote_server_binary(ROOT, both) == conn.remote_server_binary(
        ROOT, both, conn.DEVICE_GPU
    )


def test_a_missing_build_is_absent_rather_than_substituted(conn):
    """NOT the other device. A fallback here is the silent substitution."""
    gpu_only = _listing(("target/cuda/release", "cuda"))
    assert conn.remote_server_binary(ROOT, gpu_only, conn.DEVICE_CPU) is None
    cpu_only = _listing(("target/cpu/release", "cpu"))
    assert conn.remote_server_binary(ROOT, cpu_only, conn.DEVICE_GPU) is None


def test_an_unknown_device_never_falls_back(conn):
    both = _listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    assert conn.remote_server_binary(ROOT, both, "TPU") is None
    assert conn.remote_server_binary(ROOT, both, "") is None


def test_a_cpu_build_in_the_gpu_layout_is_cpu(conn):
    """`--features cpu` links into `target/release` when it is free to.

    The marker is the only evidence of what a directory holds, and reading the
    layout alone would offer GPU and run the CPU solver.
    """
    tree = _listing(("target/release", "cpu"))
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_CPU) is not None
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_GPU) is None


def test_an_unmarked_directory_answers_only_for_its_own_layout(conn):
    tree = _listing(("target/release", ""))
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_GPU) is not None
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_CPU) is None


def test_an_unknown_backend_name_is_not_accelerated(conn):
    """A name this add-on has never heard of answers to CPU, not to GPU.

    The safe direction: the artist is told their build is slow when it is fast,
    rather than being handed a 30x slowdown labeled GPU.
    """
    tree = _listing(("target/release", "vulkan"))
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_GPU) is None
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_CPU) is not None


def test_a_named_accelerator_resolves_to_that_build_or_to_nothing(conn):
    three = _listing(
        ("target/cuda/release", "cuda"),
        ("target/rocm/release", "rocm"),
        ("target/cpu/release", "cpu"),
    )
    assert (
        conn.remote_server_binary(ROOT, three, conn.DEVICE_GPU, "ROCM")
        == f"{ROOT}/target/rocm/release/ppf-cts-server"
    )
    only_cuda = _listing(("target/cuda/release", "cuda"))
    assert conn.remote_server_binary(ROOT, only_cuda, conn.DEVICE_GPU, "ROCM") is None


def test_the_marker_wins_over_the_directory_name(conn):
    """A `target/cuda` holding a ROCm build is a ROCm build."""
    tree = _listing(("target/cuda/release", "rocm"))
    assert (
        conn.remote_server_binary(ROOT, tree, conn.DEVICE_GPU, "ROCM")
        == f"{ROOT}/target/cuda/release/ppf-cts-server"
    )
    assert conn.remote_server_binary(ROOT, tree, conn.DEVICE_GPU, "CUDA") is None


def test_automatic_prefers_cuda_over_rocm(conn):
    """The same order `frontend._backends_.GPU_BACKENDS` prefers.

    NO PROBE IS RUN, unlike the native path: the solver that would answer is on
    the other machine. So AUTO takes the first in that order rather than the
    first usable one, and the panel draws the GPU Backend row for the artist
    whose remote host has both.
    """
    three = _listing(
        ("target/cuda/release", "cuda"),
        ("target/rocm/release", "rocm"),
        ("target/cpu/release", "cpu"),
    )
    assert (
        conn.remote_server_binary(ROOT, three, conn.DEVICE_GPU)
        == f"{ROOT}/target/cuda/release/ppf-cts-server"
    )


# ---------------------------------------------------------------------------
# What a run takes out of the directory, and what a refusal says
# ---------------------------------------------------------------------------


def test_the_target_directory_is_derived_from_the_server_it_resolved(conn):
    """A run takes three things out of a build directory, not just the server.

    Naming only the binary leaves the build worker's frontend to load the
    cdylib from wherever it finds one first, and the session's own launcher
    then names THAT directory's solver. The two halves of one run come from
    different builds, and nothing reports it.
    """
    assert (
        conn.remote_target_dir(f"{ROOT}/target/cpu/release/ppf-cts-server")
        == f"{ROOT}/target/cpu"
    )
    assert (
        conn.remote_target_dir(f"{ROOT}/target/release/ppf-cts-server")
        == f"{ROOT}/target"
    )
    assert conn.remote_target_dir(f"{ROOT}/bin/ppf-cts-server") is None
    assert conn.remote_target_dir("") is None


def test_a_root_holding_any_build_validates_including_a_distribution(conn):
    """Validation asks whether the folder holds a solver AT ALL.

    Not whether it holds the device currently selected: the device is applied
    at Start Server, and a refusal at validate is a CANCELLATION rather than a
    message, so refusing there over a choice the artist can still change would
    strand the upload that had already started.
    """
    distribution = _listing(
        ("target/cuda/release", "cuda"), ("target/cpu/release", "cpu")
    )
    assert conn.remote_holds_any_server(ROOT, distribution) is True
    assert conn.remote_holds_any_server(ROOT, _listing(("target/release", ""))) is True
    assert conn.remote_holds_any_server(ROOT, {}) is False


def test_the_refusal_for_the_other_device_names_what_is_there(conn):
    """The folder is right and the Compute Device is not, so the text says so.

    A generic "not found" would send the artist to change a path that is
    correct, which is both wrong and unactionable.
    """
    gpu_only = _listing(("target/cuda/release", "cuda"))
    message = conn.remote_not_found_message(ROOT, gpu_only, conn.DEVICE_CPU)
    assert "GPU build" in message
    assert "CPU" in message


def test_the_refusal_for_the_other_accelerator_names_what_is_there(conn):
    three = _listing(
        ("target/cuda/release", "cuda"), ("target/cpu/release", "cpu")
    )
    message = conn.remote_not_found_message(ROOT, three, conn.DEVICE_GPU, "ROCM")
    assert "cuda" in message.lower()
    assert "rocm" in message.lower()


def test_a_root_with_nothing_in_it_says_which_layouts_were_examined(conn):
    message = conn.remote_not_found_message(ROOT, {}, conn.DEVICE_GPU)
    assert ROOT in message
    assert "target/cuda/release" in message


def test_a_servable_selection_has_no_refusal(conn):
    both = _listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    assert conn.remote_not_found_message(ROOT, both, conn.DEVICE_GPU) is None
    assert conn.remote_not_found_message(ROOT, both, conn.DEVICE_CPU) is None


# ---------------------------------------------------------------------------
# The two seams answer the same questions
# ---------------------------------------------------------------------------


def test_the_remote_rule_matches_the_native_one_case_for_case(conn, tmp_path):
    """The same tree, asked of a filesystem and of a listing, answers the same.

    This is the drift gate. The two seams share `_native_server_binary`,
    `gpu_builds` and the marker rule, and this holds them to it against a tree
    built on disk and a listing describing the same tree.
    """
    layouts = (
        ("target/release", "cpu"),
        ("target/cuda/release", "cuda"),
        ("target/rocm/release", "rocm"),
        ("target/cpu/release", "cpu"),
    )
    for sub, marker in layouts:
        d = tmp_path.joinpath(*sub.split("/"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppf-cts-server").write_text("")
        (d / "ppf-contact-solver").write_text("")
        (d / ".ppf-backend").write_text(marker)
    listing = {
        posixpath.join(tmp_path.as_posix(), sub): marker for sub, marker in layouts
    }
    for device in (conn.DEVICE_GPU, conn.DEVICE_CPU):
        for gpu_backend in (conn.GPU_BACKEND_AUTO, "CUDA", "ROCM"):
            native = conn.linux_native_server_binary(
                str(tmp_path), device, gpu_backend
            )
            remote = conn.remote_server_binary(
                tmp_path.as_posix(), listing, device, gpu_backend
            )
            assert (native is None) == (remote is None), (device, gpu_backend)
            if native is not None:
                assert native.replace("\\", "/") == remote, (device, gpu_backend)


def test_a_trailing_slash_names_the_same_root_as_without_one(remote_builds, conn):
    """A remote path is taken as the artist typed it, so both spellings arrive.

    THE LISTING IS KEYED BY ABSOLUTE DIRECTORY, and the panel and the launch
    both join onto a root to look one up. A native path is normalized at
    connect and a remote one is not, so `/srv/ppf/` and `/srv/ppf` would be two
    keys for one directory: the panel would report a host with no builds while
    Start Server, joining onto the same unnormalized string, launched fine.
    """
    assert remote_builds.normalize_root(ROOT + "/") == ROOT
    assert remote_builds.normalize_root(ROOT) == ROOT
    assert remote_builds.normalize_root("/") == "/"
    assert remote_builds.normalize_root("") == ""
    # The probe asks about the normalized root, so its keys are the ones a
    # later lookup computes.
    assert remote_builds.probe_command(ROOT + "/") == remote_builds.probe_command(ROOT)
    both = _listing(("target/cuda/release", "cuda"), ("target/cpu/release", "cpu"))
    assert conn.remote_server_binary(
        remote_builds.normalize_root(ROOT + "/"), both, conn.DEVICE_GPU
    ) == f"{ROOT}/target/cuda/release/ppf-cts-server"


def test_the_filesystem_root_keys_the_listing_the_way_a_lookup_joins_it(remote_builds):
    """A root of `/` is a path like any other, and must not gain a separator.

    Concatenating a quoted root onto "/<subdir>" produces `//target/release`,
    which `posixpath.join("/", "target/release")` never computes, so every
    lookup misses and the host is reported as holding no solver at all. That
    refusal cancels a pipeline, so it is worth the one join.
    """
    command = remote_builds.probe_command("/")
    assert "'/target/release'" in command
    assert "//target/release" not in command
