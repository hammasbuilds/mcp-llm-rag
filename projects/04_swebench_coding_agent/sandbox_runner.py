"""Run one SWE-bench-Lite instance's real test suite, inside a real Docker
container, against a model-generated patch.

For a given instance this does, all inside the container (isolated from the
host Python environment):

  1. apply the official `test_patch` (brings in the FAIL_TO_PASS/PASS_TO_PASS
     tests exactly as SWE-bench defines them)
  2. install the package (`pip install -e .`) + pytest
  3. run FAIL_TO_PASS + PASS_TO_PASS as a *baseline* (before the fix) — sanity
     check that FAIL_TO_PASS really fails and PASS_TO_PASS really passes
     pre-fix
  4. apply the candidate patch (either the real gold patch, for a control run,
     or the model's generated patch)
  5. re-run the same tests as the *patched* result

Every step's real stdout/exit codes are captured; nothing here fabricates a
pass/fail — if `docker` isn't available the caller gets an explicit error,
not a guess.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
WORKDIR_ROOT = HERE / "workdirs"
DOCKER_IMAGE = "swebench-lite-runner:latest"
DOCKERFILE_DIR = HERE / "docker"

_PYTEST_LINE_RE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)\b", re.MULTILINE)


@dataclass
class TestRunResult:
    exit_code: int
    stdout: str
    stderr: str
    outcomes: dict[str, str] = field(default_factory=dict)  # test_id -> PASSED/FAILED/...

    def status_for(self, test_id: str) -> str:
        return self.outcomes.get(test_id, "MISSING")


def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "ps"], capture_output=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_image(quiet: bool = False) -> None:
    cmd = ["docker", "build", "-t", DOCKER_IMAGE, str(DOCKERFILE_DIR)]
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker build failed:\n{r.stdout}\n{r.stderr}")
    if not quiet:
        print(r.stdout[-2000:])


def _docker_run(eval_dir: Path, script: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run `script` with bash inside the sandbox container, with eval_dir
    bind-mounted read-write at /workspace."""
    # Docker Desktop on Windows expects a path like //c/src/... or C:\...;
    # -v with an absolute Windows path works via Docker Desktop's path translation.
    mount = f"{eval_dir.resolve()}:/workspace"
    cmd = [
        "docker", "run", "--rm",
        "-v", mount,
        "-w", "/workspace",
        DOCKER_IMAGE,
        "bash", "-lc", script,
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


def _parse_outcomes(stdout: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _PYTEST_LINE_RE.finditer(stdout)}


def _rmtree_onerror(func, path, exc_info):
    """git marks some objects (pack tmp files, etc.) read-only; on Windows
    that makes plain os.remove/os.rmdir raise PermissionError. Clear the
    read-only bit and retry once instead of leaving a half-deleted directory
    behind (which then makes the next copytree fail with FileExistsError)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def prepare_eval_dir(instance_id: str, suffix: str) -> Path:
    src = WORKDIR_ROOT / instance_id
    dst = WORKDIR_ROOT / f"{instance_id}__{suffix}"
    if dst.exists():
        shutil.rmtree(dst, onerror=_rmtree_onerror)
    shutil.copytree(src, dst)
    return dst


def run_instance(
    instance_id: str,
    test_patch: str,
    candidate_patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    *,
    install_extra: str = "",
    run_baseline: bool = True,
    timeout: int = 900,
    test_cwd: str = ".",
) -> dict:
    """Full pipeline for one instance. Returns a JSON-able results dict.

    `test_cwd` is the directory (relative to the repo root) pytest should be
    invoked from — SWE-bench's own FAIL_TO_PASS/PASS_TO_PASS node-ids are
    written relative to whatever cwd the upstream project's own CI uses
    (e.g. psf/requests' ids are relative to `tests/`, not the repo root).
    """
    eval_dir = prepare_eval_dir(instance_id, "eval")
    # newline="\n" is required on Windows: Path.write_text()'s default text
    # mode translates '\n' -> os.linesep ('\r\n'), which corrupts a unified
    # diff enough that `git apply` inside the (Linux) container rejects every
    # hunk as "does not apply" even though the context matches byte-for-byte
    # modulo line endings.
    (eval_dir / "_test.diff").write_text(test_patch, encoding="utf-8", newline="\n")
    (eval_dir / "_candidate.diff").write_text(candidate_patch, encoding="utf-8", newline="\n")

    all_tests = fail_to_pass + pass_to_pass
    test_args = " ".join(f'"{t}"' for t in all_tests)
    cd_prefix = f"cd {test_cwd} && " if test_cwd != "." else ""

    result: dict = {
        "instance_id": instance_id,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
    }

    # Step 1: apply the official test patch + install the package.
    #
    # pip's defaults give up quickly on a slow link, and when it does the whole
    # run is recorded as `setup_failed` -- which looks like a result and is not
    # one. A 2026-09-16 run lost both instances this way while a large model
    # download was saturating the connection ("This is an issue with network
    # connectivity, not pip"). These settings make the sandbox wait rather than
    # manufacture a false negative.
    pip_net = (
        "PIP_RETRIES=10 PIP_TIMEOUT=120 PIP_DEFAULT_TIMEOUT=120 "
        "PIP_DISABLE_PIP_VERSION_CHECK=1"
    )
    setup_script = (
        "set -e\n"
        "git config --global --add safe.directory /workspace\n"
        "git apply --whitespace=fix _test.diff\n"
        f"{pip_net} pip install -e . -q {install_extra}\n"
        f"{pip_net} pip install pytest -q\n"
    )
    # Setup is the only step that touches the network, so it gets the most room.
    setup = _docker_run(eval_dir, setup_script, timeout=max(timeout, 1800))
    result["setup"] = {
        "exit_code": setup.returncode,
        "stdout_tail": setup.stdout[-3000:],
        "stderr_tail": setup.stderr[-3000:],
    }
    if setup.returncode != 0:
        result["outcome"] = "setup_failed"
        return result

    # Step 2 (optional): baseline run, pre-fix.
    if run_baseline:
        baseline_script = f"{cd_prefix}pytest {test_args} -v --no-header --tb=line -p no:cacheprovider || true"
        baseline = _docker_run(eval_dir, baseline_script, timeout=timeout)
        baseline_outcomes = _parse_outcomes(baseline.stdout)
        result["baseline"] = {
            "exit_code": baseline.returncode,
            "outcomes": baseline_outcomes,
            "stdout_tail": baseline.stdout[-4000:],
        }

    # Step 3: apply the candidate (model-generated, or gold) patch.
    apply_script = "git apply --whitespace=fix _candidate.diff"
    apply_res = _docker_run(eval_dir, apply_script, timeout=120)
    result["patch_apply"] = {
        "exit_code": apply_res.returncode,
        "stdout": apply_res.stdout,
        "stderr": apply_res.stderr,
    }
    if apply_res.returncode != 0:
        result["outcome"] = "patch_did_not_apply"
        return result

    # Step 4: patched run.
    patched_script = f"{cd_prefix}pytest {test_args} -v --no-header --tb=line -p no:cacheprovider || true"
    patched = _docker_run(eval_dir, patched_script, timeout=timeout)
    patched_outcomes = _parse_outcomes(patched.stdout)
    result["patched"] = {
        "exit_code": patched.returncode,
        "outcomes": patched_outcomes,
        "stdout_tail": patched.stdout[-4000:],
    }

    fail_to_pass_now_pass = all(patched_outcomes.get(t) == "PASSED" for t in fail_to_pass)
    pass_to_pass_still_pass = all(patched_outcomes.get(t) == "PASSED" for t in pass_to_pass)
    result["fail_to_pass_resolved"] = fail_to_pass_now_pass
    result["pass_to_pass_preserved"] = pass_to_pass_still_pass
    result["outcome"] = "resolved" if (fail_to_pass_now_pass and pass_to_pass_still_pass) else "not_resolved"
    return result


if __name__ == "__main__":
    print("docker available:", docker_available())
