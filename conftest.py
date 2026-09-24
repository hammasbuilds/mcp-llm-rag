"""Repository-wide test setup.

The `live` marker means "this test needs a running Ollama server". The marker
existed and was applied, but it only ever let you *deselect* those tests with
`-m "not live"`. If you did not pass that flag and the server was down, the
tests did not skip - they failed, with an httpx connection error, which reads
as a broken project rather than a missing service.

Worse, it was inconsistent. `tests/test_03_bfcl.py` checked `/api/tags` itself
and skipped politely; `tests/test_06_fever.py` called straight into the server
and blew up. Same marker, same repository, opposite behaviour - so whether
`pytest` passed depended on which file you ran and what happened to be running
on port 11434.

Checking it once here means a `live` test skips with a reason everywhere, and
the per-file guards do not have to be remembered.
"""

from __future__ import annotations

import os

import pytest

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

_reachable: bool | None = None


def _ollama_is_up() -> bool:
    # Probed once per session. Each live test asking separately turns a down
    # server into one timeout per test, which on a full run is minutes of
    # waiting to be told something the first probe already established.
    global _reachable
    if _reachable is None:
        try:
            import httpx

            httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3).raise_for_status()
            _reachable = True
        except Exception:  # noqa: BLE001
            _reachable = False
    return _reachable


def pytest_runtest_setup(item: pytest.Item) -> None:
    if any(mark.name == "live" for mark in item.iter_markers()) and not _ollama_is_up():
        pytest.skip(f"no Ollama server reachable at {OLLAMA_HOST}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """A `live` test that timed out waiting for the model is a skip, not a failure.

    Reaching /api/tags proves the server is answering; it does not prove the
    model will produce a token before httpx gives up. On a machine whose GPU is
    busy with something else, the first call pays for loading the weights and
    the suite goes red for a reason that has nothing to do with the code.

    Deliberately narrow - `live` tests only, timeouts only. Turning any other
    failure into a skip would hide exactly what these tests exist to catch.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    if not any(mark.name == "live" for mark in item.iter_markers()):
        return
    text = str(getattr(call, "excinfo", "") or "")
    if "Timeout" in text or "ReadTimeout" in text or "ConnectError" in text:
        report.outcome = "skipped"
        report.longrepr = (
            f"{OLLAMA_HOST} accepted the connection but did not answer in time "
            "- the model is probably still loading, or the GPU is busy"
        )
