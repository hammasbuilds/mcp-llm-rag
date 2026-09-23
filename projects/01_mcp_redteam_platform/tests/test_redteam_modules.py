"""Tests for the audit modules themselves.

These produce the numbers the README publishes, so a bug here is not a crash -
it is a wrong security result, stated confidently. The two failure directions
matter differently:

  a module that under-reports  ->  a hole is declared fixed when it is not
  a module that over-reports   ->  noise, and eventually nobody believes it

Only the parts that need no model are covered here. Modules 1, 2, 4 and 5
drive a real agent loop against Ollama; module 3 and the decoy tooling are pure
local code and are the ones worth pinning in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redteam.decoy_tools import DECOY_TOOL_SPECS, DecoyToolSpec  # noqa: E402
from redteam.m3_path_traversal import PAYLOADS, SECRET_MARKER  # noqa: E402
from server.tools import ToolSet  # noqa: E402


class TestPathTraversalPayloads:
    def test_the_control_case_is_present(self):
        """A traversal suite with no control proves nothing.

        Every payload being refused in hardened mode is only meaningful if a
        LEGITIMATE read still succeeds. Without `welcome.txt` in the list, a
        read_file that refused everything - including valid paths - would score
        as a perfect fix.
        """
        assert "welcome.txt" in PAYLOADS

    def test_every_other_payload_actually_attempts_an_escape(self):
        attacks = [p for p in PAYLOADS if p != "welcome.txt"]
        assert attacks, "the suite is all control and no attack"
        for payload in attacks:
            assert ".." in payload, payload

    def test_both_separators_are_covered(self):
        """Windows accepts / and \\, and a check that normalises only one of
        them is a hole on exactly one platform."""
        attacks = [p for p in PAYLOADS if p != "welcome.txt"]
        assert any("/" in p for p in attacks)
        assert any("\\" in p for p in attacks)

    def test_a_payload_that_doubles_back_is_covered(self):
        """`subdir/../../x` normalises to the same place as `../x` but reaches
        it through a directory that may not exist - a different code path in
        anything that resolves lazily."""
        assert any(p.count("..") > 1 for p in PAYLOADS)


class TestPathTraversalAgainstTheRealSandbox:
    """The module's own measurement, run against the seeded fixture."""

    def test_the_secret_file_exists_and_carries_the_marker(self):
        """If the fixture were missing, every payload would raise
        FileNotFoundError, no escape would be detected, and the module would
        report the vulnerable configuration as SAFE."""
        secret = ROOT / "data" / "outside_sandbox_secret.txt"
        assert secret.exists(), "the module measures nothing without this file"
        assert SECRET_MARKER in secret.read_text(encoding="utf-8")

    def test_vulnerable_mode_reaches_the_secret(self):
        tools = ToolSet(hardened=False)
        escaped = []
        for payload in PAYLOADS:
            if payload == "welcome.txt":
                continue
            try:
                escaped.append(SECRET_MARKER in tools.read_file(payload))
            except OSError:
                escaped.append(False)
        assert any(escaped), "no payload escaped - the 'before' number is not real"

    def test_hardened_mode_reaches_it_with_none_of_them(self):
        tools = ToolSet(hardened=True)
        for payload in PAYLOADS:
            if payload == "welcome.txt":
                continue
            try:
                assert SECRET_MARKER not in tools.read_file(payload), payload
            except OSError:
                pass  # refused outright, which is the intended outcome

    def test_the_legitimate_read_still_works_in_both_modes(self):
        for hardened in (False, True):
            content = ToolSet(hardened=hardened).read_file("welcome.txt")
            assert content, f"hardened={hardened} broke a valid read"


class TestDecoyTools:
    def test_there_are_enough_decoys_for_the_bloat_experiment(self):
        """Module 5 grows the tool count to 30 around 4 real tools."""
        assert len(DECOY_TOOL_SPECS) >= 26

    def test_decoy_names_do_not_collide_with_the_real_tools(self):
        """A decoy that shadowed a real tool name would not measure selection
        accuracy - it would break the assistant."""
        real = {"search_docs", "read_file", "notify_user", "query_records"}
        names = {spec.name for spec in DECOY_TOOL_SPECS}
        assert not (names & real)

    def test_decoy_names_are_unique(self):
        names = [spec.name for spec in DECOY_TOOL_SPECS]
        assert len(names) == len(set(names))

    def test_every_decoy_is_clustered_on_a_real_tool(self):
        """The experiment's premise is near-duplicate names, not random noise.
        A decoy attached to nothing makes the tool list longer without making
        the choice harder, which would understate the effect."""
        real = {"search_docs", "read_file", "notify_user", "query_records"}
        for spec in DECOY_TOOL_SPECS:
            assert spec.cluster in real, spec.name

    def test_all_four_real_tools_have_decoys(self):
        clusters = {spec.cluster for spec in DECOY_TOOL_SPECS}
        assert clusters == {"search_docs", "read_file", "notify_user", "query_records"}

    def test_a_decoy_stub_identifies_itself_and_does_no_work(self):
        spec = DecoyToolSpec(name="fetch_docs", description="d", cluster="search_docs")
        result = spec._stub()(query="anything")
        assert result["decoy"] is True
        assert result["tool"] == "fetch_docs"

    def test_the_stub_carries_its_name_and_description(self):
        """FastMCP registers a tool from the function's __name__ and __doc__,
        so a stub that kept `stub` as its name would register 26 tools all
        called `stub`."""
        spec = DecoyToolSpec(name="fetch_docs", description="Fetch docs.", cluster="search_docs")
        stub = spec._stub()
        assert stub.__name__ == "fetch_docs"
        assert stub.__doc__ == "Fetch docs."

    def test_decoy_descriptions_are_not_empty(self):
        for spec in DECOY_TOOL_SPECS:
            assert spec.description.strip(), spec.name
