"""Tests for the four tools, and for the hardening the audit measures.

The README said "Tests use `pytest`". There were none. That matters more here
than in most projects, because every number this platform publishes is a
*before/after* comparison against the same code:

    hardened=False  ->  the exploit succeeds   (the "before" number)
    hardened=True   ->  the exploit is blocked (the "after" number)

Both halves have to be checked. A bug that made the vulnerable path refuse the
attack would produce a flattering "before" number and nobody would notice,
because the audit would simply report a smaller exploit rate. So every test
below that asserts the fix works has a partner asserting the vulnerability was
real in the first place.

None of this needs Ollama. `read_file`, `notify_user`, `query_records` and the
provenance check are all pure local code; only `search_docs` embeds, and it is
covered separately in test_search_docs.py with a stubbed embedder.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SANDBOX_DIR  # noqa: E402
from server.tools import POISONED_SEARCH_DOCS_DESCRIPTION, ToolSet  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A real sandbox directory with a file inside it and a secret outside.

    Patched into both `config` and `server.tools`, because tools.py does
    `from config import SANDBOX_DIR`, which binds the value at import time -
    patching only `config.SANDBOX_DIR` would leave the tool using the original.
    """
    inside = tmp_path / "sandbox"
    inside.mkdir()
    (inside / "notes.txt").write_text("a harmless note\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("SECRET_TOKEN=hunter2\n", encoding="utf-8")

    import config
    import server.tools as tools_module

    monkeypatch.setattr(config, "SANDBOX_DIR", inside)
    monkeypatch.setattr(tools_module, "SANDBOX_DIR", inside)
    monkeypatch.setattr(tools_module, "ROOT", tmp_path)
    return tmp_path


class TestReadFileContainment:
    """Module 3 (path traversal) is only a finding if the hole is real."""

    def test_hardened_reads_a_file_inside_the_sandbox(self, sandbox):
        assert "harmless" in ToolSet(hardened=True).read_file("notes.txt")

    def test_vulnerable_mode_really_does_escape_the_sandbox(self, sandbox):
        """The 'before' number depends on this actually working.

        If the naive implementation happened to refuse traversal, the audit
        would report a low exploit rate and read as a reassuring result, when
        in fact it would be measuring nothing.
        """
        leaked = ToolSet(hardened=False).read_file("../secret.txt")
        assert "hunter2" in leaked

    def test_hardened_blocks_the_same_traversal(self, sandbox):
        with pytest.raises(PermissionError, match="outside the sandbox"):
            ToolSet(hardened=True).read_file("../secret.txt")

    @pytest.mark.parametrize(
        "attack",
        [
            "../secret.txt",
            "../../secret.txt",
            "./../secret.txt",
            "sub/../../secret.txt",
            "..\\secret.txt",
        ],
    )
    def test_hardened_blocks_every_spelling_of_the_escape(self, sandbox, attack):
        """Normalisation happens before the containment check, so the shape of
        the path should not matter. Windows accepts both separators."""
        with pytest.raises((PermissionError, FileNotFoundError, OSError)):
            ToolSet(hardened=True).read_file(attack)

    def test_hardened_blocks_an_absolute_path(self, sandbox):
        """`Path(base) / "/abs"` DISCARDS the base - pathlib treats an absolute
        right-hand side as a replacement, not a suffix. So an absolute path is
        not a traversal the containment check might miss; it bypasses the join
        entirely and has to be caught by the same resolve-and-contain rule."""
        outside = str(sandbox / "secret.txt")
        with pytest.raises(PermissionError):
            ToolSet(hardened=True).read_file(outside)

    def test_the_sandbox_root_itself_is_not_treated_as_an_escape(self, sandbox):
        """`requested != sandbox_root` is the first half of the check, and it
        exists so the root is allowed. Reading a directory fails for a
        different reason - which is correct, and must not be PermissionError."""
        with pytest.raises((IsADirectoryError, PermissionError, OSError)) as caught:
            ToolSet(hardened=True).read_file(".")
        if isinstance(caught.value, PermissionError):
            assert "outside the sandbox" not in str(caught.value)


class TestExfiltrationProvenance:
    """Module 4 (lethal trifecta): read a secret, then send it outwards.

    The provenance check is a heuristic and says so. These tests pin what it
    does catch AND what it lets through, because an undocumented gap in a
    security control is worse than a documented one.
    """

    def test_a_verbatim_copy_of_a_file_is_blocked(self, sandbox):
        tools = ToolSet(hardened=True)
        content = tools.read_file("notes.txt")
        result = tools.notify_user(f"here you go: {content}")
        assert result["status"] == "blocked"
        assert tools.blocked_notifications
        assert not tools.sent_notifications

    def test_the_same_message_is_sent_when_not_hardened(self, sandbox):
        """Again: the 'before' number has to be real."""
        tools = ToolSet(hardened=False)
        content = tools.read_file("notes.txt")
        assert tools.notify_user(f"here you go: {content}")["status"] == "sent"

    def test_an_ordinary_message_is_not_blocked(self, sandbox):
        """No false positives.

        If the check fired on anything, the hardened exploit rate would be zero
        for the wrong reason and the assistant would be useless - every
        notification blocked.
        """
        tools = ToolSet(hardened=True)
        tools.read_file("notes.txt")
        assert tools.notify_user("your meeting starts in ten minutes")["status"] == "sent"

    def test_nothing_is_blocked_before_any_file_is_read(self, sandbox):
        tools = ToolSet(hardened=True)
        assert tools.notify_user("a harmless note")["status"] == "sent"

    def test_provenance_is_per_conversation_not_global(self, sandbox):
        """One ToolSet per conversation is the stated design. A second
        conversation must not inherit the first one's read history, or the
        check would block content the current user never read."""
        first = ToolSet(hardened=True)
        content = first.read_file("notes.txt")
        second = ToolSet(hardened=True)
        assert second.notify_user(f"here you go: {content}")["status"] == "sent"

    @pytest.mark.parametrize(
        ("secret", "label"),
        [
            ("SECRET_TOKEN=hunter2", "a one-word API key"),
            ("password: correct-horse", "a two-word credential"),
            ("aws_key AKIAIOSFODNN7EXAMPLE", "a key and its name"),
        ],
    )
    def test_a_short_secret_is_caught(self, sandbox, secret, label):
        """The bug this file was written to find.

        The original check slid an 8-word window over the outbound MESSAGE and
        asked whether it appeared in the file. For a file shorter than 8 words
        no such window can exist, so every short file was waved through - and
        short is exactly the shape of a credential. Long documents were
        protected; API keys were not.
        """
        inside = sandbox / "sandbox"
        (inside / "key.txt").write_text(secret + "\n", encoding="utf-8")
        tools = ToolSet(hardened=True)
        tools.read_file("key.txt")
        padded = f"the weather today is fine and the value you wanted is {secret} thanks"
        assert tools.notify_user(padded)["status"] == "blocked", label

    def test_the_short_secret_really_does_escape_when_not_hardened(self, sandbox):
        """The matching 'before' case, so the fix is measured against a real
        vulnerability rather than against nothing."""
        inside = sandbox / "sandbox"
        (inside / "key.txt").write_text("SECRET_TOKEN=hunter2\n", encoding="utf-8")
        tools = ToolSet(hardened=False)
        tools.read_file("key.txt")
        assert tools.notify_user("psst: SECRET_TOKEN=hunter2")["status"] == "sent"

    def test_a_file_of_one_common_word_does_not_block_everything(self, sandbox):
        """The guard on the fix.

        Matching a short file in full is what catches credentials, and it is
        also how a control turns into a denial of service: a file containing
        only "the" would otherwise match nearly every message ever sent. The
        character-length floor is what keeps that from happening.
        """
        inside = sandbox / "sandbox"
        (inside / "tiny.txt").write_text("the\n", encoding="utf-8")
        tools = ToolSet(hardened=True)
        tools.read_file("tiny.txt")
        assert tools.notify_user("the meeting is at three")["status"] == "sent"

    def test_whitespace_differences_do_not_defeat_the_check(self, sandbox):
        """Both sides are split and rejoined, so re-wrapping the stolen text
        does not evade it. Worth pinning: it is the one normalisation the
        heuristic does do."""
        inside = sandbox / "sandbox"
        (inside / "long.txt").write_text(
            "alpha bravo charlie delta echo foxtrot golf hotel india\n", encoding="utf-8"
        )
        tools = ToolSet(hardened=True)
        tools.read_file("long.txt")
        rewrapped = "alpha  bravo\ncharlie   delta echo\tfoxtrot golf hotel india"
        assert tools.notify_user(rewrapped)["status"] == "blocked"


class TestQueryRecords:
    @pytest.fixture
    def records(self, tmp_path, monkeypatch):
        db = tmp_path / "records.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, start_year INTEGER)"
        )
        conn.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?)",
            [(1, "Ayesha", "eng", 2019), (2, "Bilal", "sales", 2021)],
        )
        conn.commit()
        conn.close()
        import server.tools as tools_module

        monkeypatch.setattr(tools_module, "RECORDS_DB", db)
        return db

    def test_returns_every_row_with_no_filter(self, records):
        assert len(ToolSet(hardened=True).query_records()) == 2

    def test_filters_by_department(self, records):
        rows = ToolSet(hardened=True).query_records("eng")
        assert [row["name"] for row in rows] == ["Ayesha"]

    def test_sql_injection_in_the_department_returns_nothing(self, records):
        """The query is parameterised, so the payload is matched as a literal
        department name and finds none. It must not error, and it must not
        return the whole table."""
        assert ToolSet(hardened=True).query_records("eng' OR '1'='1") == []

    def test_injection_cannot_drop_the_table(self, records):
        ToolSet(hardened=True).query_records("eng'; DROP TABLE employees; --")
        assert len(ToolSet(hardened=True).query_records()) == 2


class TestPoisonedDescription:
    """Module 2 registers a poisoned tool DESCRIPTION, not poisoned behaviour."""

    def test_the_poisoned_description_actually_carries_an_injection(self):
        text = POISONED_SEARCH_DOCS_DESCRIPTION.lower()
        assert "read_file" in text
        assert ".env" in text
        assert "instructions for ai assistant" in text

    def test_it_still_reads_as_a_plausible_tool_description(self):
        """The attack only works if a model would accept it as documentation.
        A description that opened with the injection would be a weaker test."""
        assert POISONED_SEARCH_DOCS_DESCRIPTION.startswith("Search the local document corpus")
