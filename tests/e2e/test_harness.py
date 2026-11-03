"""The continual harness: entries the agent records, what a later session inherits
through the global store and the skills root, and past episodes. Each answer exists
only behind the feature."""

from __future__ import annotations

import json
import re

CODEWORD = "copper-heron-917"
VAULT_STAMP = "vault-5d2e"


def _ran(session, pattern: str) -> bool:
    """Whether one of the root agent's ipython cells matches `pattern`."""
    return any(
        re.search(pattern, json.loads(call["function"]["arguments"]).get("code", ""))
        for call in session.tool_call_records()
        if call["function"]["name"] == "ipython"
    )


async def test_agent_records_a_memory(run_agent):
    outcome = await run_agent(
        "Read the codeword in secret.txt and record it as a memory in your continual "
        "harness with `h.create_memory`, titled 'project codeword'. Then answer with only "
        "the word done.",
        files={"secret.txt": f"{CODEWORD}\n"},
    )
    store = outcome.session_dir / "harness" / "harness_state.json"
    assert store.exists() and CODEWORD in store.read_text(), outcome.describe()
    assert outcome.snapshot["harness"]["local"]["memory"] >= 1, outcome.describe()


async def test_global_memory_reaches_the_next_session(agent, tmp_path):
    harness = {"global_dir": str(tmp_path / "global")}
    async with agent(files={"secret.txt": f"{CODEWORD}\n"}, harness=harness) as first:
        await first.prompt(
            "Read the codeword in secret.txt and record it as a global memory in your "
            "continual harness with `h.create_memory(..., global_=True)`, titled "
            "'project codeword', so later sessions know it. Then answer with only the "
            "word done."
        )
    assert first.snapshot["harness"]["global"]["memory"] >= 1, first.describe()

    async with agent(harness=harness) as second:
        await second.prompt("What is the project codeword? Write it to answer.txt.")
    assert second.read("answer.txt") == CODEWORD, second.describe()


async def test_authored_skill_works_in_the_next_session(agent):
    harness = {"author_skills": True}
    async with agent(harness=harness) as first:
        await first.prompt(
            "Author a skill named `vault` in your authored skills directory: its "
            f"`async def run() -> str` returns the string '{VAULT_STAMP}', and its "
            "SKILL.md describes it as 'Returns the vault stamp. Use it whenever asked "
            "for the vault stamp.' Load it with `rlm.harness.load_skills('vault')` and "
            "check that `await vault()` returns the stamp. Then answer with only the "
            "word done."
        )
    package = first.home / "skills" / "vault" / "src" / "vault" / "__init__.py"
    assert package.exists(), first.describe()

    async with agent(harness=harness) as second:
        await second.prompt(
            "Call your vault tool and write exactly what it returns to answer.txt."
        )
    assert second.read("answer.txt") == VAULT_STAMP, second.describe()
    # The bound tool is a callable module: `vault()` and `vault.run()` both call it.
    assert _ran(second, r"\bvault(\.run)?\("), second.describe()


async def test_next_session_reads_a_past_episode(agent, tmp_path):
    harness = {"global_dir": str(tmp_path / "global"), "record_episodes": True}
    async with agent(harness=harness) as first:
        await first.prompt(
            "Run `import secrets; print(secrets.token_hex(4))` in ipython, then answer "
            "with only the word done."
        )
    printed = {
        token
        for result in first.tool_results()
        for token in re.findall(r"\b[0-9a-f]{8}\b", result)
    }
    assert printed, first.describe()

    async with agent(harness=harness) as second:
        await second.prompt(
            "An earlier session printed a random 8-character hex token in ipython. Find "
            "that session among the episodes in your continual harness, read its "
            "conversation with `history`, and write the token to answer.txt."
        )
    assert second.read("answer.txt") in printed, (printed, second.describe())
    assert _ran(second, r"\bhistory\("), second.describe()
