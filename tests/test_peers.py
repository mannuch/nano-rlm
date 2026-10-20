"""Peer roots, mesh messaging, promotion and the shared scratchpad."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result
from rlm.broker import parse_request
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor, _has_waking_events
from rlm.tools import get_builtin_tool
from rlm.types import RLMResult


def _config(*, max_peers: int = 2, max_depth: int = 1) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url="http://interceptor", api_key="secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=max_depth, max_subagent_calls=8, max_peers=max_peers
        ),
    )


def _factory(calls: list[tuple[str, str, str]]):
    class _Engine:
        def __init__(
            self, *, runtime_config, session, supervisor, invocation_id, **kwargs
        ):
            self.runtime_config = runtime_config
            self.session = session
            self.supervisor = supervisor
            self.invocation_id = invocation_id

        async def prompt(self, prompt: str, message_type: str = "user", **kwargs):
            # A real engine announces its inbox every turn, which settles pending wakes.
            self.supervisor.inbox_notice(self.invocation_id)
            calls.append((self.invocation_id, prompt, message_type))
            return RLMResult(
                answer=f"answer:{prompt}", session_dir=self.session.dir, turns=1
            )

        async def aclose(self):
            self.session.close()

    return _Engine


async def _supervisor(tmp_path, calls, **config):
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(**config),
        cwd=str(tmp_path),
        engine_factory=_factory(calls),
    )
    await supervisor.start()
    return supervisor, session


async def _op(supervisor, agent, scope, op, **arguments):
    request = parse_request(
        {"op": op, "capability": agent.capability, "scope_id": scope, **arguments}
    )
    return await supervisor._agent_operation(request)


async def _spawn_peer(supervisor, owner, name):
    scope = await supervisor.open_scope(owner.id)
    info = await _op(supervisor, owner, scope, "peer.spawn", task=name, name=name)
    peer = supervisor._invocations[info["id"]]
    await asyncio.wait_for(peer.done.wait(), 10)
    return peer


async def test_mesh_disabled_rejects_peer_operations(tmp_path):
    supervisor, session = await _supervisor(tmp_path, [], max_peers=0)
    root = supervisor._invocations[supervisor.root_id]
    try:
        scope = await supervisor.open_scope(root.id)
        with pytest.raises(PermissionError, match="mesh is disabled"):
            await _op(supervisor, root, scope, "peer.spawn", task="t", name="p")
        assert supervisor.scratchpad_dir is None
        assert supervisor.scratchpad_env() == {}
    finally:
        await supervisor.aclose()
        session.close()


async def test_spawned_peers_are_flat_persistent_roots(tmp_path):
    calls: list[tuple[str, str, str]] = []
    supervisor, session = await _supervisor(tmp_path, calls)
    root = supervisor._invocations[supervisor.root_id]
    try:
        reviewer = await _spawn_peer(supervisor, root, "reviewer")
        tester = await _spawn_peer(supervisor, reviewer, "tester")
        # tester's answer reaches its owner as peer.completed, which wakes reviewer.
        await asyncio.wait_for(reviewer.done.wait(), 10)

        for peer, owner in ((reviewer, root), (tester, reviewer)):
            assert peer.status == "idle"
            assert (peer.kind, peer.owner_id, peer.parent_id) == (
                "peer",
                owner.id,
                None,
            )
            assert peer.runtime_config.invocation.depth == 0
            assert peer.session.dir.parent == session.dir
            assert peer.session.dir.name.startswith("peer-")
        assert [c[2] for c in calls] == [
            "peer_task",
            "peer_task",
            "supervisor_notification",
        ]

        completed = [e for e in root.inbox if e["type"] == "peer.completed"]
        # One per answer: reviewer's task, then the turn tester's completion woke.
        assert [e["content"]["status"] for e in completed] == ["idle", "idle"]
        mesh = [
            json.loads(line)
            for line in (session.dir / "mesh.jsonl").read_text().splitlines()
        ]
        assert [(r["name"], r["dir"]) for r in mesh] == [
            ("reviewer", reviewer.session.dir.name),
            ("tester", tester.session.dir.name),
        ]
        spawns = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
            if '"peer_spawn"' in line
        ]
        assert spawns[0]["peer_dir"] == reviewer.session.dir.name

        scope = await supervisor.open_scope(root.id)
        for name, error in (("tester", ValueError), ("root", ValueError)):
            with pytest.raises(error, match="reserved"):
                await _op(supervisor, root, scope, "peer.spawn", task="t", name=name)
        with pytest.raises(RuntimeError, match="peer limit"):
            await _op(supervisor, root, scope, "peer.spawn", task="t", name="third")
        listed = await _op(supervisor, root, scope, "peer.list", all=False)
        assert [a["kind"] for a in listed] == ["root", "peer", "peer"]
        assert supervisor.scratchpad_dir == session.dir / "scratchpad"
        assert supervisor.scratchpad_dir.is_dir()
        assert supervisor.scratchpad_env() == {
            "RLM_SCRATCHPAD": str(session.dir / "scratchpad")
        }
    finally:
        await supervisor.aclose()
        session.close()


async def test_messages_wake_idle_agents_unless_quiet(tmp_path):
    calls: list[tuple[str, str, str]] = []
    supervisor, session = await _supervisor(tmp_path, calls)
    root = supervisor._invocations[supervisor.root_id]
    try:
        peer = await _spawn_peer(supervisor, root, "reviewer")
        scope = await supervisor.open_scope(root.id)
        quiet = await _op(
            supervisor,
            root,
            scope,
            "peer.send",
            agent_id="reviewer",
            message="fyi",
            in_reply_to=None,
            wake=False,
        )
        await asyncio.sleep(0.05)
        assert peer.status == "idle" and len(calls) == 1
        assert peer.inbox[-1]["id"] == quiet and peer.inbox[-1]["wake"] is False
        assert not _has_waking_events(peer)

        await _op(
            supervisor,
            root,
            scope,
            "peer.send",
            agent_id=peer.id,
            message="question",
            in_reply_to=quiet,
            wake=True,
        )
        await asyncio.wait_for(peer.done.wait(), 10)
        assert peer.status == "idle" and len(calls) == 2
        assert calls[-1][2] == "supervisor_notification"
        assert peer.inbox[-1]["content"] == {
            "from_id": root.id,
            "from_name": None,
            "text": "question",
            "in_reply_to": quiet,
        }
        with pytest.raises(ValueError, match="yourself"):
            await _op(
                supervisor,
                root,
                scope,
                "peer.send",
                agent_id="root",
                message="me",
                in_reply_to=None,
                wake=True,
            )
        stats = supervisor.mesh_stats()
        assert (stats["peer_messages"], stats["peer_messages_nowake"]) == (2, 1)
    finally:
        await supervisor.aclose()
        session.close()


async def test_agents_at_any_depth_message_across_trees(tmp_path):
    supervisor, session = await _supervisor(tmp_path, [])
    root = supervisor._invocations[supervisor.root_id]
    try:
        root_scope = await supervisor.open_scope(root.id)
        root_child = supervisor._spawn(root, root_scope, "a", "a", True)
        peer = await _spawn_peer(supervisor, root, "reviewer")
        peer_scope = await supervisor.open_scope(peer.id)
        peer_child = supervisor._spawn(peer, peer_scope, "b", "b", True)
        await asyncio.wait_for(root_child.done.wait(), 10)
        await asyncio.wait_for(peer_child.done.wait(), 10)

        scope = await supervisor.open_scope(peer_child.id)
        info = await _op(
            supervisor, peer_child, scope, "peer.get", name_or_id=root_child.id
        )
        assert info["depth"] == 1 and info["kind"] == "child"
        await _op(
            supervisor,
            peer_child,
            scope,
            "peer.send",
            agent_id=root_child.id,
            message="hello",
            in_reply_to=None,
            wake=False,
        )
        event = root_child.inbox[-1]
        assert event["type"] == "peer.message"
        assert event["content"]["from_id"] == peer_child.id
        everyone = await _op(supervisor, peer_child, scope, "peer.list", all=True)
        assert {a["id"] for a in everyone} == {
            root.id,
            root_child.id,
            peer.id,
            peer_child.id,
        }
    finally:
        await supervisor.aclose()
        session.close()


async def test_only_owner_or_root_cancels_a_peer(tmp_path):
    supervisor, session = await _supervisor(tmp_path, [])
    root = supervisor._invocations[supervisor.root_id]
    try:
        owner = await _spawn_peer(supervisor, root, "owner")
        owned = await _spawn_peer(supervisor, owner, "owned")
        root_scope = await supervisor.open_scope(root.id)
        bystander = supervisor._spawn(root, root_scope, "x", "x", True)
        await asyncio.wait_for(bystander.done.wait(), 10)

        scope = await supervisor.open_scope(bystander.id)
        with pytest.raises(PermissionError, match="owner or the root"):
            await _op(supervisor, bystander, scope, "peer.cancel", agent_id=owned.id)
        info = await _op(supervisor, root, root_scope, "peer.cancel", agent_id="owned")
        assert info["status"] == "cancelled"
        assert owner.inbox[-1]["type"] == "peer.completed"
        assert owner.inbox[-1]["content"]["status"] == "cancelled"
        with pytest.raises(PermissionError, match="only peers"):
            await _op(
                supervisor, root, root_scope, "peer.cancel", agent_id=bystander.id
            )
    finally:
        await supervisor.aclose()
        session.close()


async def test_promoted_child_detaches_and_outlives_former_parent(tmp_path):
    supervisor, session = await _supervisor(tmp_path, [])
    root = supervisor._invocations[supervisor.root_id]
    try:
        parent = await _spawn_peer(supervisor, root, "parent")
        parent_scope = await supervisor.open_scope(parent.id)
        child = supervisor._spawn(parent, parent_scope, "c", "c", False)
        await asyncio.wait_for(child.done.wait(), 10)
        assert child.status == "completed"
        with pytest.raises(RuntimeError, match="no longer active"):
            await _op(
                supervisor,
                parent,
                parent_scope,
                "agent.promote",
                agent_id=child.id,
                name="late",
            )

        child = supervisor._spawn(parent, parent_scope, "s", "s", True)
        await asyncio.wait_for(child.done.wait(), 10)
        info = await _op(
            supervisor,
            parent,
            parent_scope,
            "agent.promote",
            agent_id="s",
            name="specialist",
        )
        assert (info["kind"], info["owner_id"], info["parent_id"], info["depth"]) == (
            "peer",
            parent.id,
            None,
            0,
        )
        assert supervisor.take_promotion(child.id) == {
            "name": "specialist",
            "owner_id": parent.id,
        }
        assert supervisor.take_promotion(child.id) is None
        with pytest.raises(PermissionError):
            supervisor._child(parent, child.id)
        mesh = (session.dir / "mesh.jsonl").read_text().splitlines()
        promoted = json.loads(mesh[-1])
        assert promoted["type"] == "promoted"
        assert promoted["dir"] == f"{parent.session.dir.name}/{child.session.dir.name}"

        root_scope = await supervisor.open_scope(root.id)
        await _op(supervisor, root, root_scope, "peer.cancel", agent_id="parent")
        assert parent.status == "cancelled"
        assert child.status == "idle" and not child.released
        target = await _op(
            supervisor, root, root_scope, "peer.get", name_or_id="specialist"
        )
        assert target["id"] == child.id
    finally:
        await supervisor.aclose()
        session.close()
    assert child.status == "cancelled"


async def test_root_aggregates_each_peer_session_once(tmp_path):
    session = Session(tmp_path / "root")
    try:
        child = Session.child_dir(session.dir)
        Session.child_dir(child)
        peer = Session.child_dir(session.dir, prefix="peer")
        Session.child_dir(peer)
        assert session.aggregate_child_metrics().num_sessions == 4
    finally:
        session.close()


def test_peer_prompt_appears_only_with_the_mesh():
    tools = [get_builtin_tool("ipython")]

    def render(**agent_info):
        info = {
            "id": "a",
            "parent_id": None,
            "name": None,
            "persistent": True,
            "kind": "root",
            "owner_id": None,
            "mesh": False,
            **agent_info,
        }
        return build_system_prompt(
            "/work",
            [],
            depth=0 if info["kind"] != "child" else 1,
            allow_recursion=True,
            allow_git=False,
            active_tools=tools,
            agent_info=info,
        )

    assert "## Peers and the mesh" not in render()
    assert "## Peers and the mesh" in render(mesh=True)
    peer = render(mesh=True, kind="peer", name="fe", owner_id="r1")
    assert "You are the peer `fe`" in peer and "You are the root agent" not in peer
    child = render(mesh=True, kind="child", parent_id="r1", persistent=False)
    assert "reach siblings and other agents through rlm.peer" in child


async def test_promoted_engine_rebuilds_its_prompt_at_depth_zero(tmp_path):
    engines: list[RLMEngine] = []

    def factory(**kwargs):
        client = DummyClient(
            [DummyMessage(content="ready"), DummyMessage(content="done")]
        )
        engine = RLMEngine(client=client, **kwargs)
        engines.append(engine)
        return engine

    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_peers=1, max_depth=1),
        cwd=str(tmp_path),
        engine_factory=factory,
    )
    await supervisor.start()
    root = supervisor._invocations[supervisor.root_id]
    try:
        scope = await supervisor.open_scope(root.id)
        child = supervisor._spawn(root, scope, "research", "leaf", True)
        await asyncio.wait_for(child.done.wait(), 30)
        assert (
            "Delegation is disabled at this depth"
            in engines[0].session.messages[0]["content"]
        )
        await _op(
            supervisor, root, scope, "agent.promote", agent_id="leaf", name="spec"
        )
        await _op(
            supervisor,
            root,
            scope,
            "peer.send",
            agent_id="spec",
            message="still there?",
            in_reply_to=None,
            wake=True,
        )
        await asyncio.wait_for(child.done.wait(), 30)
        engine = engines[0]
        assert engine.depth == 0 and child.status == "idle"
        system = engine.session.messages[0]["content"]
        assert "You are the peer `spec`" in system
        assert "Delegation is disabled at this depth" not in system
        records = [
            json.loads(line)
            for line in (child.session.dir / "messages.jsonl").read_text().splitlines()
        ]
        assert any(
            r.get("type") == "context_window" and r.get("reason") == "promotion"
            for r in records
        )
        assert any(r.get("type") == "promotion_notice" for r in records)
    finally:
        await supervisor.aclose()
        session.close()


async def test_real_kernel_uses_peers_and_the_scratchpad(tmp_path):
    session = Session(tmp_path / "root")
    config = _config(max_peers=1)
    calls: list[tuple[str, str, str]] = []
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=_factory(calls),
    )
    code = """
p = await rlm.peer.spawn('own the board', name='board-keeper')
await p.wait(timeout=10)
(rlm.scratchpad / 'board.md').write_text('todo')
job = await rlm.shell.run('cat "$RLM_SCRATCHPAD/board.md"', yield_after=10)
await p.send('board ready', wake=False)
names = [info.name for info in await rlm.peer.list()]
reply = await p.result(yield_after=10)
print(job.text.strip(), names, reply.answer)
"""
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=config,
        supervisor=supervisor,
        invocation_id=supervisor.root_id,
    )
    try:
        result = await engine.run("coordinate")
        stats = supervisor.mesh_stats()
    finally:
        await supervisor.aclose()

    assert result.answer == "done"
    assert tool_result(client).strip() == (
        "todo [None, 'board-keeper'] answer:own the board"
    )
    assert (session.dir / "scratchpad" / "board.md").read_text() == "todo"
    assert stats["peers_spawned"] == 1 and stats["peer_messages_nowake"] == 1
    assert stats["max_live_kernels"] == 1 and stats["kernel_busy_seconds"] > 0
