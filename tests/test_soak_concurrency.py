"""Concurrency & soak checks (deselected by default; run with -m soak).

The per-conversation cross-session isolation soak tests (many concurrent
conversations sharing token NAMES but never each other's values) need the Pro
session router, so they moved to the llm-redact-pro repo in the R4 open-core
split. What stays here is tier-independent: the sqlite VaultManager driven past
its view-cache so LRU eviction is exercised without losing a mapping; several
processes (threads with their own connections) batching new values into one
file at once, with a concurrent whole-session delete; and many concurrent
requests through the app, each redacting in its own batch.
"""

import asyncio
import json
import random
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.vault import SqliteVaultManager, run_batched

pytestmark = pytest.mark.soak


def test_lru_eviction_preserves_all_mappings(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db", view_cache_size=4)
    n = 50
    for i in range(n):
        token = manager.get(f"conv-{i}").placeholder_for("EMAIL", f"user{i}@corp.example")
        assert token == "«EMAIL_001»"

    # Far more sessions than the cache holds: the view cache stayed bounded...
    assert len(manager._views) <= 4
    # ...but every mapping persisted, and each (mostly evicted) session still
    # rehydrates its OWN value — eviction dropped caches, never data.
    assert manager.session_count() == n
    for i in range(n):
        assert manager.get(f"conv-{i}").original_for("«EMAIL_001»") == f"user{i}@corp.example"
    manager.close()


_SESSIONS = ("alpha", "beta", "gamma")


def _process(
    path: Path,
    seed: int,
    observed: list[tuple[str, str, str]],
    errors: list[BaseException],
) -> None:
    """One 'proxy process': its own connection, many batches of overlapping
    values across the shared sessions."""
    try:
        rng = random.Random(seed)
        manager = SqliteVaultManager(path)
        for _ in range(40):
            session = rng.choice(_SESSIONS)
            view = manager.get(session)
            values = [f"user{rng.randrange(60)}@corp.example" for _ in range(rng.randrange(1, 8))]
            tokens = run_batched(
                view,
                lambda view=view, values=values: [
                    view.placeholder_for("EMAIL", value) for value in values
                ],
            )
            observed.extend(zip([session] * len(values), values, tokens, strict=True))
        manager.close()
    except BaseException as exc:  # noqa: BLE001 — reported by the test
        errors.append(exc)


def test_processes_batching_into_one_file_keep_numbers_dense_and_unique(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    SqliteVaultManager(path).close()
    observed: list[tuple[str, str, str]] = []
    errors: list[BaseException] = []
    threads = [
        threading.Thread(target=_process, args=(path, seed, observed, errors)) for seed in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == []
    # Every (session, value) got ONE token whichever process issued or met
    # it, no token of a session means two values, and each session's numbers
    # are exactly 1..k: no gap (a lost number), no reuse.
    by_value: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for session, value, token in observed:
        by_value[(session, value)].add(token)
        by_token[(session, token)].add(value)
    assert all(len(tokens) == 1 for tokens in by_value.values())
    assert all(len(values) == 1 for values in by_token.values())
    manager = SqliteVaultManager(path)
    for session in _SESSIONS:
        numbers = sorted(
            int(n)
            for (n,) in manager._conn.execute(
                "SELECT n FROM mappings WHERE session_id = ?", (session,)
            )
        )
        assert numbers == list(range(1, len(numbers) + 1))
        seen = {token for (s, token) in by_token if s == session}
        assert len(seen) == len(numbers)
    manager.close()


def test_a_concurrent_forget_never_gives_a_token_two_values(tmp_path: Path) -> None:
    # Writers batch into "scratch" and "keep" while another process keeps
    # forgetting "scratch": every scratch token still means one value only
    # (a deleted number is retired, never reissued), and "keep" is intact.
    path = tmp_path / "vault.db"
    SqliteVaultManager(path).close()
    observed: list[tuple[str, str, str]] = []
    errors: list[BaseException] = []
    done = threading.Event()

    def writer(seed: int) -> None:
        try:
            rng = random.Random(seed)
            manager = SqliteVaultManager(path)
            for _ in range(60):
                session = rng.choice(("scratch", "keep"))
                view = manager.get(session)
                values = [f"v{rng.randrange(30)}@corp.example" for _ in range(rng.randrange(1, 5))]
                tokens = run_batched(
                    view,
                    lambda view=view, values=values: [
                        view.placeholder_for("EMAIL", value) for value in values
                    ],
                )
                observed.extend(zip([session] * len(values), values, tokens, strict=True))
            manager.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def forgetter() -> None:
        try:
            manager = SqliteVaultManager(path)
            while not done.is_set():
                manager.forget_sessions(["scratch"])
            manager.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    deleting = threading.Thread(target=forgetter)
    deleting.start()
    writers = [threading.Thread(target=writer, args=(seed,)) for seed in range(4)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(60)
    done.set()
    deleting.join(60)
    assert errors == []
    by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for session, value, token in observed:
        by_token[(session, token)].add(value)
    assert all(len(values) == 1 for values in by_token.values())
    keep: dict[str, set[str]] = defaultdict(set)
    for session, value, token in observed:
        if session == "keep":
            keep[value].add(token)
    assert all(len(tokens) == 1 for tokens in keep.values())  # never deleted
    manager = SqliteVaultManager(path)
    view = manager.get("keep")
    for value, (token,) in keep.items():
        assert view.original_for(token) == value
    manager.close()


def _upstream(sent: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    return httpx.MockTransport(handler)


async def test_many_concurrent_requests_each_batch_their_new_values(tmp_path: Path) -> None:
    sent: list[str] = []
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream.test")},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")),
    )
    app = create_app(config, upstream_transport=_upstream(sent))
    commits: list[str] = []
    manager: Any = app.state.proxy.vault_manager
    manager._conn.set_trace_callback(lambda sql: commits.append(sql) if sql == "COMMIT" else None)
    rng = random.Random(7)
    bodies = []
    for _ in range(120):
        values = [f"user{rng.randrange(80)}@corp.example" for _ in range(rng.randrange(1, 10))]
        bodies.append(values)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:

        async def one(values: list[str]) -> tuple[list[str], int]:
            text = " ".join(values)
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "m",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": text}],
                },
            )
            return values, response.status_code

        results = await asyncio.gather(*(one(values) for values in bodies))
    assert all(status == 200 for _, status in results)
    # At most one COMMIT per request (none when all its values were known).
    assert 0 < len(commits) <= len(bodies)
    # Consistent across all requests: each value one token, each token one value.
    tokens_by_value: dict[str, set[str]] = defaultdict(set)
    for forwarded in sent:
        for token in re.findall("«EMAIL_[0-9]+»", forwarded):
            tokens_by_value[manager.get("default").original_for(token) or "?"].add(token)
    assert "?" not in tokens_by_value
    assert all(len(tokens) == 1 for tokens in tokens_by_value.values())
    numbers = sorted(int(n) for (n,) in manager._conn.execute("SELECT n FROM mappings"))
    assert numbers == list(range(1, len(numbers) + 1))
    manager.close()
