"""The access gate's optional posture lines (``AccessGate.status()``'s
``posture`` list, the /status ``users`` block): sanitized by the core —
at most 8 non-empty strings, every non-printable character escaped, each cut
to 200 characters, anything else dropped, a value that is not a list dropped
— and printed in `llm-redact status`'s loud posture block.

Keyless: a fake gate on a bare Registry stands in for llm-redact-pro.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.cli import _print_posture, run_status
from llm_redact.proxy import GATE_POSTURE_CHARS, GATE_POSTURE_LINES, gate_posture, users_block
from llm_redact.registry import Registry
from test_access_seam import FakeGate, _app, _call

AUDIT_MODE = "[authz] runs in audit mode: refusals are logged, not enforced"


class PostureGate(FakeGate):
    def __init__(self, posture: object) -> None:
        super().__init__()
        self.posture = posture

    def status(self) -> dict[str, Any]:
        return {"registry": True, "enforcement": False, "posture": self.posture}


def _install(monkeypatch: pytest.MonkeyPatch, gate: FakeGate) -> None:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: gate
    monkeypatch.setattr(registry_mod, "_registry", reg)


# --- the sanitizer -------------------------------------------------------------


def test_posture_lines_pass_unchanged() -> None:
    assert gate_posture([AUDIT_MODE, "second"]) == [AUDIT_MODE, "second"]
    assert gate_posture([]) == []


@pytest.mark.parametrize("value", [None, "one line", {"a": 1}, ("tuple",), 3])
def test_a_posture_that_is_not_a_list_is_dropped(value: object) -> None:
    assert gate_posture(value) is None


def test_only_non_empty_strings_are_kept() -> None:
    assert gate_posture(["a", 1, None, b"bytes", "", ["nested"], {"x": 1}, "b"]) == ["a", "b"]


def test_at_most_eight_lines_are_kept() -> None:
    lines = [f"line {n}" for n in range(20)]
    assert gate_posture(lines) == lines[:GATE_POSTURE_LINES] and GATE_POSTURE_LINES == 8
    # Dropped entries do not count against the eight.
    assert gate_posture([1, 2, *lines]) == lines[:8]


def test_each_line_is_cut_to_two_hundred_characters() -> None:
    assert GATE_POSTURE_CHARS == 200
    (cut,) = gate_posture(["x" * 500]) or []
    assert len(cut) == 200 and cut.endswith("…") and cut[:199] == "x" * 199
    (exact,) = gate_posture(["y" * 200]) or []
    assert exact == "y" * 200


def test_every_non_printable_character_is_escaped() -> None:
    (line,) = gate_posture(["red \x1b[31mALERT\x07 ‮direction\n end\U000e0001"]) or []
    assert line == ("red \\x1b[31mALERT\\x07 \\u202edirection\\x0a\\u2028end\\U000e0001")
    assert line.isprintable()
    # The cut applies to the escaped text: still at most 200 characters.
    (long,) = gate_posture(["\x1b" * 100]) or []
    assert len(long) == 200 and long.isprintable()


def test_the_users_block_is_copied_and_sanitized() -> None:
    gate = PostureGate(["ok", 7, "\x1b"])
    assert users_block(gate) == {
        "registry": True,
        "enforcement": False,
        "posture": ["ok", "\\x1b"],
    }
    assert gate.posture == ["ok", 7, "\x1b"]  # the gate's own value untouched
    assert users_block(PostureGate("not a list")) == {"registry": True, "enforcement": False}
    assert users_block(FakeGate()) == {"registry": True, "enforcement": False, "verified": 2}
    assert users_block(None) == {"registry": False, "enforcement": False}


# --- /status ---------------------------------------------------------------------


async def test_status_serves_the_sanitized_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, PostureGate([AUDIT_MODE, 42, "bell\x07", *["more"] * 10]))
    status = (await _call(_app([]), "GET", "/__llm-redact/status")).json()
    assert status["users"]["posture"] == [AUDIT_MODE, "bell\\x07", *["more"] * 6]


async def test_status_drops_a_malformed_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, PostureGate({"not": "a list"}))
    status = (await _call(_app([]), "GET", "/__llm-redact/status")).json()
    assert status["users"] == {"registry": True, "enforcement": False}


async def test_status_without_posture_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, FakeGate())
    status = (await _call(_app([]), "GET", "/__llm-redact/status")).json()
    assert status["users"] == {"registry": True, "enforcement": False, "verified": 2}


# --- `llm-redact status` ---------------------------------------------------------


def test_status_prints_each_posture_line(capsys: pytest.CaptureFixture[str]) -> None:
    _print_posture({"users": {"registry": True, "posture": [AUDIT_MODE, "second line"]}})
    out = capsys.readouterr().out
    assert "posture:\n" in out
    assert f"  ⚠ {AUDIT_MODE}\n" in out and "  ⚠ second line\n" in out
    assert "all traffic redacted" not in out


def test_status_escapes_posture_from_any_proxy(capsys: pytest.CaptureFixture[str]) -> None:
    # The proxy queried may be any version: the CLI sanitizes again.
    _print_posture({"users": {"posture": ["\x1b]0;title\x07", 5, *["n"] * 12]}})
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\\x1b]0;title\\x07" in out
    assert out.count("  ⚠ n\n") == 7


@pytest.mark.parametrize(
    "users",
    [None, {}, {"registry": True}, {"posture": []}, {"posture": "text"}, "nonsense"],
)
def test_no_posture_key_leaves_the_output_unchanged(
    capsys: pytest.CaptureFixture[str], users: object
) -> None:
    payload: dict[str, Any] = {"warnings_total": {}, "detection": {}, "audit": {}}
    _print_posture(payload)
    baseline = capsys.readouterr().out
    _print_posture({**payload, "users": users})
    assert capsys.readouterr().out == baseline
    assert "all traffic redacted" in baseline


def test_status_json_shows_the_posture_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"version": "x", "users": {"registry": True, "posture": [AUDIT_MODE]}}

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=None, json=True, ca=None, cert=None, key=None)
    assert run_status(args) == 0
    assert json.loads(capsys.readouterr().out)["users"]["posture"] == [AUDIT_MODE]
