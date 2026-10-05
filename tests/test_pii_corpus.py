"""The out-of-band corpus tooling in scripts/pii_corpus (plan T40-T43), run
against a fake Ollama server (tests/fake_ollama.py; no network)."""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from fake_ollama import APACHE, DIGEST, FakeOllama

# The tooling is a dev-only package under scripts/, not part of llm_redact.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pii_corpus import audit, generate, grounding, prompts, teacher  # noqa: E402
from pii_corpus.private_files import (  # noqa: E402
    CorpusError,
    default_data_dir,
    open_private,
    output_problem,
    read_jsonl,
)

VALUES = {
    "PERSON": "Ann Lee",
    "ADDRESS": "12 Elm Way",
    "DATE_OF_BIRTH": "1990-04-01",
    "PASSPORT": "X1234567",
    "DRIVER_LICENSE": "D-55-0192",
    "USERNAME": "annlee90",
    "ACCOUNT_NUMBER": "00431977",
    "EMAIL": "ann@example.com",
    "PHONE": "+1 555 0100",
}
_ASKED = re.compile(r"at least once: ([A-Z_, ]+)\.")


def tagged_answer(body: dict[str, Any]) -> str:
    """A well-behaved teacher: every asked type tagged once, negatives clean."""
    user = body["messages"][1]["content"]
    asked = _ASKED.search(user)
    if asked is None:
        return '{"job": "JenkinsBuild", "id": "4f1c2a"}'
    lines = [
        f'"{t.lower()}": "<pii type="{t}">{VALUES[t]}</pii>"' for t in asked.group(1).split(", ")
    ]
    return "{\n" + ",\n".join(lines) + "\n}"


# --- the teacher policy -----------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gemma4:e2b",
        "gemma4:e4b",
        "gemma4:31b",
        "gemma4:12b-it-q8_0",
        "mistral:7b",
        "mistral-nemo:12b",
        "mistral-small:24b",
        "mistral-small3.2:24b",
        "devstral:24b",
        "magistral:24b",
        "ministral-3:8b",
        "mixtral:8x22b",
    ],
)
def test_allowlisted_teachers(model: str) -> None:
    assert teacher.model_refusal(model) is None


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        ("llama3.1:8b", "Llama Community License"),
        ("codellama:7b", "Llama Community License"),
        ("qwen2.5-coder:7b", "Qwen family"),
        ("Qwen3:8b", "Qwen family"),
        ("deepseek-r1:8b", "DeepSeek family"),
        ("gemma:7b", "Gemma Terms of Use"),
        ("gemma2:9b", "Gemma Terms of Use"),
        ("gemma3:4b", "Gemma Terms of Use"),
        ("gemma3n:e4b", "Gemma Terms of Use"),
        ("codegemma:7b", "Gemma Terms of Use"),
        ("codestral:22b", "Non-Production License"),
        ("mistral-large:123b", "Research License"),
        ("mistral-small:22b", "Research License"),
        ("mistral-small:22b-instruct-2409-q4_0", "Research License"),
        ("ministral:8b", "Research License"),
        ("gemma4", "fixed size tag"),
        ("gemma4:latest", "fixed size tag"),
        ("gemma4:", "fixed size tag"),
        ("gemma4:31b-cloud", "cloud tag"),
        ("gemma4:cloud", "cloud tag"),
        ("phi4:14b", "not on the teacher allowlist"),
        ("hf.co/someone/model-GGUF:Q4_K_M", "not on the teacher allowlist"),
        ("mistral-small3.1:24b", "not on the teacher allowlist"),
        ("gemma4:7b", "tag not on the allowlist for gemma4 (allowed: e2b, e4b, 12b, 26b, 31b)"),
    ],
)
def test_refused_teachers(model: str, reason: str) -> None:
    refusal = teacher.model_refusal(model)
    assert refusal is not None and refusal.startswith(f"{model}: ")
    assert reason in refusal


@pytest.mark.parametrize(
    ("url", "allow_remote", "problem"),
    [
        ("http://127.0.0.1:11434", False, None),
        ("http://localhost:11434/", False, None),
        ("http://[::1]:11434", False, None),
        ("https://127.0.0.2:8443", False, None),
        ("http://gpu-box.lan:11434", False, "refusing the non-loopback server gpu-box.lan"),
        ("http://10.0.0.5:11434", True, "must be reached over https"),
        ("https://gpu-box.lan", True, None),
        ("ftp://127.0.0.1", False, "must look like"),
        ("127.0.0.1:11434", False, "must look like"),
        ("http://user:pw@127.0.0.1:11434", False, "credentials"),
        ("http://127.0.0.1:11434/?x=1", False, "credentials, a query"),
        ("http://127.0.0.1:11434/api", False, "root (no path)"),
    ],
)
def test_server_url_policy(url: str, allow_remote: bool, problem: str | None) -> None:
    found = teacher.server_problem(url, allow_remote=allow_remote)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        (APACHE, True),
        ([APACHE, "Gemma notice"], True),
        ("Gemma Terms of Use\nLast modified: February 21, 2024", False),
        ("# Mistral AI Research License\n", False),
        ("Apache License\nVersion 1.1", False),
        (None, False),
        (42, False),
    ],
)
def test_apache_license_check(text: object, ok: bool) -> None:
    assert teacher.apache_license(text) is ok


def _client(fake: FakeOllama) -> teacher.OllamaClient:
    return teacher.OllamaClient(transport=fake.transport())


def test_verify_returns_the_digest_and_checks_the_license() -> None:
    fake = FakeOllama(tagged_answer)
    info = _client(fake).verify("gemma4:e4b")
    assert info == teacher.TeacherInfo("gemma4:e4b", DIGEST, "Apache-2.0")
    assert [path for path, _ in fake.requests] == ["/api/show", "/api/tags"]


@pytest.mark.parametrize(
    ("fake", "model", "message"),
    [
        (FakeOllama(tagged_answer), "llama3.1:8b", "Llama Community License"),
        (FakeOllama(tagged_answer), "gemma4:12b", "pull it first: ollama pull gemma4:12b"),
        (
            FakeOllama(tagged_answer, license_text="Gemma Terms of Use"),
            "gemma4:e4b",
            "does not report the Apache License 2.0",
        ),
        (FakeOllama(tagged_answer, license_text=None), "gemma4:e4b", "Apache License 2.0"),
        (
            FakeOllama(tagged_answer, show_extra={"remote_host": "https://ollama.com:443"}),
            "gemma4:e4b",
            "runs on a remote host",
        ),
    ],
)
def test_verify_refusals(fake: FakeOllama, model: str, message: str) -> None:
    with pytest.raises(teacher.TeacherError, match=re.escape(message)):
        _client(fake).verify(model)
    # A refused name never reaches the server.
    if model.startswith("llama"):
        assert fake.requests == []


def test_verify_needs_a_listed_digest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"license": APACHE})
        return httpx.Response(200, json={"models": [{"name": "gemma4:e4b", "digest": ""}]})

    client = teacher.OllamaClient(transport=httpx.MockTransport(handler))
    with pytest.raises(teacher.TeacherError, match="lists no digest for gemma4:e4b"):
        client.verify("gemma4:e4b")


def test_chat_is_deterministic_and_errors_name_no_text() -> None:
    fake = FakeOllama(lambda body: "answer")
    client = _client(fake)
    assert client.chat("gemma4:e4b", "sys", "user", seed=7, json_format=True) == "answer"
    body = fake.chats[0]
    assert body["options"] == {"temperature": 0, "seed": 7}
    assert body["stream"] is False and body["format"] == "json"
    client.chat("gemma4:e4b", "sys", "user", seed=7, json_format=False)
    assert "format" not in fake.chats[1]
    failing = _client(FakeOllama(lambda body: "x", chat_status=500))
    with pytest.raises(teacher.TeacherError) as excinfo:
        failing.chat("gemma4:e4b", "sys", "user", seed=7, json_format=False)
    assert str(excinfo.value) == "the server answered HTTP 500 to /api/chat"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, text="not json"), "/api/chat: the answer is not JSON"),
        (httpx.Response(200, json={"message": {"content": 3}}), "unexpected answer shape"),
        (httpx.Response(200, json=[1]), "unexpected answer shape"),
        (httpx.Response(302, headers={"location": "http://evil.example/"}), "HTTP 302"),
    ],
)
def test_chat_rejects_odd_answers(response: httpx.Response, message: str) -> None:
    client = teacher.OllamaClient(transport=httpx.MockTransport(lambda request: response))
    with pytest.raises(teacher.TeacherError, match=re.escape(message)):
        client.chat("gemma4:e4b", "s", "u", seed=1, json_format=False)


def test_transport_failures_name_the_exception_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to 127.0.0.1:11434")

    client = teacher.OllamaClient(transport=httpx.MockTransport(handler))
    with pytest.raises(teacher.TeacherError) as excinfo:
        client.chat("gemma4:e4b", "s", "u", seed=1, json_format=False)
    assert str(excinfo.value) == "/api/chat: ConnectError"
    with pytest.raises(teacher.TeacherError, match="/api/show: unexpected answer shape"):
        teacher.OllamaClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))
        ).verify("gemma4:e4b")


def test_the_client_ignores_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    client = teacher.OllamaClient()
    assert client._client._mounts == {}  # noqa: SLF001 - no proxy mount was made
    assert client._client.follow_redirects is False  # noqa: SLF001
    client.close()


# --- grounding --------------------------------------------------------------------

ALL = prompts.TYPES


def _values(found: grounding.Grounded) -> list[tuple[str, str]]:
    return [(found.text[s.start : s.end], s.type) for s in found.spans]


def test_ground_removes_tags_and_keeps_exact_spans() -> None:
    answer = 'author: <pii type="PERSON">Ann Lee</pii> <<pii type="EMAIL">ann@example.com</pii>>'
    found = grounding.ground(answer, types=ALL, negative=False)
    assert isinstance(found, grounding.Grounded)
    assert found.text == "author: Ann Lee <ann@example.com>"
    assert _values(found) == [("Ann Lee", "PERSON"), ("ann@example.com", "EMAIL")]
    assert found.propagated == 0


def test_ground_makes_whole_token_repeats_gold() -> None:
    answer = 'Hi <pii type="PERSON">Ann</pii>,\nAnn wrote to Annabel; Ann_x and Ann.\n'
    found = grounding.ground(answer, types=ALL, negative=False)
    assert isinstance(found, grounding.Grounded)
    # "Ann" twice more as a whole token; not inside Annabel or Ann_x.
    last = found.text.rindex("Ann.")
    assert [(s.start, s.end) for s in found.spans] == [(3, 6), (8, 11), (last, last + 3)]
    assert found.propagated == 2


def test_ground_keeps_a_part_inside_a_longer_span_of_its_type() -> None:
    answer = '<pii type="PERSON">Ann Lee</pii> said <pii type="PERSON">Ann</pii> is fine'
    found = grounding.ground(answer, types=ALL, negative=None)
    assert isinstance(found, grounding.Grounded)
    assert _values(found) == [("Ann Lee", "PERSON"), ("Ann", "PERSON")]


@pytest.mark.parametrize(
    ("answer", "negative", "reason"),
    [
        ('<pii type="PERSON">Ann <pii type="PERSON">Lee</pii></pii>', False, grounding.MALFORMED),
        ('<pii type="PERSON">Ann', False, grounding.MALFORMED),
        ('Ann</pii> and <pii type="EMAIL">a@b.example</pii>', False, grounding.MALFORMED),
        ('<pii type="person">Ann</pii>', False, grounding.MALFORMED),
        ('<PII type="PERSON">Ann</PII>', False, grounding.MALFORMED),
        ('<pii type="SECRET">hunter2</pii>', False, grounding.UNKNOWN_TYPE),
        ('<pii type="PERSON"> Ann</pii>', False, grounding.BAD_VALUE),
        ('<pii type="PERSON"></pii> x', False, grounding.BAD_VALUE),
        ('<pii type="PERSON">Ann\nLee</pii>', False, grounding.BAD_VALUE),
        ('<pii type="PERSON">' + "a" * 201 + "</pii>", False, grounding.BAD_VALUE),
        ('<pii type="PERSON">Lee</pii> <pii type="USERNAME">Lee</pii>', False, grounding.TWO_TYPES),
        (
            '<pii type="USERNAME">ann</pii> <pii type="EMAIL">x@y.example</pii> ann@y',
            False,
            None,
        ),
        (
            '<pii type="PERSON">Lee</pii> <pii type="ADDRESS">4 Lee Road</pii>',
            False,
            grounding.CONFLICT,
        ),
        ('{"ok": true, "who": <pii type="PERSON">Ann</pii>}', True, grounding.NEGATIVE_TAGGED),
        ('{"ok": true}', False, grounding.NO_SPANS),
        ("   \n", None, grounding.EMPTY),
        ('<pii type="PERSON">Ann</pii> «PERSON_001»', False, grounding.PLACEHOLDER),
    ],
)
def test_ground_drops_what_cannot_be_grounded(
    answer: str, negative: bool | None, reason: str | None
) -> None:
    found = grounding.ground(answer, types=ALL, negative=negative)
    if reason is None:
        assert isinstance(found, grounding.Grounded)
    else:
        assert found == reason


def test_ground_allows_a_multiline_address_and_strips_one_fence() -> None:
    answer = '```json\n{"to": "<pii type="ADDRESS">12 Elm Way\nSpringfield</pii>"}\n```'
    found = grounding.ground(answer, types=ALL, negative=False)
    assert isinstance(found, grounding.Grounded)
    assert found.text == '{"to": "12 Elm Way\nSpringfield"}'
    assert grounding.strip_fence("```\nx\n```\nmore") == "```\nx\n```\nmore"


def test_ground_respects_the_asked_types() -> None:
    answer = '<pii type="PERSON">Ann</pii>'
    assert grounding.ground(answer, types=("EMAIL",), negative=False) == grounding.UNKNOWN_TYPE


def test_tagged_round_trips() -> None:
    answer = 'x <pii type="PERSON">Ann Lee</pii>, y <pii type="EMAIL">a@b.example</pii> z'
    found = grounding.ground(answer, types=ALL, negative=None)
    assert isinstance(found, grounding.Grounded)
    assert grounding.tagged(found.text, found.spans) == answer


# --- prompts ----------------------------------------------------------------------


def test_prompts_are_reproducible_and_varied() -> None:
    built = [prompts.build(7, i, 0.25) for i in range(200)]
    assert built == [prompts.build(7, i, 0.25) for i in range(200)]
    assert built != [prompts.build(8, i, 0.25) for i in range(200)]
    ids = {p.prompt_id for p in built}
    assert {a.id for a in prompts.ARTIFACTS} <= ids
    assert {f"{a.id}.negative" for a in prompts.ARTIFACTS} & ids
    negatives = [p for p in built if p.negative]
    assert 20 < len(negatives) < 80
    assert all(p.types == () and "NO personal data" in p.user for p in negatives)
    for p in built:
        if not p.negative:
            assert 2 <= len(p.types) <= 4 and set(p.types) <= set(prompts.TYPES)
    assert not any(prompts.build(1, i, 0.0).negative for i in range(50))
    assert all(prompts.build(1, i, 1.0).negative for i in range(50))


def test_the_system_prompt_names_every_type_and_the_catalog_is_digested() -> None:
    for name in prompts.TYPES:
        assert f"- {name}: " in prompts.SYSTEM
    assert re.fullmatch(r"[0-9a-f]{64}", prompts.catalog_sha256())


# --- private files ----------------------------------------------------------------


def test_default_data_dir(tmp_path: Path) -> None:
    assert (
        default_data_dir({"XDG_DATA_HOME": str(tmp_path)}) == tmp_path / "llm-redact" / "pii-corpus"
    )
    assert default_data_dir({"XDG_DATA_HOME": ""}) == (
        Path.home() / ".local" / "share" / "llm-redact" / "pii-corpus"
    )


def test_corpus_files_stay_outside_git_work_trees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    problem = output_problem(repo / "data" / "rows.jsonl")
    assert problem is not None and "must never be committed" in problem
    assert output_problem(tmp_path / "out" / "rows.jsonl") is None
    with pytest.raises(CorpusError, match="inside the git work tree"):
        open_private(repo / "rows.jsonl", overwrite=False)


def test_open_private_refuses_overwrites_and_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "new" / "rows.jsonl"
    with open_private(path, overwrite=False) as out:
        out.write("{}\n")
    with pytest.raises(CorpusError, match="exists; pass --force"):
        open_private(path, overwrite=False)
    with open_private(path, overwrite=True) as out:
        out.write("[]\n")
    assert path.read_text() == "[]\n"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        link = tmp_path / "link.jsonl"
        link.symlink_to(path)
        with pytest.raises(CorpusError, match="cannot write"):
            open_private(link, overwrite=True)


def test_read_jsonl_names_lines_not_content(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n[1]\n')
    rows = read_jsonl(path)
    assert next(rows) == (1, {"a": 1})
    with pytest.raises(CorpusError, match=r"line 3: not a JSON object"):
        next(rows)
    path.write_text("{secret value\n")
    with pytest.raises(CorpusError) as excinfo:
        list(read_jsonl(path))
    assert "secret" not in str(excinfo.value)
    with pytest.raises(CorpusError, match="cannot read"):
        list(read_jsonl(tmp_path / "missing.jsonl"))


# --- generate.py ------------------------------------------------------------------


def test_generate_writes_grounded_rows_and_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[int] = []

    def answer(body: dict[str, Any]) -> str:
        seen.append(len(seen))
        if len(seen) == 3:  # one ungroundable answer
            return '<pii type="PERSON">Ann'
        return tagged_answer(body)

    fake = FakeOllama(answer)
    out = tmp_path / "corpus" / "rows.jsonl"
    argv = ["--model", "gemma4:e4b", "--count", "12", "--seed", "5", "--out", str(out)]
    assert generate.main(argv, transport=fake.transport()) == 0
    rows = [row for _, row in read_jsonl(out)]
    assert len(rows) == 11
    for row in rows:
        assert set(row) == {"id", "text", "spans", "teacher", "prompt_id", "seed"}
        assert row["teacher"] == "gemma4:e4b" and row["seed"] == 5
        assert re.fullmatch(r"gemma4-e4b-5-\d{6}", row["id"])
        for span in row["spans"]:
            assert set(span) == {"start", "end", "type"}
            assert row["text"][span["start"] : span["end"]] == VALUES[span["type"]]
        assert "<pii" not in row["text"]
    assert len({row["id"] for row in rows}) == 11
    assert all(body["options"] == {"temperature": 0, "seed": 5} for body in fake.chats)
    assert len(fake.chats) == 12
    manifest = json.loads(generate.manifest_path(out).read_text())
    assert manifest["teacher"] == {"model": "gemma4:e4b", "digest": DIGEST, "license": "Apache-2.0"}
    assert manifest["server"] == "loopback" and manifest["complete"] is True
    assert manifest["catalog_sha256"] == prompts.catalog_sha256()
    assert manifest["counts"]["written"] == 11
    assert manifest["counts"][f"dropped: {grounding.MALFORMED}"] == 1
    assert manifest["verified"] is False
    assert "text" not in json.dumps(manifest).replace("rows_file", "")
    printed = capsys.readouterr()
    assert "written 11" in printed.out
    for value in VALUES.values():
        assert value not in printed.out + printed.err
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600
        assert generate.manifest_path(out).stat().st_mode & 0o777 == 0o600


def test_generate_default_output_and_teacher_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    calls: list[int] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            calls.append(1)
            if len(calls) > 2:
                raise httpx.ReadTimeout("timed out")
        return fake.handler(request)

    fake = FakeOllama(tagged_answer)
    argv = ["--model", "gemma4:e4b", "--count", "5", "--negatives", "0"]
    assert generate.main(argv, transport=httpx.MockTransport(flaky)) == 1
    out = tmp_path / "data" / "llm-redact" / "pii-corpus" / "generated-gemma4-e4b-42.jsonl"
    assert len(list(read_jsonl(out))) == 2
    manifest = json.loads(generate.manifest_path(out).read_text())
    assert manifest["complete"] is False
    assert "stopped after a teacher error: /api/chat: ReadTimeout" in capsys.readouterr().err
    # The same output again needs --force.
    assert generate.main(argv, transport=fake.transport()) == 2
    assert "exists; pass --force" in capsys.readouterr().err
    assert generate.main([*argv, "--force"], transport=fake.transport()) == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--url", "http://gpu.lan:11434"], "refusing the non-loopback server gpu.lan"),
        (["--count", "0"], "--count must be at least 1"),
        (["--negatives", "1.5"], "--negatives must be between 0 and 1"),
        (["--model", "llama3.1:8b"], "Llama Community License"),
        (["--model", "gemma4:26b"], "pull it first"),
        (["--out", "{repo}/rows.jsonl"], "must never be committed"),
    ],
)
def test_generate_input_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    fake = FakeOllama(tagged_answer)
    extra = [a.replace("{repo}", str(repo)) for a in extra]
    argv = ["--model", "gemma4:e4b", "--out", str(tmp_path / "rows.jsonl"), *extra]
    assert generate.main(argv, transport=fake.transport()) == 2
    assert message in capsys.readouterr().err
    assert fake.chats == []


def test_generate_runs_as_a_file() -> None:
    import subprocess

    script = Path(generate.__file__)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0 and "--allow-remote-server" in result.stdout
    assert generate.slug("gemma4:12b-it-q8_0") == "gemma4-12b-it-q8-0"
    assert generate.slug("mistral-small3.2:24b") == "mistral-small3.2-24b"


# --- audit.py ---------------------------------------------------------------------

_CORPUS_A = "Contact Ann Lee at ann@example.com\nbuild by Jenkins\nmail foo@bar.example\n"


def _corpus(root: Path) -> Path:
    corpus = root / "fp_corpus"
    corpus.mkdir()
    (corpus / "MANIFEST.toml").write_text('["a.txt"]\nEMAIL = 2\n')
    (corpus / "a.txt").write_text(_CORPUS_A)
    (corpus / "b.txt").write_text("nothing here\n")
    return corpus


def _audit_answer(body: dict[str, Any]) -> str:
    chunk = body["messages"][1]["content"]
    assert body["format"] == "json" and body["options"]["temperature"] == 0
    if "Ann Lee" not in chunk:
        return "I cannot help with that"
    entities = [
        {"type": "person", "value": "Ann Lee"},
        {"type": "EMAIL", "value": "ann@example.com"},
        {"type": "USERNAME", "value": "foo@bar.example"},
        {"type": "PERSON", "value": "Zed"},
        {"type": "COLOR", "value": "Jenkins"},
        {"type": "PERSON"},
        "Ann",
    ]
    return json.dumps({"entities": entities})


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


def test_audit_lists_disagreements_by_offset_type_and_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(tmp_path)
    before = _snapshot(corpus)
    fake = FakeOllama(_audit_answer)
    out = tmp_path / "report.json"
    argv = ["--model", "gemma4:e4b", "--corpus", str(corpus), "--out", str(out)]
    assert audit.main(argv, transport=fake.transport()) == 0
    assert _snapshot(corpus) == before  # report only: nothing in the corpus changes
    report = json.loads(out.read_text())
    ann = _CORPUS_A.index("Ann Lee")
    foo = _CORPUS_A.index("foo@bar.example")
    assert report["findings"] == [
        {"file": "a.txt", "start": ann, "end": ann + 7, "type": "PERSON", "reason": "teacher-only"},
        {
            "file": "a.txt",
            "start": foo,
            "end": foo + 15,
            "type": "USERNAME",
            "reason": "type-differs",
        },
    ]
    assert report["counts"] == {
        "chunks": 2,
        "files": 2,
        "teacher answers unusable": 1,
        "teacher entities malformed": 2,
        "teacher types unknown": 1,
        "teacher values ungrounded": 1,
        "teacher-only": 1,
        "type-differs": 1,
    }
    assert report["teacher"] == {"model": "gemma4:e4b", "digest": DIGEST}
    printed = capsys.readouterr()
    assert f"| a.txt | {ann} | {ann + 7} | PERSON | teacher-only |" in printed.out
    for value in ("Ann Lee", "ann@example.com", "foo@bar.example", "Jenkins", "nothing here"):
        assert value not in printed.out + printed.err + out.read_text()


def test_audit_reports_detector_only_spans_and_audits_chosen_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(tmp_path)
    fake = FakeOllama(lambda body: '{"entities": []}')
    argv = ["--model", "gemma4:e4b", "--corpus", str(corpus), "--file", "a.txt"]
    assert audit.main(argv, transport=fake.transport()) == 0
    printed = capsys.readouterr().out
    rows = [line for line in printed.splitlines() if line.startswith("| a.txt")]
    assert len(rows) == 2 and all(row.endswith("| EMAIL | detector-only |") for row in rows)
    assert "files: 1." in printed and "default configuration" in printed
    assert len(fake.chats) == 1


def test_audit_compare_and_teacher_spans() -> None:
    counts: Counter[str] = Counter()
    chunk = "Lee met Leeds; Lee"
    assert audit.teacher_spans(
        '{"entities": [{"type": "PERSON", "value": "Lee"}]}', chunk, counts
    ) == [
        (0, 3, "PERSON"),
        (15, 18, "PERSON"),
    ]
    assert audit.teacher_spans("[]", chunk, counts) is None
    assert audit.teacher_spans('{"entities": {}}', chunk, counts) is None
    found = audit.compare("f", 100, [(0, 3, "PERSON")], [(0, 3, "PERSON"), (8, 13, "ADDRESS")])
    assert found == [audit.Finding("f", 108, 113, "ADDRESS", "detector-only")]


@pytest.mark.parametrize(
    ("extra", "message", "status"),
    [
        (["--url", "http://box.lan:11434"], "non-loopback server box.lan", 2),
        (["--corpus", "{tmp}/missing"], "is not a directory", 2),
        (["--out", "{corpus}/report.json"], "--out must not be inside the corpus", 2),
        (["--model", "qwen3:8b"], "Qwen family", 2),
        (["--config", "{tmp}/missing.toml"], "missing.toml", 2),
    ],
)
def test_audit_input_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
    status: int,
) -> None:
    corpus = _corpus(tmp_path)
    extra = [a.replace("{tmp}", str(tmp_path)).replace("{corpus}", str(corpus)) for a in extra]
    fake = FakeOllama(_audit_answer)
    argv = ["--model", "gemma4:e4b", "--corpus", str(corpus), *extra]
    assert audit.main(argv, transport=fake.transport()) == status
    assert message in capsys.readouterr().err
    assert fake.chats == []


def test_audit_with_a_config_and_a_failing_teacher(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(tmp_path)
    config = tmp_path / "rules-only.toml"
    config.write_text("[detection]\n")
    fake = FakeOllama(_audit_answer, chat_status=503)
    argv = ["--model", "gemma4:e4b", "--corpus", str(corpus), "--config", str(config)]
    assert audit.main(argv, transport=fake.transport()) == 1
    assert (
        "the teacher failed: the server answered HTTP 503 to /api/chat" in capsys.readouterr().err
    )
