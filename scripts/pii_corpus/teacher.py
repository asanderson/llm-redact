"""The teacher: a local LLM served by Ollama, and the policy for which ones
may be asked.

POLICY (plan D8/T40; README.md "Licensing"):

* only models whose weights are published under Apache-2.0, by exact
  Ollama library name and size tag (:data:`ALLOWED`); a name or tag that can
  move (no tag, ``latest``) is refused, and so is every ``cloud`` tag (those
  run on Ollama's servers, not this machine);
* Llama, Qwen, DeepSeek, Gemma 1 to 3n and the Mistral research and
  non-production licensed models are refused BY NAME (:data:`DENIED`), with
  the reason, whatever else is configured;
* at run time the server must report the Apache License 2.0 text for the
  model (``/api/show``'s ``license``, which Ollama copies from the model's
  registry manifest), the model must already be pulled (this tooling never
  pulls), and its digest is recorded in the run manifest;
* the server is loopback by default; a non-loopback server needs an
  explicit flag AND https (the teacher receives the corpus text).

Every error names a model, a tag, a host, an HTTP status or an exception
type; never a prompt, an answer or a server error body.
"""

import ipaddress
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://127.0.0.1:11434"
# Generation on a laptop CPU can take minutes per answer.
DEFAULT_TIMEOUT_SECONDS = 600.0
# The date the allowlist and the denylist were checked against the Ollama
# registry (each tag's license layer) and the models' Hugging Face cards.
CHECKED = "2026-10-05"


class TeacherError(Exception):
    """The teacher cannot be used or did not answer. The message names a
    model, a host, a status or an exception type, never text."""


@dataclass(frozen=True)
class Family:
    """An allowed Ollama library model: the size tags whose license layer
    is the Apache License 2.0 (a tag may carry a variant suffix such as
    ``-it-q8_0``), and where the license was read."""

    sizes: tuple[str, ...]
    source: str


# Checked 2026-10-05: the license layer of each tag's registry manifest
# (registry.ollama.ai/v2/library/NAME/manifests/TAG) is the Apache License
# 2.0 text, and the Hugging Face card of the published weights says
# `license: apache-2.0`. mistral-small3.1:24b is left out on purpose: its
# manifest carries no license layer, so the run-time check could not pass.
ALLOWED: Mapping[str, Family] = {
    "gemma4": Family(("e2b", "e4b", "12b", "26b", "31b"), "google/gemma-4-* (Gemma 4)"),
    "mistral": Family(("7b",), "mistralai/Mistral-7B-Instruct-v0.3"),
    "mistral-nemo": Family(("12b",), "mistralai/Mistral-Nemo-Instruct-2407"),
    "mistral-small": Family(("24b",), "mistralai/Mistral-Small-24B-Instruct-2501"),
    "mistral-small3.2": Family(("24b",), "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    "devstral": Family(("24b",), "mistralai/Devstral-Small-2505"),
    "magistral": Family(("24b",), "mistralai/Magistral-Small-2506"),
    "ministral-3": Family(("3b", "8b", "14b"), "mistralai/Ministral-3 (December 2025)"),
    "mixtral": Family(("8x7b", "8x22b"), "mistralai/Mixtral-8x7B and 8x22B"),
}

_LLAMA = (
    "Llama family: the Llama Community License attaches naming and use terms to models"
    " trained on its output"
)
_GEMMA = "Gemma 1 to 3n: the Gemma Terms of Use attach use restrictions to derived models"
_MISTRAL_RESEARCH = "Mistral AI Research License: no commercial use of the model or its output"

# (pattern on the lowercased full name, reason). Checked before the
# allowlist, so a refused family is named even under an allowed spelling.
DENIED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"llama"), _LLAMA),
    (
        re.compile(r"qwen"),
        "Qwen family: refused by name (lineage and procurement policy, plan T40)",
    ),
    (re.compile(r"deepseek"), "DeepSeek family: refused by name (lineage and procurement policy)"),
    (re.compile(r"(?<![a-z])(code|shield|embedding)?gemma(?:[123]n?)?(?![0-9])"), _GEMMA),
    (re.compile(r"(?:^|/)codestral"), "Codestral: Mistral AI Non-Production License"),
    (re.compile(r"(?:^|/)mistral-large"), _MISTRAL_RESEARCH),
    (re.compile(r"(?:^|/)pixtral-large"), _MISTRAL_RESEARCH),
    (re.compile(r"(?:^|/)ministral(?!-3)"), _MISTRAL_RESEARCH),
    (re.compile(r"(?:^|/)mistral-small:22b"), _MISTRAL_RESEARCH),
)


def model_refusal(model: str) -> str | None:
    """Why ``model`` (an Ollama ``NAME:TAG``) may not be the teacher; None
    when the allowlist admits it (the run-time license check still runs)."""
    name, sep, tag = model.strip().lower().partition(":")
    for pattern, reason in DENIED:
        if pattern.search(model.strip().lower()):
            return f"{model}: {reason}"
    if not sep or not tag or tag == "latest":
        return (
            f"{model}: name a fixed size tag (NAME:TAG, e.g. gemma4:e4b); an untagged or"
            " 'latest' name can move to another model"
        )
    if "cloud" in tag or name.endswith("-cloud"):
        return f"{model}: a cloud tag runs on Ollama's servers, not on this machine"
    family = ALLOWED.get(name)
    if family is None:
        return (
            f"{model}: not on the teacher allowlist (Apache-2.0 models only; see"
            " scripts/pii_corpus/README.md)"
        )
    if not any(tag == size or tag.startswith(size + "-") for size in family.sizes):
        return f"{model}: tag not on the allowlist for {name} (allowed: {', '.join(family.sizes)})"
    return None


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a name that is not provably loopback is remote
        return False


def server_problem(url: str, *, allow_remote: bool) -> str | None:
    """Why the teacher may not be reached at ``url`` (None = it may):
    http(s) with a host and nothing else; loopback, or with ``allow_remote``
    a non-loopback https server."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "the server URL must look like http://127.0.0.1:11434"
    if parts.username or parts.password or parts.query or parts.fragment:
        return "the server URL must not carry credentials, a query or a fragment"
    if parts.path not in ("", "/"):
        return "the server URL must be the server's root (no path)"
    if _loopback(parts.hostname):
        return None
    if not allow_remote:
        return (
            f"refusing the non-loopback server {parts.hostname}: the teacher receives the"
            " corpus text; pass --allow-remote-server to use one over https"
        )
    if parts.scheme != "https":
        return f"a non-loopback server ({parts.hostname}) must be reached over https"
    return None


def apache_license(text: object) -> bool:
    """Whether a reported license is the Apache License 2.0 text."""
    if isinstance(text, list):
        text = "\n".join(str(part) for part in text)
    if not isinstance(text, str):
        return False
    words = " ".join(text.split())
    return words.startswith("Apache License") and "Version 2.0" in words[:200]


@dataclass(frozen=True)
class TeacherInfo:
    model: str
    digest: str
    license: str = "Apache-2.0"


class OllamaClient:
    """Ollama's HTTP API (``/api/chat``, ``/api/show``, ``/api/tags``).
    Environment proxies are ignored and redirects are not followed, so a
    loopback URL stays loopback."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise TeacherError(f"{path}: {type(exc).__name__}") from exc
        if response.status_code == 404 and path == "/api/show":
            return None
        if response.status_code != 200:
            raise TeacherError(f"the server answered HTTP {response.status_code} to {path}")
        try:
            return response.json()
        except ValueError as exc:
            raise TeacherError(f"{path}: the answer is not JSON") from exc

    def verify(self, model: str) -> TeacherInfo:
        """The policy's run-time half: the model is pulled, local, and
        reported under the Apache License 2.0."""
        refusal = model_refusal(model)
        if refusal is not None:
            raise TeacherError(refusal)
        shown = self._call("POST", "/api/show", {"model": model, "name": model})
        if shown is None:
            raise TeacherError(f"{model} is not on the server; pull it first: ollama pull {model}")
        if not isinstance(shown, dict):
            raise TeacherError("/api/show: unexpected answer shape")
        if shown.get("remote_host") or shown.get("remote_model"):
            raise TeacherError(f"{model} runs on a remote host, not on the server's machine")
        if not apache_license(shown.get("license")):
            raise TeacherError(f"the server does not report the Apache License 2.0 for {model}")
        listed = self._call("GET", "/api/tags")
        models = listed.get("models") if isinstance(listed, dict) else None
        for entry in models if isinstance(models, list) else []:
            if isinstance(entry, dict) and model in (entry.get("name"), entry.get("model")):
                digest = entry.get("digest")
                if isinstance(digest, str) and digest:
                    return TeacherInfo(model=model, digest=digest)
        raise TeacherError(f"/api/tags lists no digest for {model}")

    def chat(self, model: str, system: str, user: str, *, seed: int, json_format: bool) -> str:
        """One deterministic answer (temperature 0, fixed seed)."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0, "seed": seed},
        }
        if json_format:
            body["format"] = "json"
        answer = self._call("POST", "/api/chat", body)
        message = answer.get("message") if isinstance(answer, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise TeacherError("/api/chat: unexpected answer shape")
        return content
