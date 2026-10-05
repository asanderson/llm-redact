"""Deployment helpers for the billing service.

Talks to the Jenkins controller that runs the deploy jobs, the Kafka
topics the ledger publishes to, and the Celery workers that render
invoices. Every name here is a tool, a class or a variable; nothing
refers to a person.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("billing.deploy")

DEFAULT_TIMEOUT_S = 900
POLL_INTERVAL_S = 10
JENKINS_URL_ENV = "JENKINS_URL"
JENKINS_TOKEN_ENV = "JENKINS_API_TOKEN"


class BuildResult(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ABORTED = "ABORTED"
    UNSTABLE = "UNSTABLE"


@dataclass(frozen=True)
class JobRef:
    """A Jenkins job and one of its build numbers."""

    name: str
    number: int

    def path(self) -> str:
        return f"/job/{self.name}/{self.number}/api/json"


@dataclass
class JenkinsClient:
    """Minimal client for triggering and polling deploy jobs.

    The token is read from the environment at construction time and never
    logged; only job names and build numbers are.
    """

    base_url: str
    token: str = field(repr=False)
    timeout_s: int = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> JenkinsClient:
        return cls(base_url=environ[JENKINS_URL_ENV], token=environ[JENKINS_TOKEN_ENV])

    def trigger(self, job: str, params: Mapping[str, str]) -> JobRef:
        logger.info("triggering %s with %d parameters", job, len(params))
        number = self._post(f"/job/{job}/buildWithParameters", params)
        return JobRef(job, number)

    def wait(self, ref: JobRef) -> BuildResult:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            state = self._get(ref.path())
            if state.get("result"):
                return BuildResult(state["result"])
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"{ref.name} #{ref.number} did not finish in {self.timeout_s}s")

    def _post(self, path: str, params: Mapping[str, str]) -> int:
        raise NotImplementedError("wired to httpx in production")

    def _get(self, path: str) -> dict[str, str]:
        raise NotImplementedError("wired to httpx in production")


class KafkaTopicNamer:
    """Topic names follow <domain>.<entity>.<version>, e.g. billing.ledger.v2."""

    def __init__(self, domain: str, version: int = 2) -> None:
        self.domain = domain
        self.version = version

    def topic(self, entity: str) -> str:
        return f"{self.domain}.{entity}.v{self.version}"

    def all_topics(self, entities: list[str]) -> Iterator[str]:
        for entity in entities:
            yield self.topic(entity)


class JacksonCompatSerializer:
    """Writes invoices in the field order the legacy Jackson-based exporter
    produced, so downstream CSV diffing keeps working."""

    FIELD_ORDER = ("invoiceId", "accountId", "currency", "amountCents", "postedAt")

    def serialize(self, invoice: Mapping[str, object]) -> list[object]:
        return [invoice.get(name) for name in self.FIELD_ORDER]


def deploy(environment: str, revision: str, client: JenkinsClient | None = None) -> BuildResult:
    """Trigger billing-deploy for ``environment`` at ``revision`` and wait."""
    client = client or JenkinsClient.from_env()
    ref = client.trigger("billing-deploy", {"ENV": environment, "REVISION": revision})
    result = client.wait(ref)
    if result is not BuildResult.SUCCESS:
        logger.error("deploy of %s to %s ended %s", revision[:7], environment, result.value)
    return result


if __name__ == "__main__":
    namer = KafkaTopicNamer("billing")
    for name in namer.all_topics(["ledger", "invoice", "refund"]):
        print(name)
