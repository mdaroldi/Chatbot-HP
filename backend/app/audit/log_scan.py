"""Canaries and the scan behind the stack log audit (`make test-log-audit`).

A canary token is lowercase letters and digits only, so URL encoding, form encoding and
JSON escaping leave it unchanged, and a case-insensitive search finds it in any of
those forms. The summary names containers, canary ids and line counts and never a log
line, because CI logs are public and a leaked line may hold more than the canary.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_ID = re.compile(r"[a-z0-9]+")
_NONCE = re.compile(r"[0-9a-f]{6,}")


class EmptyLogError(Exception):
    """A container gave no log output: the scan could not see it, so it cannot pass."""


@dataclass(frozen=True)
class Canary:
    id: str
    token: str


@dataclass(frozen=True)
class Leak:
    container: str
    canary_id: str
    lines: int


def new_canary(canary_id: str, nonce: str) -> Canary:
    """The nonce makes each run's tokens unique, so an old run's lines never match."""
    if not _ID.fullmatch(canary_id):
        raise ValueError("canary id must be lowercase letters and digits")
    if not _NONCE.fullmatch(nonce):
        raise ValueError("nonce must be at least six lowercase hex digits")
    return Canary(id=canary_id, token=f"zqcanary{nonce}x{canary_id}")


def find_leaks(logs: Mapping[str, str], canaries: Sequence[Canary]) -> list[Leak]:
    """Every (container, canary) pair whose token appears, with its line count."""
    if not logs:
        raise ValueError("no container logs to scan")
    if not canaries:
        raise ValueError("no canaries to look for")
    empty = sorted(name for name, text in logs.items() if not text.strip())
    if empty:
        raise EmptyLogError(f"no log output from: {', '.join(empty)}")
    leaks = []
    for container in sorted(logs):
        lines = logs[container].lower().splitlines()
        for canary in canaries:
            count = sum(canary.token in line for line in lines)
            if count:
                leaks.append(Leak(container=container, canary_id=canary.id, lines=count))
    return leaks


def summary(leaks: Sequence[Leak]) -> str:
    if not leaks:
        return "no leaks"
    parts = ", ".join(f"{leak.container} {leak.canary_id} lines={leak.lines}" for leak in leaks)
    return f"{len(leaks)} leak(s): {parts}"
