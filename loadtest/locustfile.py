"""Locust glue (ADR 0013): wires the tested helpers in hpdocs_load to Locust.

It holds no decision of its own, which is why it is the one file outside the coverage
gate (a test pins that). Run it through `make load`; settings come from LOAD_* variables
(hpdocs_load.config). Message text is never logged: requests are recorded as times,
statuses and error codes only.
"""

import os
import pathlib
import random
import time
from collections.abc import Iterator

import gevent
import psutil
import requests
from locust import HttpUser, LoadTestShape, constant, events, task
from locust.env import Environment

from hpdocs_load.budget import PROFILE_B, SpendGuard, admission
from hpdocs_load.burst import BURST_USERS, burst_questions, shape_users
from hpdocs_load.client_id import new_client_id
from hpdocs_load.config import config_from_env
from hpdocs_load.ramp import measured_windows, tick
from hpdocs_load.run_files import RunFiles
from hpdocs_load.samplers import HostProbes, run_command, take_samples
from hpdocs_load.session import Reply, Session, StreamReply
from hpdocs_load.stages import logs_command, parse_stage_logs
from hpdocs_load.workload import (
    CREATE,
    GOLDEN,
    Action,
    load_question_pool,
    page_open,
    plan_conversation,
)

CONFIG = config_from_env(os.environ)
POOL = load_question_pool(GOLDEN)
# S311: load-test randomness must be reproducible from a seed; nothing here is a secret.
SEEDS = random.Random(CONFIG.seed)  # noqa: S311
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
HOST = HostProbes(psutil.virtual_memory, psutil.swap_memory, os.sysconf("SC_PAGE_SIZE"))
STATE: dict[str, object] = {}
# One ID for every burst user, from its own generator, so the normal users draw the same
# seeds as in a capacity run.
BURST_CLIENT_ID = new_client_id(random.Random(f"burst-{CONFIG.seed}"))  # noqa: S311 - see SEEDS


def _quit_run() -> None:
    # Spawned: the refused user's own greenlet must not be the one that kills all users.
    gevent.spawn(STATE["environment"].runner.quit)  # type: ignore[attr-defined]


# Profile B (LOAD_BUDGET=1) stops at the $3.00 cap or 90 questions (hpdocs_load.budget).
GUARD = SpendGuard(PROFILE_B, _quit_run)
ADMIT = admission(enabled=CONFIG.budget, guard=GUARD)


class Clock:
    def wall(self) -> float:
        return time.time()

    def mono(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        gevent.sleep(seconds)


def _timed_lines(response: requests.Response) -> Iterator[tuple[float, str]]:
    try:
        for line in response.iter_lines(decode_unicode=True):
            yield time.monotonic(), str(line)
    finally:
        response.close()


class LocustTransport:
    def __init__(self, user: HttpUser, client_id: str) -> None:
        self._client = user.client
        self._headers = {"X-Client-ID": client_id}

    def call(self, method: str, path: str, *, name: str) -> Reply:
        response = self._client.request(method, path, name=name, headers=self._headers)
        return Reply(response.status_code, response.text)

    def stream(self, path: str, *, content: str) -> StreamReply:
        response = self._client.post(
            path, json={"content": content}, name="message", headers=self._headers, stream=True
        )
        response.encoding = "utf-8"
        return StreamReply(response.status_code, _timed_lines(response))


class Browser(HttpUser):
    """One browser tab: its own client ID and a reproducible plan stream."""

    wait_time = constant(0)  # think time is part of each plan

    def on_start(self) -> None:
        self.rng = random.Random(SEEDS.getrandbits(64))  # noqa: S311 - reproducible, see SEEDS
        files: RunFiles = STATE["files"]  # type: ignore[assignment]
        transport = LocustTransport(self, new_client_id(self.rng))
        self.session = Session(transport, Clock(), files.add_request, admit=ADMIT)
        self.session.run(page_open())

    @task
    def conversation(self) -> None:
        self.session.run(plan_conversation(self.rng, POOL))


class Burst(HttpUser):
    """The rate-limits run's burst client (hpdocs_load.burst): only with LOAD_BURST=1."""

    abstract = not CONFIG.burst
    fixed_count = BURST_USERS
    wait_time = constant(0)

    def on_start(self) -> None:
        files: RunFiles = STATE["files"]  # type: ignore[assignment]
        transport = LocustTransport(self, BURST_CLIENT_ID)
        self.session = Session(transport, Clock(), files.add_burst_request, admit=ADMIT)
        self.questions = burst_questions(POOL)
        self.session.run((Action(CREATE),))

    @task
    def ask(self) -> None:
        self.session.run((next(self.questions),))


class StepShape(LoadTestShape):
    def tick(self) -> tuple[int, float] | None:
        now = self.get_run_time()  # type: ignore[no-untyped-call]  # Locust
        return shape_users(tick(CONFIG.plan, now), burst=CONFIG.burst)


def _sample_forever(files: RunFiles) -> None:
    process = psutil.Process()
    process.cpu_percent(None)  # primes the counter; the first reading covers one interval
    while True:
        gevent.sleep(CONFIG.sample_seconds)
        files.add_samples(
            take_samples(
                at=time.time(),
                docker=run_command,
                host=HOST,
                generator_cpu=lambda: process.cpu_percent(None),
            )
        )


def _meta(environment: Environment, started_at: float) -> dict[str, object]:
    return {
        "run": CONFIG.name,
        "burst": CONFIG.burst,
        "budget": CONFIG.budget,
        "host": environment.host,
        "seed": CONFIG.seed,
        "sample_seconds": CONFIG.sample_seconds,
        "steps": list(CONFIG.plan.users),
        "step_seconds": CONFIG.plan.step_seconds,
        "warmup_seconds": CONFIG.plan.warmup_seconds,
        "spawn_rate": CONFIG.plan.spawn_rate,
        "started_at": started_at,
        "windows": [vars(w) for w in measured_windows(CONFIG.plan, run_start=started_at)],
    }


def _start(environment: Environment, **_: object) -> None:
    files = RunFiles.create(RESULTS, CONFIG.name)
    STATE["files"] = files
    STATE["environment"] = environment
    STATE["meta"] = _meta(environment, time.time())
    files.write_meta(STATE["meta"])  # type: ignore[arg-type]
    STATE["sampler"] = gevent.spawn(_sample_forever, files)


def _stop(**_: object) -> None:
    files: RunFiles = STATE["files"]  # type: ignore[assignment]
    STATE["sampler"].kill()  # type: ignore[attr-defined]
    started: float = STATE["meta"]["started_at"]  # type: ignore[index]
    files.add_stages(parse_stage_logs(run_command(logs_command(started))))
    meta = {**STATE["meta"], "ended_at": time.time(), "spend_guard": GUARD.summary()}  # type: ignore[dict-item]
    files.write_meta(meta)
    files.close()


# Locust's event hooks are untyped.
events.test_start.add_listener(_start)  # type: ignore[no-untyped-call]
events.test_stop.add_listener(_stop)  # type: ignore[no-untyped-call]
