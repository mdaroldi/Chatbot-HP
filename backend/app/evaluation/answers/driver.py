"""The host-side half of an answer run (`drive` command): the harness container cannot
run docker, so the host restarts the backend for the run and puts it back after.

1. Restart the backend with `overrides(run)` over the caller's environment (compose
   interpolates `${VAR:-default}`, and the shell wins over `.env`), waiting until
   healthy.
2. `printenv` the checked names in the backend container (never the key) and compare
   them with `backend_env`: the run's settings, the real provider, and the matrix's
   models and token bounds, on which the budget's hard stop rests.
3. Run `ask` in a one-off backend container on the compose networks, as the caller's
   user with `eval/` mounted. It gets the stub provider and an empty key: the harness
   never calls a model, so it never holds the key.
4. Always restart the backend with the caller's environment alone. A failed restore
   after a finished run is an error; it never replaces the error that stopped a run.

Commands are fixed argument lists run without a shell.
"""

import logging
import subprocess  # nosec B404 - fixed docker compose commands, no shell
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.settings import LLMProviderName
from app.domain.ingestion import ChunkerKind
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec
from app.evaluation.answers.config_check import (
    CHECKED_NAMES,
    DEFAULT_ALIAS,
    backend_env,
    check_printenv,
    index_chunker,
    overrides,
    run_settings,
)

logger = logging.getLogger(__name__)

_COMPOSE = ("docker", "compose")
UP = (*_COMPOSE, "up", "--detach", "--no-deps", "--no-build", "--wait", "backend")
PRINTENV = (*_COMPOSE, "exec", "-T", "backend", "printenv", *CHECKED_NAMES)


class DriverError(Exception):
    """A docker step failed; the message names the step, never its output."""


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str  # empty unless captured


class CommandRunner(Protocol):
    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], capture: bool
    ) -> Completed: ...


@dataclass(frozen=True)
class DriveRequest:
    run: RunSpec
    eval_dir: Path  # absolute, on the host
    ask_args: tuple[str, ...]  # passed through to `ask` (--limit, --resume, ...)
    uid: int
    gid: int
    provider: LLMProviderName = LLMProviderName.ANTHROPIC


def subprocess_runner(cwd: Path) -> CommandRunner:
    """Commands run in the repository root, where compose finds its file and `.env`."""

    def run(argv: Sequence[str], *, env: Mapping[str, str], capture: bool) -> Completed:
        done = subprocess.run(  # noqa: S603 - fixed command, no shell  # nosec B603
            list(argv), cwd=cwd, env=dict(env), capture_output=capture, text=True, check=False
        )
        return Completed(done.returncode, done.stdout if capture else "")

    return run


def _ask_argv(request: DriveRequest) -> tuple[str, ...]:
    return (
        *_COMPOSE,
        "run",
        "--rm",
        "--no-deps",
        "--user",
        f"{request.uid}:{request.gid}",
        "--volume",
        f"{request.eval_dir}:/eval",
        "--env",
        "LLM_PROVIDER=stub",
        "--env",
        "ANTHROPIC_API_KEY=",
        "--entrypoint",
        "python",
        "backend",
        "-m",
        "app.evaluation.answers",
        "ask",
        "--run",
        request.run.name,
        *request.ask_args,
        *(() if request.provider is LLMProviderName.ANTHROPIC else ("--provider", "stub")),
    )


def _pins(matrix: AnswerMatrix, request: DriveRequest) -> dict[str, str]:
    """A real run pins the run settings and the provider; the models and bounds must
    already be the backend's. A stub check also pins the stub matrix's models and
    bounds, so it runs on any `.env`.
    """
    if request.provider is LLMProviderName.ANTHROPIC:
        return overrides(request.run)
    return backend_env(matrix, request.run, request.provider)


def _step(runner: CommandRunner, name: str, argv: Sequence[str], env: Mapping[str, str]) -> str:
    done = runner(argv, env=env, capture=True)
    if done.returncode != 0:
        raise DriverError(f"{name} failed with exit code {done.returncode}")
    return done.stdout


def drive(
    matrix: AnswerMatrix,
    request: DriveRequest,
    runner: CommandRunner,
    environ: Mapping[str, str],
) -> int:
    """`ask`'s exit code, once the backend is back on the caller's settings."""
    run_env = dict(environ) | _pins(matrix, request)
    try:
        _step(runner, "backend restart (up)", UP, run_env)
        logger.info("backend restarted for the run", extra={"run": request.run.name})
        check_printenv(
            backend_env(matrix, request.run, request.provider),
            _step(runner, "printenv", PRINTENV, run_env),
        )
        logger.info("backend env checked", extra={"run": request.run.name})
        code = runner(_ask_argv(request), env=run_env, capture=False).returncode
    finally:
        restored = runner(UP, env=dict(environ), capture=True).returncode == 0
        if not restored:
            logger.error("backend not restored to its defaults; run make up")
    if not restored:
        raise DriverError("backend restore failed after the run")
    logger.info("backend restored to its defaults", extra={"run": request.run.name})
    return code


def ingest_argv(alias: str, chunker: ChunkerKind) -> tuple[str, ...]:
    return (
        *_COMPOSE,
        "run",
        "--rm",
        "--no-deps",
        "--env",
        f"CHUNKER={chunker.value}",
        "--env",
        f"QDRANT_COLLECTION_ALIAS={alias}",
        "ingest",
        "--if-needed",
    )


def build_index(run: RunSpec, runner: CommandRunner, environ: Mapping[str, str]) -> None:
    """A run on another alias than the default gets that alias's index, built only when
    it is missing or stale (`make up` keeps the default one current). The live alias is
    never touched.
    """
    alias = run_settings(run).alias
    if alias == DEFAULT_ALIAS:
        return
    chunker = index_chunker(alias)
    done = runner(ingest_argv(alias, chunker), env=environ, capture=False)
    if done.returncode != 0:
        raise DriverError(f"index build (ingest) failed with exit code {done.returncode}")
    logger.info("index ready for the run", extra={"run": run.name, "alias": alias})
