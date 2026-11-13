"""Model store baked into the backend image (ADR 0008).

The image build calls `main(["--cache", C, "--target", T])` (see the Dockerfile): every
model in `models.toml` is fetched at its pinned revision (a verified cache needs no
network), copied into T as real files in the hub cache layout, and T is then checked
to hold exactly the pinned files: none changed, none missing, nothing extra. Any
problem fails the build. At runtime the containers only read T, offline, through the
same `PinnedModelSource`, so no container ever downloads a model.
"""

import argparse
import asyncio
import hashlib
import logging
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from app.domain.ingestion import ModelDownloadError, ModelError, ModelFiles, ModelStoreError
from app.infra.model_files import (
    HubClient,
    HuggingFaceHub,
    PinnedModelSource,
    load_pins,
    snapshot_dir,
)

ATTEMPTS = 3
BACKOFF_SECONDS = 10.0

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


def check_snapshot(pin: ModelFiles, found: Mapping[str, str]) -> list[str]:
    """Compare one model's files on disk with its pins (pure)."""
    problems: list[str] = []
    for path in sorted(pin.files):
        if path not in found:
            problems.append(f"{pin.name}: {path} is missing")
        elif found[path] != pin.files[path]:
            problems.append(
                f"{pin.name}: {path} has SHA-256 {found[path]}, pinned {pin.files[path]}"
            )
    problems += [
        f"{pin.name}: {path} is not pinned" for path in sorted(set(found) - set(pin.files))
    ]
    return problems


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_files(folder: Path) -> dict[str, str]:
    """SHA-256 of every file below the folder, by path relative to it."""
    return {
        path.relative_to(folder).as_posix(): _sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def verify_store(pins: Mapping[str, ModelFiles], root: Path) -> list[str]:
    """Problems with the store at `root`: each pinned revision holds exactly its pinned
    files, and no other model or revision is present.
    """
    problems: list[str] = []
    expected = {snapshot_dir(root, pin.source, pin.revision): pin for pin in pins.values()}
    for folder, pin in expected.items():
        if not folder.is_dir():
            problems.append(f"{pin.name}: {pin.source}@{pin.revision} is not in the store")
            continue
        problems += check_snapshot(pin, snapshot_files(folder))
    present = sorted(path for path in root.glob("models--*/snapshots/*") if path.is_dir())
    problems += [
        f"{path.relative_to(root).as_posix()} is not pinned"
        for path in present
        if path not in expected
    ]
    return problems


async def _fetch_with_retries(source: PinnedModelSource, name: str, sleep: Sleep) -> Path:
    """Retry an unreachable hub only: a file that differs from its pin stays different."""
    delay = BACKOFF_SECONDS
    for _ in range(ATTEMPTS - 1):
        try:
            return (await source.fetch(name)).path
        except ModelDownloadError:
            logger.warning("model download failed; retrying", extra={"model": name})
            await sleep(delay)
            delay *= 2
    return (await source.fetch(name)).path


def _copy(pin: ModelFiles, folder: Path, target: Path) -> None:
    destination = snapshot_dir(target, pin.source, pin.revision)
    for name in sorted(pin.files):
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        # copyfile follows the cache's symlinks, so the store holds real files.
        shutil.copyfile(folder / name, destination / name)


async def bake(
    pins: Mapping[str, ModelFiles],
    cache_dir: Path,
    target: Path,
    hub: HubClient,
    *,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Fill `target` with every pinned model and verify it, or raise a ModelError."""
    source = PinnedModelSource(pins, cache_dir, hub)
    for name, pin in pins.items():
        folder = await _fetch_with_retries(source, name, sleep)
        await asyncio.to_thread(_copy, pin, folder, target)
    problems = await asyncio.to_thread(verify_store, pins, target)
    if problems:
        raise ModelStoreError(problems)


def main(
    argv: Sequence[str],
    *,
    pins: Mapping[str, ModelFiles] | None = None,
    hub: HubClient | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.infra.model_store")
    parser.add_argument("--cache", type=Path, required=True, help="download cache")
    parser.add_argument("--target", type=Path, required=True, help="store to fill")
    args = parser.parse_args(argv)
    # A build step, not the API: plain lines on stderr (a no-op if logging is set up).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per hub request
    pinned = pins if pins is not None else load_pins()
    try:
        asyncio.run(bake(pinned, args.cache, args.target, hub or HuggingFaceHub()))
    except ModelStoreError as exc:
        for problem in exc.problems:
            logger.error("model store problem: %s", problem)  # noqa: TRY400 - no traceback needed
        return 1
    except ModelError as exc:
        model = getattr(exc, "model", None)
        logger.error("cannot bake %s: %s", model, type(exc).__name__)  # noqa: TRY400 - as above
        return 1
    logger.info("model store verified: %d models", len(pinned))
    return 0
