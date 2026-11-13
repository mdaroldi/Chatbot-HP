"""The model store baked into the backend image (ADR 0008), with a fake hub."""

import logging
from pathlib import Path

import pytest

from app.domain.ingestion import ModelDownloadError, ModelIntegrityError, ModelStoreError
from app.infra.model_files import snapshot_dir
from app.infra.model_store import bake, check_snapshot, main, snapshot_files, verify_store
from tests.model_data import REPO_FILES, REVISION, FakeHub, hub_with, pin, sha256

PIN = pin()


def digests(files: dict[str, bytes]) -> dict[str, str]:
    return {path: sha256(data) for path, data in files.items()}


# ------------------------------------------------------------------ the lock verifier (pure)


def test_matching_files_have_no_problem() -> None:
    assert check_snapshot(PIN, digests(REPO_FILES)) == []


def test_a_changed_file_is_reported_with_both_digests() -> None:
    found = digests({**REPO_FILES, "model.onnx": b"other weights"})

    [problem] = check_snapshot(PIN, found)

    assert "demo/dense: model.onnx has SHA-256" in problem
    assert sha256(b"other weights") in problem
    assert PIN.files["model.onnx"] in problem


def test_a_missing_file_is_reported() -> None:
    found = digests({k: v for k, v in REPO_FILES.items() if k != "tokenizer.json"})

    assert check_snapshot(PIN, found) == ["demo/dense: tokenizer.json is missing"]


def test_an_extra_file_is_reported() -> None:
    found = digests({**REPO_FILES, "onnx/model_fp16.onnx": b"unpinned"})

    assert check_snapshot(PIN, found) == ["demo/dense: onnx/model_fp16.onnx is not pinned"]


# ------------------------------------------------------------------ files on disk


def write_snapshot(root: Path, files: dict[str, bytes]) -> Path:
    folder = snapshot_dir(root, PIN.source, PIN.revision)
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    return folder


def test_snapshot_files_hashes_every_file_below_the_folder(tmp_path: Path) -> None:
    folder = write_snapshot(tmp_path, REPO_FILES)

    assert snapshot_files(folder) == digests(REPO_FILES)


def test_a_complete_store_verifies(tmp_path: Path) -> None:
    write_snapshot(tmp_path, REPO_FILES)

    assert verify_store({PIN.name: PIN}, tmp_path) == []


def test_a_store_without_a_pinned_model_is_reported(tmp_path: Path) -> None:
    assert verify_store({PIN.name: PIN}, tmp_path) == [
        f"demo/dense: {PIN.source}@{REVISION} is not in the store"
    ]


def test_a_store_with_an_unpinned_model_or_revision_is_reported(tmp_path: Path) -> None:
    write_snapshot(tmp_path, REPO_FILES)
    (tmp_path / "models--demo--stray" / "snapshots" / REVISION).mkdir(parents=True)
    (tmp_path / "models--demo--dense-onnx" / "snapshots" / ("f" * 40)).mkdir(parents=True)

    problems = verify_store({PIN.name: PIN}, tmp_path)

    assert problems == [
        f"models--demo--dense-onnx/snapshots/{'f' * 40} is not pinned",
        f"models--demo--stray/snapshots/{REVISION} is not pinned",
    ]


# ------------------------------------------------------------------ baking


async def test_bake_copies_the_verified_files_as_real_files(tmp_path: Path) -> None:
    cache, target = tmp_path / "cache", tmp_path / "store"

    await bake({PIN.name: PIN}, cache, target, hub_with())

    folder = snapshot_dir(target, PIN.source, PIN.revision)
    assert snapshot_files(folder) == dict(PIN.files)
    assert not any(path.is_symlink() for path in folder.rglob("*"))


async def test_bake_reuses_a_verified_cache_without_the_hub(tmp_path: Path) -> None:
    cache, target = tmp_path / "cache", tmp_path / "store"
    await bake({PIN.name: PIN}, cache, target / "first", hub_with())
    offline = FakeHub(error=ConnectionError("no network"))

    await bake({PIN.name: PIN}, cache, target / "second", offline)

    assert offline.downloads == []


async def test_bake_retries_a_download_failure_with_backoff(tmp_path: Path) -> None:
    flaky = hub_with()
    flaky.failures.append(ConnectionError("reset"))
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    await bake({PIN.name: PIN}, tmp_path / "c", tmp_path / "s", flaky, sleep=sleep)

    assert waits == [10.0]
    assert verify_store({PIN.name: PIN}, tmp_path / "s") == []


async def test_bake_gives_up_after_the_last_attempt(tmp_path: Path) -> None:
    down = FakeHub(error=ConnectionError("no network"))
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    with pytest.raises(ModelDownloadError):
        await bake({PIN.name: PIN}, tmp_path / "c", tmp_path / "s", down, sleep=sleep)

    assert waits == [10.0, 20.0]
    assert len(down.downloads) == 3


async def test_bake_never_retries_files_that_differ_from_their_pins(tmp_path: Path) -> None:
    changed = hub_with(files={**REPO_FILES, "model.onnx": b"other weights"})
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    with pytest.raises(ModelIntegrityError):
        await bake({PIN.name: PIN}, tmp_path / "c", tmp_path / "s", changed, sleep=sleep)

    assert waits == []


async def test_bake_fails_when_the_store_holds_something_unpinned(tmp_path: Path) -> None:
    target = tmp_path / "store"
    (target / "models--demo--stray" / "snapshots" / REVISION).mkdir(parents=True)

    with pytest.raises(ModelStoreError) as caught:
        await bake({PIN.name: PIN}, tmp_path / "cache", target, hub_with())

    assert caught.value.problems == (f"models--demo--stray/snapshots/{REVISION} is not pinned",)


# ------------------------------------------------------------------ CLI


def test_main_bakes_every_pin_into_the_target(tmp_path: Path) -> None:
    code = main(
        ["--cache", str(tmp_path / "cache"), "--target", str(tmp_path / "store")],
        pins={PIN.name: PIN},
        hub=hub_with(),
    )

    assert code == 0
    assert verify_store({PIN.name: PIN}, tmp_path / "store") == []


def test_main_exits_non_zero_and_names_the_model_on_a_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    changed = hub_with(files={**REPO_FILES, "model.onnx": b"other weights"})

    code = main(
        ["--cache", str(tmp_path / "c"), "--target", str(tmp_path / "s")],
        pins={PIN.name: PIN},
        hub=changed,
    )

    assert code == 1
    assert "demo/dense" in caplog.text
    assert "ModelIntegrityError" in caplog.text


def test_main_keeps_the_hub_requests_out_of_the_build_log(tmp_path: Path) -> None:
    """httpx logs every request URL at INFO; the build log needs only the outcome."""
    logging.getLogger("httpx").setLevel(logging.NOTSET)

    main(
        ["--cache", str(tmp_path / "c"), "--target", str(tmp_path / "s")],
        pins={PIN.name: PIN},
        hub=hub_with(),
    )

    # Its own level, not the effective one, which the root's default would satisfy.
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_main_lists_every_store_problem_and_exits_non_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "store"
    (target / "models--demo--stray" / "snapshots" / REVISION).mkdir(parents=True)

    code = main(
        ["--cache", str(tmp_path / "c"), "--target", str(target)],
        pins={PIN.name: PIN},
        hub=hub_with(),
    )

    assert code == 1
    assert f"models--demo--stray/snapshots/{REVISION} is not pinned" in caplog.text
