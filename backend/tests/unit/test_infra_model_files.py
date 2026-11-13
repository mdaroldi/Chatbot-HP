"""Pinned and verified model files (ADR 0006) on fake file trees and a fake hub."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.core.settings import Settings
from app.domain.ingestion import (
    ModelDownloadError,
    ModelIntegrityError,
    ModelNotAllowedError,
    ModelPinsError,
)
from app.infra.model_files import (
    MODEL_PINS,
    HuggingFaceHub,
    PinnedModelSource,
    hash_files,
    load_pins,
    parse_pins,
    render_pins,
    snapshot_dir,
)
from tests.model_data import REPO_FILES, REVISION, FakeHub, hub_with, pin, sha256

PINS = f"""
[[models]]
name = "demo/dense"
source = "demo/dense-onnx"
revision = "{REVISION}"

[models.files]
"model.onnx" = "{"a" * 64}"
"""

# ------------------------------------------------------------------ pins file


def test_pins_parse_into_model_files_by_name() -> None:
    pins = parse_pins(PINS)

    assert list(pins) == ["demo/dense"]
    assert pins["demo/dense"].revision == REVISION
    assert pins["demo/dense"].files == {"model.onnx": "a" * 64}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("models = [", "not valid TOML"),
        ("extra = 1\n" + PINS, "unknown top-level key: extra"),
        ("", "at least one model"),
        ("models = [1]", r"models\[0\] must be a table"),
        (PINS.replace('source = "demo/dense-onnx"\n', ""), r"models\[0\] missing key: source"),
        (PINS.replace('"demo/dense-onnx"', "3"), r"models\[0\] source must be a string"),
        (PINS.replace("[models.files]", 'license = "mit"\n[models.files]'), "unknown key: license"),
        (PINS.replace("[models.files]\n", "files = 1\n#"), r"files must be a table of strings"),
        (PINS.replace(REVISION, "abc"), r"models\[0\] revision must be a full commit SHA"),
        (PINS + PINS, "duplicate name: demo/dense"),
    ],
)
def test_invalid_pins_are_rejected(text: str, message: str) -> None:
    with pytest.raises(ModelPinsError, match=message):
        parse_pins(text)


def test_a_pins_error_names_the_model_only_when_it_is_known() -> None:
    with pytest.raises(ModelPinsError) as duplicate:
        parse_pins(PINS + PINS)
    with pytest.raises(ModelPinsError) as malformed:
        parse_pins("models = [1]")

    assert (duplicate.value.model, malformed.value.model) == ("demo/dense", None)


def test_rendered_pins_parse_back_to_the_same_models() -> None:
    models = [pin(), pin(name="demo/sparse", source="demo/sparse")]

    text = render_pins(models)

    assert text.startswith("# ")
    assert parse_pins(text) == {model.name: model for model in models}


def test_committed_pins_cover_the_default_models() -> None:
    defaults = Settings.model_fields

    pins = load_pins()

    assert set(pins) == {
        defaults["embedding_dense_model"].default,
        defaults["embedding_sparse_model"].default,
    }
    assert MODEL_PINS.read_text(encoding="utf-8") == render_pins(list(pins.values()))


def test_unreadable_pins_file_is_a_pins_error(tmp_path: Path) -> None:
    with pytest.raises(ModelPinsError, match="cannot read model pins"):
        load_pins(tmp_path / "missing.toml")


# ------------------------------------------------------------------ hashing


def test_snapshot_dir_follows_the_hub_cache_layout() -> None:
    assert snapshot_dir(Path("/cache"), "Org/model", REVISION) == Path(
        f"/cache/models--Org--model/snapshots/{REVISION}"
    )


def test_files_are_hashed_by_content_through_symlinks(tmp_path: Path) -> None:
    (tmp_path / "blob").write_bytes(b"weights")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "vocab.txt").write_bytes(b"a b")
    (tmp_path / "model.onnx").symlink_to(tmp_path / "blob")  # how the hub cache links files

    digests = hash_files(tmp_path, ["model.onnx", "nested/vocab.txt", "absent.json"])

    assert digests == {
        "model.onnx": sha256(b"weights"),
        "nested/vocab.txt": sha256(b"a b"),
        "absent.json": None,
    }


# ------------------------------------------------------------------ pinned source


def source(hub: FakeHub, cache: Path) -> PinnedModelSource:
    return PinnedModelSource({"demo/dense": pin()}, cache, hub)


async def test_first_fetch_downloads_only_the_pinned_revision_and_files(tmp_path: Path) -> None:
    hub = hub_with(files=REPO_FILES | {"extra.bin": b"not pinned"})

    verified = await source(hub, tmp_path).fetch("demo/dense")

    assert hub.downloads == [
        ("demo/dense-onnx", REVISION, tuple(sorted(REPO_FILES)), tmp_path, False)
    ]
    assert verified.path == snapshot_dir(tmp_path, "demo/dense-onnx", REVISION)
    assert verified.files == pin()
    assert not (verified.path / "extra.bin").exists()


async def test_verified_cache_is_used_without_any_hub_call(tmp_path: Path) -> None:
    await source(hub_with(), tmp_path).fetch("demo/dense")
    offline = FakeHub(error=AssertionError("no network expected"))

    verified = await source(offline, tmp_path).fetch("demo/dense")

    assert offline.downloads == []
    assert verified.files == pin()


async def test_model_without_a_pin_is_never_downloaded(tmp_path: Path) -> None:
    hub = hub_with()

    with pytest.raises(ModelNotAllowedError, match="other/model") as refused:
        await source(hub, tmp_path).fetch("other/model")

    assert refused.value.model == "other/model"
    assert hub.downloads == []


async def test_bad_cached_file_triggers_one_forced_download(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hub = hub_with()
    first = await source(hub, tmp_path).fetch("demo/dense")
    (first.path / "model.onnx").write_bytes(b"tampered")

    with caplog.at_level(logging.WARNING):
        verified = await source(hub, tmp_path).fetch("demo/dense")

    assert [force for *_, force in hub.downloads] == [False, True]
    assert verified.files == pin()
    [record] = [r for r in caplog.records if r.name == "app.infra.model_files"]
    assert record.getMessage() == "cached model files do not match their pins; downloading again"
    assert (record.__dict__["model"], record.__dict__["file"]) == ("demo/dense", "model.onnx")


async def test_still_bad_after_the_fresh_download_is_an_integrity_error(tmp_path: Path) -> None:
    hub = hub_with(files=REPO_FILES | {"model.onnx": b"served corrupted"})

    with pytest.raises(ModelIntegrityError) as caught:
        await source(hub, tmp_path).fetch("demo/dense")

    assert (caught.value.file, caught.value.actual) == ("model.onnx", sha256(b"served corrupted"))
    assert caught.value.expected == pin().files["model.onnx"]
    assert len(hub.downloads) == 1


async def test_bad_cache_and_bad_download_stop_after_one_retry(tmp_path: Path) -> None:
    corrupted = REPO_FILES | {"model.onnx": b"served corrupted"}
    target = snapshot_dir(tmp_path, "demo/dense-onnx", REVISION)
    target.mkdir(parents=True)
    (target / "model.onnx").write_bytes(b"stale")
    hub = hub_with(files=corrupted)

    with pytest.raises(ModelIntegrityError):
        await source(hub, tmp_path).fetch("demo/dense")

    assert [force for *_, force in hub.downloads] == [True]


async def test_file_the_hub_does_not_serve_is_reported_missing(tmp_path: Path) -> None:
    hub = hub_with(files={k: v for k, v in REPO_FILES.items() if k != "tokenizer.json"})

    with pytest.raises(ModelIntegrityError, match=r"tokenizer\.json is missing"):
        await source(hub, tmp_path).fetch("demo/dense")


async def test_hub_failure_is_a_download_error_not_an_integrity_error(tmp_path: Path) -> None:
    cause = OSError("connection reset")
    hub = FakeHub(error=cause)

    with pytest.raises(ModelDownloadError, match="demo/dense") as caught:
        await source(hub, tmp_path).fetch("demo/dense")

    assert caught.value.__cause__ is cause
    assert caught.value.model == "demo/dense"
    assert not isinstance(caught.value, ModelIntegrityError)


# ------------------------------------------------------------------ hub adapter


@dataclass
class FakeInfo:
    sha: str | None


@dataclass
class FakeApi:
    sha: str | None = REVISION
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def model_info(
        self, repo_id: str, *, timeout: float | None = None, token: bool | str | None = None
    ) -> FakeInfo:
        self.calls.append(("model_info", {"repo_id": repo_id, "timeout": timeout, "token": token}))
        return FakeInfo(self.sha)

    def list_repo_files(
        self, repo_id: str, *, revision: str | None = None, token: bool | str | None = None
    ) -> list[str]:
        self.calls.append(("list", {"repo_id": repo_id, "revision": revision, "token": token}))
        return ["b.txt", "a.txt"]


def test_hub_download_asks_for_the_exact_revision_and_files_anonymously(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def snapshot(repo_id: str, **kwargs: object) -> object:
        calls.append({"repo_id": repo_id} | kwargs)
        return str(tmp_path / "snap")

    hub = HuggingFaceHub(snapshot=snapshot, api=FakeApi())

    path = hub.download("Org/model", REVISION, ["a.txt", "b.txt"], tmp_path, force=True)

    assert path == tmp_path / "snap"
    assert calls == [
        {
            "repo_id": "Org/model",
            "revision": REVISION,
            "allow_patterns": ["a.txt", "b.txt"],
            "cache_dir": str(tmp_path),
            "force_download": True,
            "token": False,
        }
    ]


def test_hub_reports_the_current_revision_and_the_files_of_one() -> None:
    api = FakeApi()
    hub = HuggingFaceHub(api=api)

    assert hub.current_revision("Org/model") == REVISION
    assert hub.list_files("Org/model", REVISION) == ["a.txt", "b.txt"]
    assert api.calls == [
        ("model_info", {"repo_id": "Org/model", "timeout": 30.0, "token": False}),
        ("list", {"repo_id": "Org/model", "revision": REVISION, "token": False}),
    ]


def test_repository_without_a_revision_is_a_download_error() -> None:
    with pytest.raises(ModelDownloadError, match="Org/model") as caught:
        HuggingFaceHub(api=FakeApi(sha=None)).current_revision("Org/model")

    assert caught.value.model == "Org/model"
