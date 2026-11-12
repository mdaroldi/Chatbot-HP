"""Regenerating models.toml (`make model-pins`) with a fake hub; fastembed's file lists are real."""

import io
from pathlib import Path

import pytest

from app.domain.ingestion import ModelDownloadError, ModelFiles, ModelPinsError
from app.infra.model_files import load_pins, render_pins
from app.infra.model_pins import FastembedFiles, fastembed_files, main, regenerate
from tests.model_data import NEXT_REVISION, REVISION, FakeHub, pin, sha256

SOURCE = "demo/dense-onnx"
NEW_FILES = {"model.onnx": b"new weights", "tokenizer.json": b"{}", "README.md": b"card"}


def describe(name: str) -> FastembedFiles:
    if name != "demo/dense":
        raise ModelPinsError(f"fastembed has no model named {name}")
    return FastembedFiles(
        source=SOURCE, required=("model.onnx",), optional=("tokenizer.json", "config.json")
    )


def hub(files: dict[str, bytes] | None = None) -> FakeHub:
    return FakeHub(
        repos={(SOURCE, NEXT_REVISION): NEW_FILES if files is None else files},
        heads={SOURCE: NEXT_REVISION},
    )


def test_regenerate_pins_fastembeds_files_at_the_current_revision(tmp_path: Path) -> None:
    fake = hub()

    [model] = regenerate(["demo/dense"], fake, describe, tmp_path)

    assert model == ModelFiles(
        name="demo/dense",
        source=SOURCE,
        revision=NEXT_REVISION,
        files={"model.onnx": sha256(b"new weights"), "tokenizer.json": sha256(b"{}")},
    )
    assert fake.downloads == [
        (SOURCE, NEXT_REVISION, ("model.onnx", "tokenizer.json"), tmp_path, True)
    ]


def test_repository_without_a_required_file_cannot_be_pinned(tmp_path: Path) -> None:
    with pytest.raises(ModelPinsError, match=r"demo/dense: .* lacks model.onnx"):
        regenerate(["demo/dense"], hub({"tokenizer.json": b"{}"}), describe, tmp_path)


def test_listed_file_the_hub_does_not_serve_is_a_download_error(tmp_path: Path) -> None:
    fake = hub()
    fake.repos[(SOURCE, NEXT_REVISION)] = {"model.onnx": b"w", "tokenizer.json": b"{}"}
    listed = fake.list_files

    def list_more(source: str, revision: str) -> list[str]:
        return [*listed(source, revision), "config.json"]

    fake.list_files = list_more  # type: ignore[method-assign]  # a hub that lists more than it serves

    with pytest.raises(ModelDownloadError, match=r"config\.json") as caught:
        regenerate(["demo/dense"], fake, describe, tmp_path)

    assert caught.value.model == "demo/dense"


def run(pins: Path, *args: str) -> str:
    out = io.StringIO()
    assert main(list(args), pins_path=pins, hub=hub(), describe=describe, stdout=out) == 0
    return out.getvalue()


def test_main_rewrites_the_pins_and_prints_the_diff(tmp_path: Path) -> None:
    pins = tmp_path / "models.toml"
    pins.write_text(render_pins([pin()]), encoding="utf-8")

    output = run(pins)

    assert f'-revision = "{REVISION}"' in output
    assert f'+revision = "{NEXT_REVISION}"' in output
    assert load_pins(pins)["demo/dense"].revision == NEXT_REVISION


def test_main_reports_unchanged_pins_and_leaves_the_file_alone(tmp_path: Path) -> None:
    pins = tmp_path / "models.toml"
    run(pins, "demo/dense")
    before = pins.read_text(encoding="utf-8")

    output = run(pins)

    assert output == f"{pins.name} is unchanged\n"
    assert pins.read_text(encoding="utf-8") == before


def test_main_pins_the_models_named_on_the_command_line(tmp_path: Path) -> None:
    pins = tmp_path / "models.toml"

    output = run(pins, "demo/dense")

    assert "+++" in output
    assert list(load_pins(pins)) == ["demo/dense"]


def test_fastembed_lists_the_dense_models_files() -> None:
    files = fastembed_files("BAAI/bge-small-en-v1.5")

    assert files.source == "Qdrant/bge-small-en-v1.5-onnx-Q"
    assert files.required == ("model_optimized.onnx",)
    assert {"tokenizer.json", "config.json", "special_tokens_map.json"} <= set(files.optional)


def test_fastembed_lists_the_sparse_models_files() -> None:
    files = fastembed_files("Qdrant/bm25")

    assert files.source == "Qdrant/bm25"
    assert {"config.json", "english.txt"} <= set(files.required)


def test_model_fastembed_does_not_know_cannot_be_pinned() -> None:
    with pytest.raises(ModelPinsError, match="fastembed has no model named demo/unknown"):
        fastembed_files("demo/unknown")
