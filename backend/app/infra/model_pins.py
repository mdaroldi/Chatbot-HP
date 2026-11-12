"""Regenerate models.toml from a fresh download (`make model-pins`, ADR 0006).

Run by a maintainer only, never automatically: it asks the hub for each model's current
revision, downloads it into a temporary folder (never the model cache), hashes the
files fastembed loads, rewrites the pins and prints the diff for review.
"""

import difflib
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from fastembed import SparseTextEmbedding, TextEmbedding

from app.domain.ingestion import ModelDownloadError, ModelFiles, ModelPinsError
from app.infra.model_files import (
    MODEL_PINS,
    HubClient,
    HuggingFaceHub,
    hash_files,
    load_pins,
    render_pins,
)

# fastembed downloads these with every model when the repository has them (fastembed
# 0.9.1, model_management.py); the tokenizer files are what the token counter loads.
FASTEMBED_COMMON_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)


@dataclass(frozen=True)
class FastembedFiles:
    """Where fastembed gets a model, and which files it loads."""

    source: str
    required: tuple[str, ...]  # the model file and its additional files
    optional: tuple[str, ...]  # common files, pinned when the repository has them


class PinsHub(HubClient, Protocol):
    def current_revision(self, source: str) -> str: ...

    def list_files(self, source: str, revision: str) -> list[str]: ...


def fastembed_files(name: str) -> FastembedFiles:
    for model_class in (TextEmbedding, SparseTextEmbedding):
        # dict[str, Any]: fastembed's public model descriptions are untyped dicts.
        for description in model_class.list_supported_models():
            if description["model"] == name:
                required = (description["model_file"], *description["additional_files"])
                optional = tuple(f for f in FASTEMBED_COMMON_FILES if f not in required)
                return FastembedFiles(description["sources"]["hf"], required, optional)
    raise ModelPinsError(f"fastembed has no model named {name}")


def _pin(name: str, hub: PinsHub, wanted: FastembedFiles, workdir: Path) -> ModelFiles:
    revision = hub.current_revision(wanted.source)
    listed = set(hub.list_files(wanted.source, revision))
    missing = [f for f in wanted.required if f not in listed]
    if missing:
        raise ModelPinsError(f"{name}: {wanted.source}@{revision} lacks {', '.join(missing)}")
    files = sorted(f for f in (*wanted.required, *wanted.optional) if f in listed)
    folder = hub.download(wanted.source, revision, files, workdir, force=True)
    digests = hash_files(folder, files)
    absent = [f for f, digest in digests.items() if digest is None]
    if absent:
        raise ModelDownloadError(f"{name}: the hub did not serve {', '.join(absent)}")
    computed = {f: digest for f, digest in digests.items() if digest is not None}
    return ModelFiles(name, wanted.source, revision, computed)


def regenerate(
    names: Sequence[str],
    hub: PinsHub,
    describe: Callable[[str], FastembedFiles],
    workdir: Path,
) -> list[ModelFiles]:
    return [_pin(name, hub, describe(name), workdir) for name in names]


def main(
    argv: Sequence[str],
    *,
    pins_path: Path = MODEL_PINS,
    hub: PinsHub | None = None,
    describe: Callable[[str], FastembedFiles] = fastembed_files,
    stdout: TextIO = sys.stdout,
) -> int:
    """Pin the models named in argv, or by default the models already pinned."""
    names = list(argv) or list(load_pins(pins_path))
    with tempfile.TemporaryDirectory(prefix="model-pins-") as workdir:
        models = regenerate(names, hub or HuggingFaceHub(), describe, Path(workdir))
    new = render_pins(models)
    old = pins_path.read_text(encoding="utf-8") if pins_path.exists() else ""
    if new == old:
        stdout.write(f"{pins_path.name} is unchanged\n")
        return 0
    pins_path.write_text(new, encoding="utf-8")
    stdout.writelines(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            f"a/{pins_path.name}",
            f"b/{pins_path.name}",
        )
    )
    return 0
