"""Model files for tests: a fake hub for unit tests, the real pinned models for model tests."""

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.settings import Settings
from app.domain.ingestion import ModelFiles, VerifiedModel
from app.infra.model_files import HuggingFaceHub, PinnedModelSource, load_pins, snapshot_dir

REVISION = "0123456789abcdef0123456789abcdef01234567"
NEXT_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
REPO_FILES = {"model.onnx": b"weights", "tokenizer.json": b"{}", "nested/vocab.txt": b"a b"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pin(name: str = "demo/dense", source: str = "demo/dense-onnx") -> ModelFiles:
    return ModelFiles(
        name=name,
        source=source,
        revision=REVISION,
        files={path: sha256(data) for path, data in REPO_FILES.items()},
    )


@dataclass
class FakeHub:
    """Serves in-memory repositories into a hub-style cache.

    Like the real hub cache, a download that is not forced keeps a file that is
    already in the snapshot, even a corrupted one.
    """

    repos: dict[tuple[str, str], dict[str, bytes]] = field(default_factory=dict)
    heads: dict[str, str] = field(default_factory=dict)
    error: Exception | None = None
    downloads: list[tuple[str, str, tuple[str, ...], Path, bool]] = field(default_factory=list)

    def download(
        self, source: str, revision: str, files: Sequence[str], cache_dir: Path, *, force: bool
    ) -> Path:
        self.downloads.append((source, revision, tuple(files), cache_dir, force))
        if self.error is not None:
            raise self.error
        target = snapshot_dir(cache_dir, source, revision)
        repo = self.repos[(source, revision)]
        for name in files:
            path = target / name
            if name in repo and (force or not path.exists()):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(repo[name])
        return target

    def current_revision(self, source: str) -> str:
        return self.heads[source]

    def list_files(self, source: str, revision: str) -> list[str]:
        return sorted(self.repos[(source, revision)])


def hub_with(source: str = "demo/dense-onnx", files: dict[str, bytes] | None = None) -> FakeHub:
    return FakeHub(repos={(source, REVISION): dict(REPO_FILES if files is None else files)})


def fetch_verified(settings: Settings, name: str) -> VerifiedModel:
    """The real pinned model, verified in EMBEDDING_CACHE_DIR (downloaded on first use)."""
    source = PinnedModelSource(load_pins(), settings.embedding_cache_dir, HuggingFaceHub())
    return asyncio.run(source.fetch(name))
