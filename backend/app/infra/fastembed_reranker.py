"""Cross-encoder reranking with a local fastembed model on CPU (ADRs 0003 and 0008)."""

from collections.abc import Iterable, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import Protocol, Self

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.core.settings import Settings
from app.domain.ingestion import VerifiedModel
from app.infra.inference import run_inference


class CrossEncoderModel(Protocol):
    def rerank(self, query: str, documents: Iterable[str], batch_size: int) -> Iterable[float]: ...


class CrossEncoderClass(Protocol):
    def __call__(
        self,
        model_name: str,
        *,
        cache_dir: str,
        threads: int | None,
        specific_model_path: str,
        local_files_only: bool,
    ) -> CrossEncoderModel: ...


class FastembedReranker:
    """Reranker over a fastembed cross-encoder loaded from its verified folder, offline.

    All candidates go to the model in one batch; the scores are raw logits, higher is
    more relevant, in input order. The texts are the stored chunk texts, breadcrumb
    included; fastembed truncates a pair at the model's 512 tokens (ADR 0008).
    """

    def __init__(
        self,
        model: VerifiedModel,
        cache_dir: Path,
        *,
        threads: int | None,
        executor: Executor | None,
        model_class: CrossEncoderClass = TextCrossEncoder,
    ) -> None:
        self._name = model.files.name
        self._executor = executor
        self._model = model_class(
            model.files.name,
            cache_dir=str(cache_dir),
            threads=threads,
            specific_model_path=str(model.path),
            local_files_only=True,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        model: VerifiedModel,
        executor: Executor | None,
        *,
        model_class: CrossEncoderClass = TextCrossEncoder,
    ) -> Self:
        return cls(
            model,
            settings.embedding_cache_dir,
            threads=settings.inference_onnx_threads,
            executor=executor,
            model_class=model_class,
        )

    @property
    def model(self) -> str:
        return self._name

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        batch = list(texts)
        return await run_inference(self._executor, lambda: self._score(query, batch))

    def _score(self, query: str, texts: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, texts, batch_size=len(texts))]
