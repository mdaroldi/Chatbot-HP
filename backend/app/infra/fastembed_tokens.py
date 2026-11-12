"""Token counts with the dense model's own tokenizer (ADR 0005)."""

from pathlib import Path
from typing import Protocol, Self

from fastembed import TextEmbedding

from app.core.settings import Settings


class TokenizingModel(Protocol):
    def token_count(self, texts: str) -> int: ...


class TokenizingModelClass(Protocol):
    def __call__(self, model_name: str, *, cache_dir: str, lazy_load: bool) -> TokenizingModel: ...


class FastembedTokenCounter:
    """TokenCounter that measures text exactly as the dense model will see it.

    The chunk limit exists because the dense model truncates its input, so the count
    must come from that model's tokenizer, not an approximation. It includes the
    [CLS] and [SEP] tokens, and it saturates at the model's input limit (512 for
    bge-small): a longer text counts as the limit, which is still over any chunk limit
    below it. `lazy_load` loads the tokenizer only, never the ONNX session.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        model_class: TokenizingModelClass = TextEmbedding,
    ) -> None:
        self._model = model_class(model_name, cache_dir=str(cache_dir), lazy_load=True)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, model_class: TokenizingModelClass = TextEmbedding
    ) -> Self:
        return cls(
            settings.embedding_dense_model, settings.embedding_cache_dir, model_class=model_class
        )

    def count(self, text: str) -> int:
        return self._model.token_count(text)
