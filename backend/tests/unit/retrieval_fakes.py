"""In-memory fakes for the retrieval interfaces (no models, no Qdrant)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.domain.ingestion import ContentType, PageRange, SparseVector
from app.domain.retrieval import QueryEmbedder, Reranker, SearchHit, SearchRequest, SearchStore

DENSE_VECTOR = [0.1, 0.2, 0.3]
SPARSE_VECTOR = SparseVector((7, 11), (1.0, 1.0))


def make_hit(
    chunk_id: str,
    score: float = 0.5,
    *,
    doc_id: str = "omen-17-service-guide",
    text: str | None = None,
    path: tuple[str, ...] = ("Removal and replacement", "Bottom cover"),
    content_type: ContentType = ContentType.PROCEDURE,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_path=path,
        pages=PageRange(30, 31, 28, 29),
        content_type=content_type,
        has_warning=False,
        text=text if text is not None else f"text of {chunk_id}",
        score=score,
    )


@dataclass
class FakeQueryEmbedder:
    """Fixed query vectors; records every text it was asked to embed."""

    dense: list[float] = field(default_factory=lambda: list(DENSE_VECTOR))
    sparse: SparseVector = SPARSE_VECTOR
    error: Exception | None = None
    dense_texts: list[str] = field(default_factory=list)
    sparse_texts: list[str] = field(default_factory=list)

    async def embed_query_dense(self, text: str) -> list[float]:
        self.dense_texts.append(text)
        if self.error is not None:
            raise self.error
        return list(self.dense)

    async def embed_query_sparse(self, text: str) -> SparseVector:
        self.sparse_texts.append(text)
        if self.error is not None:
            raise self.error
        return self.sparse


@dataclass
class FakeSearchStore:
    """Returns its hits (best first), cut at the request's limit; records requests."""

    hits: list[SearchHit] = field(default_factory=list)
    error: Exception | None = None
    requests: list[SearchRequest] = field(default_factory=list)

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        hits = [h for h in self.hits if request.doc_id is None or h.doc_id == request.doc_id]
        return hits[: request.limit]


@dataclass
class FakeReranker:
    """Scores each text with `score`; records the query and texts of every call."""

    score: Callable[[str], float] = len
    error: Exception | None = None
    model: str = "fake/cross-encoder"
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.error is not None:
            raise self.error
        return [float(self.score(text)) for text in texts]


# The fakes must keep implementing the domain interfaces (checked by mypy).
_EMBEDDER: QueryEmbedder = FakeQueryEmbedder()
_STORE: SearchStore = FakeSearchStore()
_RERANKER: Reranker = FakeReranker()
