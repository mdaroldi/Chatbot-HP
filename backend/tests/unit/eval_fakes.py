"""Golden sets, chunks, configurations and a scripted retriever for the eval runner tests."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenQuestion,
    GoldenSet,
    IndexedChunk,
    Match,
    OosKind,
    Split,
)
from app.domain.retrieval import (
    Relevance,
    RetrievalConfig,
    RetrievalResult,
    RetrievedChunk,
    SearchHit,
    SearchMode,
    StageTimings,
)
from app.evaluation.matrix import NamedConfig
from app.evaluation.runner import Retriever
from tests.unit.retrieval_fakes import make_hit

ENVY = "envy-6000-user-guide"
OMEN = "omen-17-service-guide"
DOC_IDS = frozenset({ENVY, OMEN})
GOLDEN_SHA = "ab" * 32
RERANKER = "fake/cross-encoder"


def question(
    qid: str,
    *,
    category: Category = Category.FACTUAL,
    split: Split = Split.DEV,
    facts: Sequence[Sequence[str]] = (("tray holds 80",),),
    doc_id: str = ENVY,
    match: Match = Match.ANY,
    text: str | None = None,
) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        category=category,
        split=split,
        question=text if text is not None else f"question {qid}?",
        match=match,
        anchors=tuple(Anchor(doc_id=doc_id, facts=tuple(group)) for group in facts),
    )


def oos(qid: str, kind: OosKind = OosKind.UNRELATED, split: Split = Split.DEV) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        category=Category.OUT_OF_SCOPE,
        split=split,
        question=f"question {qid}?",
        match=None,
        anchors=(),
        oos_kind=kind,
    )


def golden(*questions: GoldenQuestion) -> GoldenSet:
    return GoldenSet("1.0.0", GOLDEN_SHA, questions)


def chunk(chunk_id: str, text: str, doc_id: str = ENVY) -> IndexedChunk:
    return IndexedChunk(chunk_id, doc_id, ("Paper", "Load paper"), text)


def config(**overrides: Any) -> RetrievalConfig:
    # Any: per-case overrides of fields with different types; replace() validates them.
    base = RetrievalConfig(
        mode=SearchMode.HYBRID,
        rerank=True,
        top_k=20,
        rerank_candidates=20,
        prefetch_dense=20,
        prefetch_sparse=20,
        rrf_k=60,
        product_filter=False,
        relevance_threshold=None,
        dense_query_prefix="",
    )
    return dataclasses.replace(base, **overrides)


def named(name: str = "hybrid-rerank", **overrides: Any) -> NamedConfig:
    # Any: forwarded to config().
    return NamedConfig(name=name, config=config(**overrides), reranker=RERANKER)


def hits(*texts: str, doc_id: str = ENVY) -> list[SearchHit]:
    return [make_hit(f"c{n}", 1.0 / n, doc_id=doc_id, text=text) for n, text in enumerate(texts, 1)]


@dataclass
class Scripted:
    """What the retriever returns for one question."""

    hits: list[SearchHit]
    top_score: float | None = 1.0
    total_ms: float = 10.0
    rerank_ms: float | None = 5.0


@dataclass
class ScriptedRetriever:
    """Returns the scripted ranking per question text; records each query and config."""

    script: dict[str, Scripted] = field(default_factory=dict)
    calls: list[tuple[str, RetrievalConfig]] = field(default_factory=list)

    async def retrieve(self, query: str, config: RetrievalConfig) -> RetrievalResult:
        self.calls.append((query, config))
        scripted = self.script.get(query, Scripted([]))
        chunks = tuple(
            RetrievedChunk(
                hit=hit,
                rank=rank,
                rerank_score=(scripted.top_score - rank + 1)
                if config.rerank and scripted.top_score is not None
                else None,
            )
            for rank, hit in enumerate(scripted.hits[: config.top_k], 1)
        )
        signal = chunks[0].rerank_score if chunks else None
        return RetrievalResult(
            chunks=chunks,
            relevance=Relevance.from_top_score(signal, threshold=config.relevance_threshold),
            filter=None,
            timings=StageTimings(
                dense_embed_ms=1.0 if config.mode.uses_dense else None,
                sparse_embed_ms=1.0 if config.mode.uses_sparse else None,
                search_ms=2.0,
                rerank_ms=scripted.rerank_ms if config.rerank else None,
                total_ms=scripted.total_ms,
            ),
            config=config,
        )


_RETRIEVER: Retriever = ScriptedRetriever()
