"""A ChatService over fakes of every port, shared by the service and the SSE API tests.

Real domain code and the real RetrievalService and QueryRewriter run over fakes of the
store, search, models, language model and index identity.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.domain.llm import LLMProvider
from app.domain.prompt import HistoryBudget
from app.domain.retrieval import IndexBuildRef, RetrievalConfig, SearchHit, SearchMode
from app.services.chat import ChatConfig, ChatService
from app.services.retrieval import RetrievalService
from app.services.rewrite import QueryRewriter, RewriteConfig
from tests.unit.chat_fakes import InMemoryChatStore
from tests.unit.llm_fakes import FakeProvider
from tests.unit.retrieval_fakes import FakeQueryEmbedder, FakeReranker, FakeSearchStore, make_hit

CLIENT = "client-a"
OMEN, ENVY = "omen-17-service-guide", "envy-6000-user-guide"
TITLES = {OMEN: "Synthetic OMEN guide", ENVY: "Synthetic ENVY guide"}
BUILD = IndexBuildRef("hpdocs_chunks_20261113t173108z", "20261113t173108z")


def retrieval_config(**overrides: Any) -> RetrievalConfig:  # Any: per-test field overrides
    fields: dict[str, Any] = {
        "mode": SearchMode.HYBRID,
        "rerank": True,
        "top_k": 3,
        "rerank_candidates": 4,
        "prefetch_dense": 4,
        "prefetch_sparse": 4,
        "rrf_k": 60,
        "product_filter": False,
        "relevance_threshold": 0.0,
        "dense_query_prefix": "",
    }
    fields.update(overrides)
    return RetrievalConfig(**fields)


CONFIG = ChatConfig(
    model="configured-answer-model",
    max_tokens=900,
    effort="low",
    history=HistoryBudget(max_turns=4, max_tokens=4_000),
    snippet_chars=40,
)
REWRITE = RewriteConfig(
    model="configured-rewrite-model",
    max_tokens=60,
    effort="low",
    timeout_seconds=5.0,
    budget=HistoryBudget(max_turns=3, max_tokens=1_000),
)

# The fake reranker scores a chunk by its text length, so the rank order is c2, c3, c1
# (c4 is shortest and falls outside top_k = 3): tags S1, S2, S3 are c2, c3, c1.
HITS = [
    make_hit("c1", text="Breadcrumb > c1\n\nTen chars."),
    make_hit("c2", text="Breadcrumb > c2\n\nThe longest synthetic body of all four chunks here."),
    make_hit("c3", text="Breadcrumb > c3\n\nA middle-sized synthetic body.", doc_id=ENVY),
    make_hit("c4", text="c4"),
]
RANKED = ("c2", "c3", "c1")
ANSWER_PIECES = ("Remove the cover [S1]. ", "Then the fan [S2][S9]. ")


@dataclass
class FakeIndexIdentity:
    ref: IndexBuildRef = BUILD
    error: Exception | None = None
    calls: int = 0

    async def current(self) -> IndexBuildRef:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.ref


def ticking() -> Callable[[], float]:
    """A clock that moves 10 ms per reading, so every duration is positive."""
    readings = itertools.count()
    return lambda: next(readings) / 100


@dataclass
class Rig:
    answer: FakeProvider = field(default_factory=lambda: FakeProvider(ANSWER_PIECES))
    rewrite: FakeProvider = field(default_factory=lambda: FakeProvider(("cover fan removal",)))
    hits: list[SearchHit] = field(default_factory=lambda: list(HITS))
    store: InMemoryChatStore = field(default_factory=InMemoryChatStore)
    index: FakeIndexIdentity = field(default_factory=FakeIndexIdentity)
    reranker: FakeReranker = field(default_factory=FakeReranker)
    embedder: FakeQueryEmbedder = field(default_factory=FakeQueryEmbedder)
    search: FakeSearchStore = field(init=False)

    def __post_init__(self) -> None:
        self.search = FakeSearchStore(self.hits)

    def service(
        self, config: RetrievalConfig | None = None, *, provider: LLMProvider | None = None
    ) -> ChatService:
        """`provider` replaces the scripted answer model (e.g. with the real stub)."""
        retrieval = RetrievalService(self.embedder, self.search, self.reranker)
        return ChatService(
            store=self.store,
            retrieval=retrieval,
            retrieval_config=config or retrieval_config(),
            rewriter=QueryRewriter(self.rewrite, REWRITE),
            provider=provider or self.answer,
            index=self.index,
            titles=TITLES,
            config=CONFIG,
            clock=ticking(),
        )

    async def conversation(self) -> str:
        return (await self.store.create_conversation(CLIENT)).id
