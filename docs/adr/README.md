# Architecture decision records

One short record per significant decision: status, context, decision, alternatives
considered, consequences.

| ADR | Decision |
|---|---|
| [0001](0001-architecture-overview.md) | Architecture overview, with known limitations and deferred decisions |
| [0002](0002-qdrant-vector-database.md) | Qdrant as the vector database |
| [0003](0003-llm-provider-and-local-embeddings.md) | Anthropic behind a provider interface; local embeddings |
| [0004](0004-pdf-parsing.md) | PDF parsing with pdfplumber |
| [0005](0005-chunking-v1.md) | Structure-aware chunking v1 (450 tokens, prefix included) |
| [0006](0006-embedding-model-files.md) | Pinned and verified embedding model files, recorded per build |
| [0007](0007-startup-order.md) | Startup order: one-shot migrate and ingest services before the API |
| [0008](0008-baked-models-and-inference.md) | Models baked into the backend image, read offline; inference threading |
| [0009](0009-retrieval-strategy-and-evaluation.md) | Retrieval strategy and evaluation method |
| [0010](0010-chat-api.md) | Chat API: conversations, SSE answers, three refusal reasons, rate limits, retention |
