# ADR 0011: Chat UI: Streamlit page, state model, safe rendering and the page guard

- **Status:** accepted (Phase 4)
- **Date:** 2026-11-14

## Context

Phase 4 replaces the Streamlit status page with the chat: a conversation list, a
thread with sources, streamed answers, refusals and errors, all over the chat API of
ADR 0010. The challenge asks for a graphical frontend that talks to the backend only
over HTTP. Three properties shape the design. Streamlit reruns the whole script on
every interaction. The answers are untrusted text (model output shaped by retrieved
excerpts and by the user's question). And chat content is personal data that must
never reach the logs (CLAUDE.md).

## Decision

### Page and state

- `app.py` holds only the run order; the logic lives in `hpdocs_ui`. The UI state is
  one frozen `UiState`, and every change is a pure function in `hpdocs_ui/state.py`,
  so the behaviour across reruns is unit-tested without Streamlit. Widget code
  (`hpdocs_ui/ui/`) reads the state and applies transitions.
- **Identity in the URL.** The anonymous client ID is a random UUID4 kept in the `cid`
  query parameter and sent as `X-Client-ID`. The open conversation is in `conv`. A
  value that fails the backend's identifier rule is replaced, never repaired or
  echoed. The sidebar says "This page's link opens your conversations. Don't share
  it."
- **Lazy creation.** "New conversation" creates nothing; the conversation is created
  with its first question. The URL gets `conv` before the answer streams, so a reload
  mid-answer finds the conversation.
- **One send per submission.** The submit callback puts the question in an outbox.
  The next run sees it before drawing anything, so it draws every widget disabled,
  pops the outbox before any request, streams, stores the outcome and reruns. A
  rerun can never send a question twice. The input stays enabled when an answer is
  `pending` on reload (a crash leftover stays pending until the next turn).
- **`BACKEND_URL`** is compose wiring (`http://backend:8000`), not configuration, and
  is kept out of `.env.example`; a test pins it.
- **Timeout coupling.** No byte arrives before the first token, so
  `UI_STREAM_READ_TIMEOUT_SECONDS` (60) must exceed the rewrite, retrieval and
  first-token deadlines and the SSE heartbeat. `UI_STREAM_TOTAL_TIMEOUT_SECONDS` (185)
  must cover the rewrite plus the longest answer plus 60 s. Tests on `.env.example`
  check both sums. Only a read timeout maps to `stream_timeout`, shown as a stopped
  answer.

### Answers, errors and the stored view

- **Before the first token** nothing is stored. The error wording is shown with the
  question in a copyable code block. There is no Retry button (approved: the user
  resends on purpose). **After it**, the partial answer stays in the thread with a
  notice, and the backend has stored it as `failed` (or `cancelled`).
- **Uncited.** GET has no `flags`, so a stored answer is shown as uncited when it is
  `completed` with no citations, the same rule the backend uses to set the flag.
  The streamed view uses `flags.uncited`.
- **After a reload, `failed` and `cancelled` lose their error code**: GET stores the
  outcome, not why. The thread shows "interrupted" or "stopped before it finished".
- `pending` shows a Refresh button. A 409 `conversation_busy` shows its wording, the
  request ID and Refresh, and nothing is retried.
- Only the latest answer opens its Sources expander, so its citations show without a
  click. A source with a WARNING or CAUTION note carries a badge.
- The spinner has no text; the priming caption sits beside it.

### Safe rendering

- **`unsafe_allow_html=False` is the primary guard.** No call opts in to HTML, and a
  test forbids the argument anywhere in the frontend. An AST test checks every
  Markdown-rendering call: its argument must be a literal, a sanitizer or escaper
  call, or a named fixed-wording constant. The exceptions it lists are
  `view.headline` and `error.text`, both fixed wording.
- **The sanitizer is defence in depth** against what Streamlit still renders with HTML
  off: images (a request to any host), live links (a phishing target) and math. Answer
  text is the only Markdown on the page. It is re-sanitized as a whole on every
  streamed token, so a link split across tokens never renders, even for one update.
- **Code detection.** Code is kept byte for byte only where the sanitizer's reading
  provably matches a CommonMark renderer's. A fenced block counts only when every
  fence-like line starts at column 0, so no list or quote can own a fence. A code span
  counts only when it closes on the same line with no `|` inside, because GFM splits
  table cells first. Every other backtick is escaped.
- **Indented fences (commit 5b).** A fence indented 1 to 3 spaces is treated as text.
  The visible cost: inside such a fence, which the renderer does show as code, an
  escape shows as a literal backslash. It looks wrong but is never unsafe.
- **markdown-it-py is a proxy.** The sanitizer tests use it as a CommonMark reference
  oracle (a dev dependency). Streamlit renders Markdown in the browser with its own
  JavaScript engine (GFM plus math and Streamlit directives such as `:red[...]`), so
  the oracle is not the renderer. A manual renderer check closes the gap: a throwaway
  page renders the hostile corpus, sanitized and as a control, with every target on
  `evil.invalid`. DevTools must show zero requests to that host on the sanitized page
  and some on the control page.

### Privacy wording

- With `llm=anthropic` the sidebar says: "Your questions, recent messages and the
  manual excerpts found for them are sent to the Anthropic API to write the answers."
  It names recent messages because the rewrite and the answer prompt include recent
  history (ADR 0010). With the stub it says nothing is sent to Anthropic; when
  `/ready` is unreachable it uses a conditional version. The empty state shows a
  one-line intro and four example questions, then repeats the line as a caption.

### The page guard (rule 1)

- **Finding (commit 1).** Streamlit logs an uncaught script exception with its message
  and source lines. Either could hold a question or an answer. `showErrorDetails =
  "none"` hides them from the page but not from the logs.
- **Decision.** `app.py` runs the page through `ui/guard.run_page`. It catches
  `Exception`, logs only the type ("page run failed: RuntimeError"), and shows
  "Something went wrong. Try again." Streamlit's stop and rerun derive from
  `BaseException` and pass through, as do interrupts. A canary test sends a canary
  through every flow and error and checks the page, every `streamlit`, `hpdocs_ui`
  and `httpx` logger at DEBUG, and the process output. Neither the canary nor the
  client ID appears. httpx and httpcore are set to WARNING, because their INFO lines
  carry URLs with conversation IDs.
- **Rerun once.** A failed run is rerun a single time, so a run that drew every widget
  disabled (an answer streaming) is replaced by an enabled page. A second failure in
  a row shows the wording and does not rerun again, so a persistent bug cannot loop.
- **Callbacks run outside the page guard (finding of commit 9b).** Streamlit runs
  widget callbacks before the script (`script_runner.py:809` in the pinned
  version), outside any try block in the page. Each callback is therefore wrapped
  with `@guard.callback`, and an AST test requires the decorator on every
  `on_click`, `on_submit` and `on_change` callback. A failed callback sets the flag
  the next page run shows.
- **Stale thread.** After a caught failure the next run reads the selected thread from
  the backend again (`on_failure=session.forget_thread`). The backend may have stored
  the turn (as `cancelled`, say) while the page failed, and the page must not show
  the thread as it was before the question.

### Testing

- Unit tests drive the real script with Streamlit's AppTest against `FakeChatApi`, a
  fake of the `ChatApi` protocol. A recorder at the forward-message queue sees every
  intermediate element, such as each streamed update and the disabled widgets. Both
  projects keep 100% branch coverage.
- **`make test-ui-int`** runs the real page and `HttpChatApi` against the running stub
  stack: every answer outcome, a closed stream stored as `cancelled`, a reload, and
  another client ID getting 404. Each scenario has its own client ID (rate-limit
  burst 5). The tests fail, never skip, when the backend is down. They stop before
  sending anything unless `/ready` reports the stub. **It is local only, not part of
  `make test-int` or CI,** because it needs both images built and the index ingested.
  CI adoption is on the Phase 7 list.
- **Finding (commit 10).** The first run against the stack failed on every answer:
  the frontend modelled `flags.invalid_citations` and `flags.rewrite_fallback` as
  booleans, while the backend sends a count and a code or null. The unit fixtures
  shared the wrong types, so only the stack test could catch it (fixed in
  `fix(frontend)`, before commit 10).
- **Tolerant reader (after the Phase 4 report).** The response models declare only the
  fields the page reads and ignore every other field, whatever its type. `done.flags`
  keeps only `uncited`, as a strict bool: missing, `"true"`, `1` or `null` is
  `bad_response`. A backend change to a field the page never shows cannot break it.
  The fields the page does read are still mirrored by hand, and the stack test is what
  checks them.

## Consequences

- The page is a thin layer over pure state functions; most behaviour is tested
  without a browser. What AppTest cannot prove (painting token by token, a real
  reload, the browser's renderer, third-party requests, layout) is in a manual
  checklist run with `LLM_STUB_TOKEN_DELAY_MS=150`, then a real-key pass.
- **Known limitations:**
  - A second click on an example question before the disabled render reaches the
    browser can start a second send.
  - A tab closed while the answer is priming is noticed only at the first token, so
    one LLM call is wasted.
  - After a reload, a failed or cancelled answer no longer says why it stopped.
  - Escapes can show as backslashes inside a fence indented 1 to 3 spaces.
  - The backend status is checked once per session. A failed send in a page that is
    already open shows "The chat service can't be reached right now." and the
    copyable question, but no Retry until the page is reloaded.
  - The UI integration test is not in CI yet.
- Changing a field the page reads needs the frontend model and the stack test
  updated together.
