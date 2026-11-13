"""The refusal marker detector (pure): one hold-back rule over streamed text.

The model refuses by replying with the marker alone (layer 2), and it streams in
arbitrary pieces, so the detector never emits a character that could still be part of
a marker. A marker at the start of the answer is a refusal; one after visible text is
removed and flagged, and the answer kept.
"""

from app.domain.chat import REFUSAL_MARKER


class MarkerDetector:
    def __init__(self, marker: str = REFUSAL_MARKER) -> None:
        self._marker = marker
        self._pending = ""
        self._started = False
        self._refused = False
        self._late_marker = False

    @property
    def refused(self) -> bool:
        """The answer opened with the marker: the caller cancels the stream."""
        return self._refused

    @property
    def late_marker(self) -> bool:
        return self._late_marker

    def feed(self, delta: str) -> str:
        """Take the next text piece and return what is now safe to show."""
        if self._refused:
            raise RuntimeError("the answer was refused")
        self._pending += delta
        if not self._started and not self._start():
            return ""
        while self._marker in self._pending:
            self._pending = self._pending.replace(self._marker, "")
            self._late_marker = True
        cut = self._hold_from()
        visible, self._pending = self._pending[:cut], self._pending[cut:]
        return visible

    def finish(self) -> str:
        """The end of the stream: whatever was held is not a marker after all."""
        rest = self._pending if self._started else self._pending.lstrip()
        self._pending = ""
        return rest

    def _start(self) -> bool:
        """Leading whitespace waits with the candidate; True once the answer started."""
        candidate = self._pending.lstrip()
        if candidate.startswith(self._marker):
            self._refused = True
            self._pending = ""
            return False
        if self._marker.startswith(candidate):
            return False
        self._started = True
        self._pending = candidate
        return True

    def _hold_from(self) -> int:
        """The first index from which the pending text could still become the marker."""
        pending = self._pending
        return next(
            (i for i in range(len(pending)) if self._marker.startswith(pending[i:])),
            len(pending),
        )
