"""Verified, cached download of the allowlisted corpus documents over HTTPS."""

import asyncio
import hashlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.core.settings import Settings
from app.domain.ingestion import (
    ChecksumMismatchError,
    CorpusDocument,
    CorpusDownloadError,
    CorpusNotAllowedError,
)

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3
_PDF_MAGIC = b"%PDF-"


def create_corpus_http_client(settings: Settings) -> httpx.AsyncClient:
    """Redirects are never followed by httpx: the source checks every hop itself."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.corpus_download_timeout_seconds),
        follow_redirects=False,
    )


def _sha256_of_file(path: Path) -> str | None:
    try:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomically(path: Path, data: bytes) -> None:
    """Readers see either no file or the complete verified one, never a partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_bytes(data)
    os.replace(partial, path)


class HttpCorpusSource:
    """CorpusSource that serves verified files from `corpus_dir`, downloading on a miss.

    Only documents from the manifest are fetched, so a caller can never widen the
    corpus. A file is stored only after its SHA-256 matches the pin, which keeps the
    cache trustworthy for offline runs.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        documents: Sequence[CorpusDocument],
        *,
        corpus_dir: Path,
        max_bytes: int,
        timeout_seconds: float,
    ) -> None:
        self._client = client
        self._allowed = frozenset(documents)
        self.corpus_dir = corpus_dir
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(
        cls, client: httpx.AsyncClient, documents: Sequence[CorpusDocument], settings: Settings
    ) -> "HttpCorpusSource":
        return cls(
            client,
            documents,
            corpus_dir=settings.corpus_dir,
            max_bytes=settings.corpus_max_bytes,
            timeout_seconds=settings.corpus_download_timeout_seconds,
        )

    async def fetch(self, document: CorpusDocument) -> Path:
        if document not in self._allowed:
            raise CorpusNotAllowedError(f"{document.doc_id}: not in the corpus manifest")
        path = self.corpus_dir / f"{document.doc_id}.pdf"
        cached = await asyncio.to_thread(_sha256_of_file, path)
        if cached == document.sha256:
            return path
        if cached is not None:
            # A stale file (older pin or tampering) must not outlive a failed download.
            logger.warning(
                "cached corpus file does not match its pin; discarding it",
                extra={"doc_id": document.doc_id},
            )
            await asyncio.to_thread(path.unlink)
        data = await self._download(document)
        if not data.startswith(_PDF_MAGIC):
            raise CorpusDownloadError(f"{document.doc_id}: response is not a PDF")
        actual = await asyncio.to_thread(_sha256, data)
        if actual != document.sha256:
            raise ChecksumMismatchError(
                f"{document.doc_id}: SHA-256 mismatch (expected {document.sha256}, "
                f"got {actual}); the upstream file changed or was tampered with"
            )
        await asyncio.to_thread(_write_atomically, path, data)
        return path

    async def _download(self, document: CorpusDocument) -> bytes:
        """The total deadline also bounds slow-drip servers, which per-read timeouts miss."""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._follow_redirects(document)
        except TimeoutError as exc:
            raise CorpusDownloadError(
                f"{document.doc_id}: download timed out after {self.timeout_seconds:g} s"
            ) from exc
        except httpx.HTTPError as exc:
            raise CorpusDownloadError(f"{document.doc_id}: {type(exc).__name__}") from exc

    async def _follow_redirects(self, document: CorpusDocument) -> bytes:
        origin = url = httpx.URL(document.url)
        for _ in range(_MAX_REDIRECTS + 1):
            async with self._client.stream("GET", url) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    url = _redirect_target(document, origin, url, response)
                    continue
                if response.status_code != httpx.codes.OK:
                    raise CorpusDownloadError(f"{document.doc_id}: HTTP {response.status_code}")
                return await self._read_capped(document, response)
        raise CorpusDownloadError(f"{document.doc_id}: too many redirects")

    async def _read_capped(self, document: CorpusDocument, response: httpx.Response) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > self.max_bytes:
            raise CorpusDownloadError(f"{document.doc_id}: body exceeds {self.max_bytes} bytes")
        body = bytearray()
        async for part in response.aiter_bytes():
            body.extend(part)
            if len(body) > self.max_bytes:
                raise CorpusDownloadError(f"{document.doc_id}: body exceeds {self.max_bytes} bytes")
        return bytes(body)


def _redirect_target(
    document: CorpusDocument, origin: httpx.URL, current: httpx.URL, response: httpx.Response
) -> httpx.URL:
    """Only same-host HTTPS hops: a redirect must never move the corpus elsewhere."""
    location = response.headers.get("Location")
    if not location:
        raise CorpusDownloadError(f"{document.doc_id}: redirect without Location")
    target = current.join(location)
    if target.scheme != "https":
        raise CorpusDownloadError(f"{document.doc_id}: redirect target is not https")
    if (target.host, target.port) != (origin.host, origin.port):
        raise CorpusDownloadError(f"{document.doc_id}: redirect to a different host")
    return target
