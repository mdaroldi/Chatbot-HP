"""Behavior of the verified, cached corpus downloader (httpx MockTransport, tmp_path)."""

import asyncio
import hashlib
import logging
import pathlib
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from app.domain.ingestion import (
    ChecksumMismatchError,
    CorpusDocument,
    CorpusDownloadError,
    CorpusError,
    CorpusNotAllowedError,
)
from app.infra.corpus_http import HttpCorpusSource, create_corpus_http_client
from tests.unit.conftest import SettingsFactory

Handler = Callable[[httpx.Request], httpx.Response]

PDF = b"%PDF-1.7\nsynthetic body\n%%EOF\n"
URL = "https://docs.test/manuals/demo.pdf"


def document(body: bytes = PDF, url: str = URL, doc_id: str = "demo-guide") -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        title="Demo Guide",
        product="Demo Printer 100",
        url=url,
        sha256=hashlib.sha256(body).hexdigest(),
    )


class Recorder:
    """Transport handler that serves canned responses and remembers every request."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        assert self._responses, f"unexpected request to {request.url}"
        return self._responses.pop(0)


def source(
    handler: Handler,
    corpus_dir: pathlib.Path,
    *documents: CorpusDocument,
    max_bytes: int = 1024,
    timeout_seconds: float = 5.0,
) -> HttpCorpusSource:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    return HttpCorpusSource(
        client,
        documents or (document(),),
        corpus_dir=corpus_dir,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )


def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"Location": location})


def leftovers(corpus_dir: pathlib.Path) -> list[str]:
    return sorted(p.name for p in corpus_dir.iterdir()) if corpus_dir.exists() else []


async def chunks(*parts: bytes, delay: float = 0.0) -> AsyncIterator[bytes]:
    for part in parts:
        if delay:
            await asyncio.sleep(delay)
        yield part


# ------------------------------------------------------------------ happy paths


async def test_download_is_verified_and_stored_under_the_doc_id(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(httpx.Response(200, content=PDF))
    corpus_dir = tmp_path / "nested" / "corpus"

    path = await source(recorder, corpus_dir).fetch(document())

    assert path == corpus_dir / "demo-guide.pdf"
    assert path.read_bytes() == PDF
    assert recorder.urls == [URL]
    assert leftovers(corpus_dir) == ["demo-guide.pdf"]


async def test_streamed_body_without_length_is_accepted(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(httpx.Response(200, content=chunks(PDF[:5], PDF[5:])))

    path = await source(recorder, tmp_path).fetch(document())

    assert path.read_bytes() == PDF


async def test_valid_cache_is_used_without_network(tmp_path: pathlib.Path) -> None:
    (tmp_path / "demo-guide.pdf").write_bytes(PDF)
    recorder = Recorder()

    path = await source(recorder, tmp_path).fetch(document())

    assert path == tmp_path / "demo-guide.pdf"
    assert recorder.urls == []


async def test_stale_cache_is_replaced_by_a_verified_download(tmp_path: pathlib.Path) -> None:
    (tmp_path / "demo-guide.pdf").write_bytes(b"%PDF-1.7 older edition")
    recorder = Recorder(httpx.Response(200, content=PDF))

    path = await source(recorder, tmp_path).fetch(document())

    assert path.read_bytes() == PDF
    assert recorder.urls == [URL]


async def test_discarding_a_stale_cache_logs_one_warning_with_the_document_id(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "demo-guide.pdf").write_bytes(b"%PDF-1.7 older edition")
    caplog.set_level(logging.WARNING, logger="app.infra.corpus_http")

    await source(Recorder(httpx.Response(200, content=PDF)), tmp_path).fetch(document())

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "cached corpus file does not match its pin; discarding it"
    assert record.__dict__["doc_id"] == "demo-guide"


async def test_a_valid_cache_logs_nothing(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "demo-guide.pdf").write_bytes(PDF)
    caplog.set_level(logging.DEBUG, logger="app.infra.corpus_http")

    await source(Recorder(), tmp_path).fetch(document())

    assert caplog.records == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_same_host_https_redirects_are_followed(tmp_path: pathlib.Path, status: int) -> None:
    recorder = Recorder(
        redirect("/mirror/demo.pdf", status),
        redirect("https://docs.test/final/demo.pdf"),
        httpx.Response(200, content=PDF),
    )

    path = await source(recorder, tmp_path).fetch(document())

    assert path.read_bytes() == PDF
    assert recorder.urls == [
        URL,
        "https://docs.test/mirror/demo.pdf",
        "https://docs.test/final/demo.pdf",
    ]


# ------------------------------------------------------------------ allowlist


@pytest.mark.parametrize(
    "requested",
    [
        document(doc_id="other-guide"),
        document(url="https://docs.test/manuals/other.pdf"),
        document(body=b"%PDF-1.7 another pin"),
    ],
)
async def test_documents_outside_the_allowlist_are_refused_without_network(
    tmp_path: pathlib.Path, requested: CorpusDocument
) -> None:
    recorder = Recorder()

    with pytest.raises(CorpusNotAllowedError, match=requested.doc_id):
        await source(recorder, tmp_path).fetch(requested)

    assert recorder.urls == []
    assert leftovers(tmp_path) == []


# ------------------------------------------------------------------ integrity


async def test_checksum_mismatch_is_distinct_and_leaves_no_file(tmp_path: pathlib.Path) -> None:
    served = b"%PDF-1.7 tampered body"
    recorder = Recorder(httpx.Response(200, content=served))
    expected = document()

    with pytest.raises(ChecksumMismatchError) as caught:
        await source(recorder, tmp_path).fetch(expected)

    message = str(caught.value)
    assert "demo-guide" in message
    assert expected.sha256 in message
    assert hashlib.sha256(served).hexdigest() in message
    assert not isinstance(caught.value, CorpusDownloadError)
    assert isinstance(caught.value, CorpusError)
    assert leftovers(tmp_path) == []


async def test_stale_cache_is_kept_out_when_the_download_mismatches(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "demo-guide.pdf").write_bytes(b"%PDF-1.7 older edition")
    recorder = Recorder(httpx.Response(200, content=b"%PDF-1.7 tampered body"))

    with pytest.raises(ChecksumMismatchError):
        await source(recorder, tmp_path).fetch(document())

    assert leftovers(tmp_path) == []


async def test_body_that_is_not_a_pdf_is_a_download_error(tmp_path: pathlib.Path) -> None:
    """Pinned to the HTML's own digest, so only the magic check can reject it."""
    html = b"<html>maintenance</html>"
    recorder = Recorder(httpx.Response(200, content=html))
    pinned = document(body=html)

    with pytest.raises(CorpusDownloadError, match="not a PDF"):
        await source(recorder, tmp_path, pinned).fetch(pinned)

    assert leftovers(tmp_path) == []


# ------------------------------------------------------------------ download failures


@pytest.mark.parametrize("status", [204, 304, 404, 500, 503])
async def test_non_200_status_is_a_download_error(tmp_path: pathlib.Path, status: int) -> None:
    recorder = Recorder(httpx.Response(status))

    with pytest.raises(CorpusDownloadError, match=f"demo-guide: HTTP {status}"):
        await source(recorder, tmp_path).fetch(document())

    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("x")],
)
async def test_transport_errors_are_download_errors(
    tmp_path: pathlib.Path, error: httpx.HTTPError
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(CorpusDownloadError, match=type(error).__name__) as caught:
        await source(handler, tmp_path).fetch(document())

    assert caught.value.__cause__ is error


async def test_declared_length_over_the_cap_is_refused_before_reading(
    tmp_path: pathlib.Path,
) -> None:
    class NeverRead:
        def __aiter__(self) -> "NeverRead":
            return self

        async def __anext__(self) -> bytes:
            raise AssertionError("body must not be read")

    response = httpx.Response(200, headers={"Content-Length": "1025"}, content=NeverRead())

    with pytest.raises(CorpusDownloadError, match="exceeds 1024 bytes"):
        await source(Recorder(response), tmp_path).fetch(document())


async def test_streamed_body_over_the_cap_is_refused(tmp_path: pathlib.Path) -> None:
    body = PDF + b"x" * (64 - len(PDF) + 1)
    recorder = Recorder(httpx.Response(200, content=chunks(body[:32], body[32:])))

    with pytest.raises(CorpusDownloadError, match="exceeds 64 bytes"):
        await source(recorder, tmp_path, document(body=body), max_bytes=64).fetch(
            document(body=body)
        )

    assert leftovers(tmp_path) == []


async def test_body_exactly_at_the_cap_is_accepted(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(httpx.Response(200, content=chunks(PDF)))

    path = await source(recorder, tmp_path, max_bytes=len(PDF)).fetch(document())

    assert path.read_bytes() == PDF


async def test_slow_download_hits_the_total_deadline(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(httpx.Response(200, content=chunks(PDF, PDF, delay=0.2)))

    with pytest.raises(CorpusDownloadError, match=r"timed out after 0\.05 s"):
        await source(recorder, tmp_path, timeout_seconds=0.05).fetch(document())

    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("location", "reason"),
    [
        ("https://elsewhere.test/demo.pdf", "different host"),
        ("http://docs.test/demo.pdf", "not https"),
        ("https://docs.test.elsewhere.test/demo.pdf", "different host"),
    ],
)
async def test_redirects_off_host_or_off_https_are_refused(
    tmp_path: pathlib.Path, location: str, reason: str
) -> None:
    recorder = Recorder(redirect(location))

    with pytest.raises(CorpusDownloadError, match=reason):
        await source(recorder, tmp_path).fetch(document())

    assert recorder.urls == [URL]


async def test_redirect_without_location_is_refused(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(httpx.Response(302))

    with pytest.raises(CorpusDownloadError, match="without Location"):
        await source(recorder, tmp_path).fetch(document())


async def test_redirect_chains_are_capped(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(*(redirect(f"/hop/{n}.pdf") for n in range(4)))

    with pytest.raises(CorpusDownloadError, match="too many redirects"):
        await source(recorder, tmp_path).fetch(document())

    assert len(recorder.urls) == 4


# ------------------------------------------------------------------ client factory


def test_client_has_timeouts_from_settings_and_never_follows_redirects(
    make_settings: SettingsFactory,
) -> None:
    client = create_corpus_http_client(make_settings(corpus_download_timeout_seconds=7.5))

    assert client.timeout == httpx.Timeout(7.5)
    assert client.follow_redirects is False


def test_source_is_built_from_settings(
    make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(corpus_dir=tmp_path, corpus_max_bytes=2048)
    client = create_corpus_http_client(settings)

    built = HttpCorpusSource.from_settings(client, (document(),), settings)

    assert built.corpus_dir == tmp_path
    assert built.max_bytes == 2048
    assert built.timeout_seconds == settings.corpus_download_timeout_seconds
