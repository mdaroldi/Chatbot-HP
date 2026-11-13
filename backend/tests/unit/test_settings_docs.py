"""The README configuration table is generated from Settings and must stay in sync."""

import pathlib

from app.core.settings import Settings
from app.core.settings_docs import README_END, README_START, settings_markdown_table
from app.infra.model_files import load_pins

ROOT = pathlib.Path(__file__).resolve().parents[3]
README = ROOT / "README.md"
ADR_0009 = ROOT / "docs" / "adr" / "0009-retrieval-strategy-and-evaluation.md"


def table_rows() -> list[list[str]]:
    lines = settings_markdown_table().strip().splitlines()
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def test_every_setting_has_a_description() -> None:
    missing = [name for name, field in Settings.model_fields.items() if not field.description]

    assert missing == []


def test_table_has_header_and_one_row_per_setting_in_declaration_order() -> None:
    lines = settings_markdown_table().strip().splitlines()

    assert lines[0] == "| Variable | Default | Description |"
    assert lines[1] == "|---|---|---|"
    assert [row[0] for row in table_rows()] == [
        f"`{name.upper()}`" for name in Settings.model_fields
    ]


def test_defaults_are_rendered_as_env_values() -> None:
    defaults = {row[0]: row[1] for row in table_rows()}

    assert defaults["`APP_ENV`"] == "`development`"
    assert defaults["`LOG_LEVEL`"] == "`INFO`"
    assert defaults["`LOG_INCLUDE_EXCEPTION_MESSAGES`"] == "`false`"
    assert defaults["`CORS_ALLOWED_ORIGINS`"] == "`http://localhost:8501`"
    assert defaults["`POSTGRES_PORT`"] == "`5432`"
    assert defaults["`QDRANT_URL`"] == "`http://qdrant:6333/`"
    assert defaults["`READINESS_TIMEOUT_SECONDS`"] == "`2.0`"


def test_empty_defaults_are_named_not_rendered_as_code() -> None:
    defaults = {row[0]: row[1] for row in table_rows()}

    assert defaults["`RETRIEVAL_RELEVANCE_THRESHOLD`"] == "`-6.54`"
    assert defaults["`RETRIEVAL_DENSE_QUERY_PREFIX`"] == "*empty*"
    assert defaults["`ANTHROPIC_API_KEY`"] == "*unset*"


def test_the_threshold_names_the_reranker_pin_it_was_calibrated_on() -> None:
    """The threshold belongs to one reranker revision: moving the pin must fail here
    until the calibration is redone and its record moved with it (ADR 0009)."""
    model = Settings().reranker_model
    revision = load_pins()[model].revision
    rows = {row[0]: row[2] for row in table_rows()}

    assert f"{model} @ {revision[:12]}" in rows["`RETRIEVAL_RELEVANCE_THRESHOLD`"]
    assert f"`{model}` @ `{revision}`, threshold `-6.54`" in ADR_0009.read_text(encoding="utf-8")


def test_required_secret_has_no_default_shown() -> None:
    defaults = {row[0]: row[1] for row in table_rows()}

    assert defaults["`POSTGRES_PASSWORD`"] == "*required*"


def test_readme_contains_the_generated_table() -> None:
    """Regenerate with `make docs-config` and paste between the README markers."""
    text = README.read_text()
    assert README_START.startswith("<!--")
    assert README_END.startswith("<!--")
    assert text.count(README_START) == 1
    assert text.count(README_END) == 1
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)

    assert text[start:end].strip() == settings_markdown_table().strip()
