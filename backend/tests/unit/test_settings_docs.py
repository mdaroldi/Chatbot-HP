"""The README configuration table is generated from Settings and must stay in sync."""

import pathlib

from app.core.settings import Settings
from app.core.settings_docs import README_END, README_START, settings_markdown_table

README = pathlib.Path(__file__).resolve().parents[3] / "README.md"


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
