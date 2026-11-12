"""Static guarantees of the Alembic setup (the migrations themselves run in integration)."""

import ast
import configparser
import pathlib
import re

BACKEND = pathlib.Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND / "alembic.ini"
ENV_PY = BACKEND / "alembic" / "env.py"
VERSIONS = BACKEND / "alembic" / "versions"
LOGGING_SECTIONS = {"loggers", "handlers", "formatters"}


def read_ini() -> configparser.ConfigParser:
    """Parse alembic.ini, failing if it is missing (configparser would read nothing)."""
    assert ALEMBIC_INI.is_file(), f"missing {ALEMBIC_INI}"
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ALEMBIC_INI)
    assert parser.has_section("alembic")
    return parser


def test_alembic_folder_is_not_a_python_package() -> None:
    # backend/alembic/ shares the library's name. As a plain folder (no __init__.py)
    # the installed `alembic` package always wins on import; as a package it would
    # shadow the library whenever backend/ is on sys.path (prepend_sys_path does that).
    assert not (BACKEND / "alembic" / "__init__.py").exists()
    assert not (VERSIONS / "__init__.py").exists()


def test_ini_has_no_database_url() -> None:
    ini = read_ini()

    assert not any(ini.has_option(section, "sqlalchemy.url") for section in ini.sections())
    assert "sqlalchemy.url" not in ALEMBIC_INI.read_text()


def test_ini_has_no_logging_configuration() -> None:
    sections = set(read_ini().sections())

    assert not sections & LOGGING_SECTIONS
    assert not any(name.startswith(("logger_", "handler_", "formatter_")) for name in sections)


def test_script_location_does_not_depend_on_working_directory() -> None:
    assert read_ini().get("alembic", "script_location") == "%(here)s/alembic"
    assert read_ini().get("alembic", "prepend_sys_path") == "%(here)s"


def test_file_template_is_date_prefixed() -> None:
    template = read_ini().get("alembic", "file_template")

    assert template.startswith("%%(year)d%%(month).2d%%(day).2d_")
    assert template.endswith("%%(rev)s_%%(slug)s")


def test_env_never_configures_logging_or_url_itself() -> None:
    source = ENV_PY.read_text()

    assert "fileConfig" not in source
    assert "set_main_option" not in source
    assert "sqlalchemy.url" not in source
    assert "configure_logging(" in source
    assert "create_engine(" in source


def test_env_compares_column_types() -> None:
    assert "compare_type=True" in ENV_PY.read_text()


def migration_files() -> list[pathlib.Path]:
    return sorted(VERSIONS.glob("*.py"))


def test_only_the_baseline_migration_exists_and_its_name_is_date_prefixed() -> None:
    names = [path.name for path in migration_files()]

    assert len(names) == 1
    assert re.fullmatch(r"\d{8}_\d{4}_0001_baseline\.py", names[0])


def test_baseline_migration_is_empty() -> None:
    tree = ast.parse(migration_files()[0].read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    for name in ("upgrade", "downgrade"):
        body = [stmt for stmt in functions[name].body if not isinstance(stmt, ast.Expr)]
        assert [type(stmt) for stmt in body] == [ast.Pass], f"{name}() must stay empty"
