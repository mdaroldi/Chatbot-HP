"""Static guarantees for the Streamlit app: privacy, safety, no stale status."""

import ast
import pathlib
import tomllib
from collections.abc import Iterator

import pytest
from streamlit import config as streamlit_config
from streamlit.proto.NewSession_pb2 import Config
from streamlit.runtime.app_session import _get_toolbar_mode
from streamlit.testing.v1 import AppTest

FRONTEND = pathlib.Path(__file__).resolve().parents[1]
CONFIG = FRONTEND / ".streamlit" / "config.toml"
SOURCES = [FRONTEND / "app.py", *sorted((FRONTEND / "hpdocs_ui").rglob("*.py"))]
# The Streamlit layer: the only code allowed to import streamlit.
UI_SOURCES = [FRONTEND / "app.py", *sorted((FRONTEND / "hpdocs_ui" / "ui").rglob("*.py"))]

# Streamlit calls whose first argument (or label/body) is rendered as Markdown.
RENDERING = frozenset(
    {
        "markdown", "write", "caption", "info", "warning", "error", "success", "button",
        "expander", "badge", "title", "header", "subheader", "toast", "popover", "tabs",
        "checkbox", "toggle", "radio", "selectbox", "multiselect", "text_input",
        "text_area", "chat_input", "download_button", "link_button", "page_link",
        "form_submit_button", "metric", "status", "pills", "segmented_control", "help",
        "chat_message",
    }
)  # fmt: skip
# Calls that return Markdown-safe text: the sanitizer, the escaper, fixed wording.
SAFE_HELPERS = frozenset({"sanitize_markdown", "escape_markdown", "conversation_label"})
# Tables of fixed wording, indexed by a code or state.
WORDING_TABLES = frozenset(
    {"NOTICE_TEXT", "ERROR_TEXT", "REFUSAL_TEXT", "PRIVACY_TEXT", "EXAMPLE_QUESTIONS"}
)
# Module constants holding one fixed string each (hpdocs_ui.chat_view).
FIXED_TEXT = frozenset({"PRIMING_TEXT", "WARNING_BADGE", "LINK_TEXT", "EXAMPLES_INTRO"})
# The one function allowed to call markdown(): it renders answers, always sanitized.
ANSWER_RENDERER = ("chat.py", "render_answer_text")
# Reviewed exceptions: attributes that only ever hold fixed wording.
EXCEPTIONS = {
    "view.headline": "StatusView.headline, fixed headlines from view.describe()",
    "error.text": "ErrorView.text, from errors.error_text()",
}


def load_config() -> dict[str, dict[str, object]]:
    assert CONFIG.is_file(), f"missing {CONFIG}"
    return tomllib.loads(CONFIG.read_text())


def test_usage_stats_are_off_and_server_is_headless() -> None:
    config = load_config()

    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["headless"] is True


@pytest.fixture
def loaded_by_streamlit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The project config as Streamlit itself loads it ($CWD/.streamlit/config.toml),
    so a value the pinned Streamlit does not know shows up as a failure here.
    """
    monkeypatch.chdir(FRONTEND)
    streamlit_config.get_config_options(force_reparse=True)
    yield
    monkeypatch.undo()
    streamlit_config.get_config_options(force_reparse=True)


def test_toolbar_hides_deploy_and_the_developer_menu() -> None:
    assert load_config()["client"]["toolbarMode"] == "minimal"


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_pinned_streamlit_accepts_the_minimal_toolbar() -> None:
    # Streamlit's own mapping from the option to the browser message; it raises on
    # a value this version does not know.
    assert _get_toolbar_mode() == Config.ToolbarMode.MINIMAL


def test_error_details_are_hidden_from_the_browser() -> None:
    assert load_config()["client"]["showErrorDetails"] == "none"


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_an_uncaught_error_shows_no_message_type_or_traceback_in_the_page() -> None:
    app = AppTest.from_string("raise RuntimeError('detail-canary')").run()

    shown = app.exception[0]
    assert "detail-canary" not in shown.message
    assert shown.stack_trace == []
    assert "RuntimeError" not in shown.type


@pytest.mark.usefixtures("loaded_by_streamlit")
def test_pinned_streamlit_loads_usage_stats_off() -> None:
    assert streamlit_config.get_option("browser.gatherUsageStats") is False


def test_xsrf_and_cors_protection_keep_their_secure_defaults() -> None:
    server = load_config()["server"]

    assert "enableXsrfProtection" not in server
    assert "enableCORS" not in server


def test_sources_never_render_raw_html() -> None:
    assert len(SOURCES) >= 3
    for source in SOURCES:
        assert "unsafe_allow_html" not in source.read_text(), source.name


def test_status_results_are_never_cached() -> None:
    for source in SOURCES:
        text = source.read_text()
        assert "cache_data" not in text, source.name
        assert "cache_resource" not in text, source.name


def _rendered_argument(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg in {"label", "body"}:
            return keyword.value
    return call.args[0] if call.args else None


def _is_safe(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in SAFE_HELPERS
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id in WORDING_TABLES
    if isinstance(node, ast.Name):
        return node.id in FIXED_TEXT
    return ast.unparse(node) in EXCEPTIONS


def rendering_violations(source: str) -> list[str]:
    """Each Markdown-rendering call whose text is not a literal, a safe helper's
    result, fixed wording or a reviewed exception; and each rendering function used
    as a value (a dispatch table would hide its calls from this check).
    """
    tree = ast.parse(source)
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in RENDERING and id(node) not in called:
            found.append(f"uncalled {ast.unparse(node)}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in RENDERING:
                continue
            argument = _rendered_argument(node)
            if argument is not None and not _is_safe(argument):
                found.append(ast.unparse(node))
    return found


@pytest.mark.parametrize(
    "snippet",
    [
        "st.markdown(text)",
        "st.sidebar.button(f'{title}')",
        "col.info(message.text)",
        "st.button(label=title)",
        "RENDER = {'ok': st.success}",
        "st.error(OTHER_TABLE[code])",
        "slot.caption(OTHER_TEXT)",
    ],
)
def test_the_rendering_check_flags_unsafe_text(snippet: str) -> None:
    assert rendering_violations(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "st.markdown(sanitize_markdown(text))",
        "st.button(escape_markdown(title), key=k)",
        "st.sidebar.info(NOTICE_TEXT[notice])",
        "st.button('Delete', key=k)",
        "st.error(error.text)",
        "st.text(anything)",
        "slot.caption(PRIMING_TEXT)",
        "st.badge(WARNING_BADGE, color='orange')",
        "st.caption(PRIVACY_TEXT[provider])",
        "st.button(EXAMPLE_QUESTIONS[index], key=k)",
        "st.caption(LINK_TEXT)",
    ],
)
def test_the_rendering_check_accepts_safe_text(snippet: str) -> None:
    assert rendering_violations(snippet) == []


def test_every_markdown_rendering_call_gets_safe_text() -> None:
    assert len(UI_SOURCES) >= 2
    for source in UI_SOURCES:
        assert rendering_violations(source.read_text()) == [], source.name


def markdown_callers(source: str) -> list[str]:
    """The function enclosing each markdown() call ("<module>" at the top level)."""
    callers = []

    def visit(node: ast.AST, enclosing: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call) and getattr(child.func, "attr", None) == "markdown":
                callers.append(enclosing)
            visit(child, enclosing)

    visit(ast.parse(source), "<module>")
    return callers


def test_the_markdown_caller_check_names_the_enclosing_function() -> None:
    source = (
        "st.markdown('a')\ndef f():\n    slot.markdown(x)\n    def g():\n        st.markdown(y)\n"
    )

    assert markdown_callers(source) == ["<module>", "f", "g"]


def test_markdown_is_called_only_where_answers_are_rendered() -> None:
    callers = [
        (source.name, caller)
        for source in UI_SOURCES
        for caller in markdown_callers(source.read_text())
    ]

    assert callers == [ANSWER_RENDERER]


def test_only_the_streamlit_layer_imports_streamlit() -> None:
    for source in SOURCES:
        if source in UI_SOURCES:
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not [n for n in names if n.split(".")[0] == "streamlit"], source.name


def test_the_sidebar_starts_expanded_so_its_notices_are_seen() -> None:
    tree = ast.parse((FRONTEND / "app.py").read_text())
    configs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "set_page_config"
    ]

    (config,) = configs
    options = {k.arg: ast.literal_eval(k.value) for k in config.keywords}
    assert options["initial_sidebar_state"] == "expanded"
