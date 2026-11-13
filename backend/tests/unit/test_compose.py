"""Hardening invariants of docker-compose.yml.

Trivy's config scanner has no docker-compose support, so these rules are pinned
here instead of being checked only by hand.
"""

import pathlib
import re
import tomllib
from typing import Any

import pytest
import yaml

from app.core.settings import Settings

REPO = pathlib.Path(__file__).resolve().parents[3]
COMPOSE = REPO / "docker-compose.yml"
TEST_COMPOSE = REPO / "docker-compose.test.yml"
MAKEFILE = REPO / "Makefile"

SERVICES = {"backend", "frontend", "postgres", "qdrant", "migrate", "ingest"}
APP_SERVICES = {"backend", "frontend"}
DATA_SERVICES = {"postgres", "qdrant"}
ONE_SHOT_SERVICES = {"migrate", "ingest"}
BACKEND_IMAGE_SERVICES = {"backend", *ONE_SHOT_SERVICES}
BIND = "${BIND_ADDRESS:-127.0.0.1}"
BACKEND_IMAGE = "hpdocs-backend:local"
FRONTEND_IMAGE = "hpdocs-frontend:local"
APP_USER = "10001:10001"
CORPUS_CACHE = "/var/cache/hpdocs/corpus"
# Models are baked into the image (ADR 0008), read-only, at the same path in every service.
MODEL_STORE = "/opt/hpdocs/models"

# Any: parsed YAML is untyped by nature; each test asserts the shape it relies on.
Service = dict[str, Any]


def load(path: pathlib.Path) -> dict[str, Any]:
    assert path.is_file(), f"missing {path}"
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return load(COMPOSE)


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Service]:
    found: dict[str, Service] = compose["services"]
    return found


def test_stack_has_exactly_the_six_services(services: dict[str, Service]) -> None:
    assert set(services) == SERVICES


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_every_service_is_hardened(services: dict[str, Service], name: str) -> None:
    service = services[name]

    assert service["cap_drop"] == ["ALL"]
    assert "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["read_only"] is True
    assert "/tmp" in service["tmpfs"]  # noqa: S108 - checks a tmpfs mount, uses no temp file
    assert service.get("privileged", False) is False
    assert "network_mode" not in service
    assert "pid" not in service


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_no_service_mounts_the_docker_socket(services: dict[str, Service], name: str) -> None:
    volumes = [str(volume) for volume in services[name].get("volumes", [])]

    assert not any("docker.sock" in volume for volume in volumes)


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_every_service_runs_as_non_root(services: dict[str, Service], name: str) -> None:
    """App images set USER 10001; postgres runs as uid 70; qdrant's image is unprivileged."""
    user = services[name].get("user")

    assert user is None or not str(user).startswith(("0", "root"))
    if name == "postgres":
        assert user == "70:70"
    if name == "qdrant":
        assert services[name]["image"].startswith("qdrant/qdrant:")
        assert "-unprivileged@" in services[name]["image"]
    if name in BACKEND_IMAGE_SERVICES:
        assert user == APP_USER


def test_only_api_and_frontend_publish_ports_on_the_bind_address(
    services: dict[str, Service],
) -> None:
    published = {name for name, service in services.items() if service.get("ports")}

    assert published == APP_SERVICES
    assert services["backend"]["ports"] == [f"{BIND}:${{API_PORT:-8000}}:8000"]
    assert services["frontend"]["ports"] == [f"{BIND}:${{FRONTEND_PORT:-8501}}:8501"]
    assert {name for name, service in services.items() if "expose" in service} == set()


def test_data_network_is_internal_and_only_backend_and_ingest_bridge_it(
    compose: dict[str, Any], services: dict[str, Service]
) -> None:
    """ingest also bridges it: it downloads the PDFs and models, then writes the index."""
    assert compose["networks"]["data"]["internal"] is True
    assert compose["networks"]["edge"].get("internal") is not True
    assert set(services["backend"]["networks"]) == {"edge", "data"}
    assert set(services["ingest"]["networks"]) == {"edge", "data"}
    assert set(services["frontend"]["networks"]) == {"edge"}
    for name in {*DATA_SERVICES, "migrate"}:
        assert set(services[name]["networks"]) == {"data"}


@pytest.mark.parametrize("name", sorted(DATA_SERVICES))
def test_third_party_images_are_pinned_by_digest(services: dict[str, Service], name: str) -> None:
    assert "@sha256:" in services[name]["image"]


def test_postgres_password_is_required_and_never_defaulted(
    services: dict[str, Service],
) -> None:
    for name in ("backend", "postgres", "migrate", "ingest"):
        required = "${POSTGRES_PASSWORD:?set it in .env}"
        assert services[name]["environment"]["POSTGRES_PASSWORD"] == required
    assert "POSTGRES_PASSWORD" not in services["frontend"]["environment"]
    assert "env_file" not in services["frontend"]


POSTGRES_MAJOR = "17"
DEPENDABOT = REPO / ".github" / "dependabot.yml"


def test_postgres_stays_on_one_major_version_in_both_compose_files(
    services: dict[str, Service],
) -> None:
    """Postgres 18 refuses the volume at /var/lib/postgresql/data, even when it is empty."""
    stack = services["postgres"]["image"]
    test = load(TEST_COMPOSE)["services"]["postgres"]["image"]

    assert stack.startswith(f"postgres:{POSTGRES_MAJOR}.")
    assert test == stack
    assert services["postgres"]["volumes"] == ["postgres-data:/var/lib/postgresql/data"]


def dependabot_ignores(ecosystem: str) -> list[dict[str, Any]]:
    updates = [u for u in load(DEPENDABOT)["updates"] if u["package-ecosystem"] == ecosystem]
    assert len(updates) == 1, ecosystem
    ignores: list[dict[str, Any]] = updates[0].get("ignore", [])
    return ignores


def major_ignore(name: str) -> dict[str, Any]:
    return {"dependency-name": name, "update-types": ["version-update:semver-major"]}


def test_dependabot_never_proposes_a_postgres_major_upgrade() -> None:
    """A major upgrade needs a new mount point and pg_upgrade; it is a decision, not a bump."""
    assert major_ignore("postgres") in dependabot_ignores("docker-compose")


def test_dependabot_skips_huggingface_hub_majors_while_the_backend_caps_them() -> None:
    """fastembed caps huggingface-hub below 2.0. Dependabot widens the backend's own cap
    anyway, the resolver rejects 2.x, and the whole uv update job fails every week."""
    backend = tomllib.loads(backend_file("pyproject.toml"))["project"]["dependencies"]
    hub = [spec for spec in backend if spec.startswith("huggingface-hub")]

    assert hub == ["huggingface-hub>=1.33.0,<2.0"]
    assert major_ignore("huggingface-hub") in dependabot_ignores("uv")


def test_qdrant_telemetry_is_disabled(services: dict[str, Service]) -> None:
    assert services["qdrant"]["environment"]["QDRANT__TELEMETRY_DISABLED"] == "true"


@pytest.mark.parametrize("name", sorted(BACKEND_IMAGE_SERVICES))
def test_backend_image_services_disable_library_telemetry(
    services: dict[str, Service], name: str
) -> None:
    environment: dict[str, str] = services[name]["environment"]

    assert {k: environment.get(k) for k in TELEMETRY_ENV} == TELEMETRY_ENV


def test_streamlit_usage_statistics_are_off_in_compose_and_in_the_image() -> None:
    """The image's config.toml and the compose variable both say no, so either holds alone."""
    environment = load(COMPOSE)["services"]["frontend"]["environment"]
    config = tomllib.loads((REPO / "frontend" / ".streamlit" / "config.toml").read_text())

    assert environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert config["browser"]["gatherUsageStats"] is False


def test_app_and_one_shot_services_use_an_init_process(services: dict[str, Service]) -> None:
    for name in APP_SERVICES | ONE_SHOT_SERVICES:
        assert services[name]["init"] is True


@pytest.mark.parametrize("name", sorted(ONE_SHOT_SERVICES))
def test_one_shot_services_run_once_without_the_api_healthcheck(
    services: dict[str, Service], name: str
) -> None:
    """The shared hardening restarts services; a finished job must stay finished."""
    service = services[name]

    assert service["restart"] == "no"
    assert service["healthcheck"] == {"disable": True}


def test_long_running_services_restart(services: dict[str, Service]) -> None:
    for name in APP_SERVICES | DATA_SERVICES:
        assert services[name]["restart"] == "unless-stopped"


def test_only_the_backend_service_builds_the_backend_image(services: dict[str, Service]) -> None:
    """Three services building one tag race on export with a cold cache ("already exists")."""
    builders = {name for name, service in services.items() if "build" in service}

    assert builders == APP_SERVICES
    assert services["backend"]["build"] == "./backend"
    assert services["backend"]["image"] == BACKEND_IMAGE
    assert services["frontend"]["build"] == "./frontend"
    assert services["frontend"]["image"] == FRONTEND_IMAGE


@pytest.mark.parametrize("name", sorted(ONE_SHOT_SERVICES))
def test_one_shot_services_run_the_backend_image_and_never_pull_it(
    services: dict[str, Service], name: str
) -> None:
    """The tag exists only locally; a pull would fail or fetch a stranger's image."""
    assert services[name]["image"] == BACKEND_IMAGE
    assert services[name]["pull_policy"] == "never"


def recipe(target: str) -> list[str]:
    lines = MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{target}:"))
    body = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        body.append(line.strip())
    return body


def test_make_ingest_builds_the_backend_image_once_before_the_forced_run() -> None:
    """ingest has no build section, so `run --build` would not rebuild the image."""
    assert recipe("ingest") == [
        "docker compose build backend",
        "docker compose run --rm ingest --force",
    ]


def test_make_eval_retrieval_runs_the_benchmark_in_the_backend_image() -> None:
    """In process against the running stack (no deps, no profiles: ADR 0007), as the
    caller's user so a Linux bind mount stays writable, with the commit it measures.
    """
    command = " ".join(line.removesuffix("\\").strip() for line in recipe("eval-retrieval"))

    assert command.startswith("docker compose run --rm --no-deps ")
    assert '--user "$$(id -u):$$(id -g)"' in command
    assert '--volume "$(CURDIR)/eval:/eval"' in command
    assert '--env GIT_SHA="$$(git rev-parse HEAD)$$(git diff --quiet HEAD || echo -dirty)"' in (
        command
    )
    assert command.endswith("--entrypoint python backend -m app.evaluation $(EVAL_ARGS)")
    assert re.search(r"^EVAL_ARGS \?= run$", MAKEFILE.read_text(), re.MULTILINE)


def test_migrate_applies_migrations_and_ingest_builds_only_when_needed(
    services: dict[str, Service],
) -> None:
    assert services["migrate"]["entrypoint"] == ["alembic", "-c", "/app/alembic.ini"]
    assert services["migrate"]["command"] == ["upgrade", "head"]
    assert services["ingest"]["entrypoint"] == ["python", "-m", "app.ingestion"]
    assert services["ingest"]["command"] == ["--if-needed"]
    assert "entrypoint" not in services["backend"]
    assert "command" not in services["backend"]


@pytest.mark.parametrize("name", sorted(DATA_SERVICES))
def test_data_services_have_healthchecks(services: dict[str, Service], name: str) -> None:
    assert services[name]["healthcheck"]["test"]


def conditions(service: Service) -> dict[str, str]:
    return {name: dep["condition"] for name, dep in service["depends_on"].items()}


def test_startup_order_runs_migrate_then_ingest_before_the_backend(
    services: dict[str, Service],
) -> None:
    """postgres, qdrant healthy -> migrate -> ingest -> backend -> frontend (ADR 0007)."""
    assert conditions(services["migrate"]) == {"postgres": "service_healthy"}
    assert conditions(services["ingest"]) == {
        "migrate": "service_completed_successfully",
        "qdrant": "service_healthy",
    }
    assert conditions(services["backend"]) == {
        "migrate": "service_completed_successfully",
        "ingest": "service_completed_successfully",
    }
    assert conditions(services["frontend"]) == {"backend": "service_healthy"}


# Settings ingestion reads, passed through from .env; the rest of its settings are
# fixed by the stack (hosts and cache paths) or unused by it (API-only settings).
INGEST_FROM_ENV_PREFIXES = ("postgres_", "qdrant_collection_", "index_", "embedding_", "chunk_")
INGEST_FROM_ENV_EXTRA = {"app_env", "log_level", "log_include_exception_messages"}
INGEST_FIXED = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "QDRANT_URL": "http://qdrant:6333",
    "CORPUS_DIR": CORPUS_CACHE,
    "EMBEDDING_CACHE_DIR": MODEL_STORE,
}
# Not a setting: huggingface-hub and hf-xet keep their own cache and logs under
# HF_HOME (default ~/.cache), which the read-only root and the home-less user block.
# Telemetry off by configuration on every container of the backend image: onnxruntime's
# Linux build carries Microsoft 1DS telemetry, started when its environment is created
# unless ORT_DISABLE_TELEMETRY is set; huggingface-hub adds environment details to its
# user agent and draws progress bars on stderr unless told not to.
# HF_HUB_OFFLINE keeps the hub library from ever reaching the network: the models are in
# the image, so no container downloads one (ADR 0008).
TELEMETRY_ENV = {
    "ORT_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_OFFLINE": "1",
}
INGEST_LIBRARY_ENV = {
    "HF_HOME": "/tmp/huggingface",  # noqa: S108 - a tmpfs path in compose
    **TELEMETRY_ENV,
}


def test_ingest_gets_every_setting_it_reads(services: dict[str, Service]) -> None:
    environment: dict[str, str] = services["ingest"]["environment"]
    fields = set(Settings.model_fields)
    from_env = {
        name.upper()
        for name in fields
        if name.startswith((*INGEST_FROM_ENV_PREFIXES, "corpus_")) or name in INGEST_FROM_ENV_EXTRA
    } - set(INGEST_FIXED)

    assert set(environment) == from_env | set(INGEST_FIXED) | set(INGEST_LIBRARY_ENV)
    assert {k: v for k, v in environment.items() if k in INGEST_FIXED} == INGEST_FIXED
    assert {k: environment[k] for k in INGEST_LIBRARY_ENV} == INGEST_LIBRARY_ENV
    assert [k for k in from_env if not str(environment[k]).startswith(f"${{{k}:")] == []
    assert {name.lower() for name in set(environment) - set(INGEST_LIBRARY_ENV)} <= fields


def test_migrate_gets_the_database_and_logging_settings(services: dict[str, Service]) -> None:
    environment: dict[str, str] = services["migrate"]["environment"]
    ingest: dict[str, str] = services["ingest"]["environment"]
    expected = {k for k in ingest if k.startswith("POSTGRES_")} | {
        "APP_ENV",
        "LOG_LEVEL",
        "LOG_INCLUDE_EXCEPTION_MESSAGES",
    }

    assert set(environment) == expected | set(TELEMETRY_ENV)
    assert {k: environment[k] for k in expected} == {k: ingest[k] for k in expected}


def test_ingest_caches_only_the_corpus_in_a_named_volume(
    compose: dict[str, Any], services: dict[str, Service]
) -> None:
    """The models come with the image; no volume can shadow or alter them."""
    assert services["ingest"]["volumes"] == [f"corpus-cache:{CORPUS_CACHE}"]
    assert "corpus-cache" in compose["volumes"]
    assert "model-cache" not in compose["volumes"]
    assert "volumes" not in services["migrate"]
    assert "volumes" not in services["backend"]


def test_data_lives_in_named_volumes(compose: dict[str, Any], services: dict[str, Service]) -> None:
    named = set(compose["volumes"])
    mounts = [
        str(volume).split(":", 1)[0]
        for name in DATA_SERVICES
        for volume in services[name]["volumes"]
    ]

    assert mounts
    assert set(mounts) <= named


def test_integration_compose_binds_only_to_loopback() -> None:
    for name, service in load(TEST_COMPOSE)["services"].items():
        for mapping in service.get("ports", []):
            assert str(mapping).startswith("127.0.0.1:"), name


# ------------------------------------------------------------------ built images

DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
NON_ROOT_NUMERIC = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


def dockerfile(services: dict[str, Service], name: str) -> str:
    context = REPO / str(services[name]["build"])
    return (context / "Dockerfile").read_text()


@pytest.mark.parametrize("name", sorted(APP_SERVICES))
def test_app_services_are_built_from_digest_pinned_base_images(
    services: dict[str, Service], name: str
) -> None:
    """Built images are tagged locally; their pinning lives in each Dockerfile."""
    text = dockerfile(services, name)
    image_args = dict(re.findall(r"^ARG (\w+_IMAGE)=(\S+)$", text, flags=re.MULTILINE))
    stages = re.findall(r"^FROM (\S+)", text, flags=re.MULTILINE)

    assert image_args
    assert all(DIGEST.search(reference) for reference in image_args.values()), image_args
    for stage in stages:
        arg = re.fullmatch(r"\$\{(\w+)\}", stage)
        assert arg is not None, f"{name}: FROM {stage} must use a pinned *_IMAGE argument"
        assert arg.group(1) in image_args


def backend_file(name: str) -> str:
    return (REPO / "backend" / name).read_text()


def test_backend_image_ships_the_corpus_and_model_pins_next_to_the_app() -> None:
    """corpus.py and model_files.py resolve both files as parents[2] of the module."""
    text = backend_file("Dockerfile")
    allowed = backend_file(".dockerignore").splitlines()

    assert re.search(r"^WORKDIR /app$", text, flags=re.MULTILINE)
    assert re.search(r"^COPY app \./app$", text, flags=re.MULTILINE)
    assert re.search(r"^COPY corpus\.toml models\.toml \./$", text, flags=re.MULTILINE)
    assert {"!corpus.toml", "!models.toml"} <= set(allowed)


def test_backend_image_owns_the_corpus_cache_mount_point() -> None:
    """A named volume copies its mount point's owner on first use; root would block 10001."""
    text = backend_file("Dockerfile")

    assert f"mkdir -p {CORPUS_CACHE}" in text
    assert f"chown {APP_USER} {CORPUS_CACHE}" in text
    assert "/var/cache/hpdocs/models" not in text


def test_backend_image_bakes_the_verified_models_read_only() -> None:
    """A models stage fills and verifies the store; the runtime copies it as root-owned
    files and reads it offline (ADR 0008).
    """
    text = backend_file("Dockerfile")

    assert re.search(r"^FROM \$\{PYTHON_IMAGE\} AS models$", text, flags=re.MULTILINE)
    assert "from app.infra.model_store import main" in text
    assert f"--target {MODEL_STORE}" in text
    assert re.search(rf"^COPY --from=models {MODEL_STORE} {MODEL_STORE}$", text, re.MULTILINE)
    assert f"EMBEDDING_CACHE_DIR={MODEL_STORE}" in text
    for name, value in TELEMETRY_ENV.items():
        assert f"{name}={value}" in text


def test_backend_entrypoint_only_starts_the_api() -> None:
    """Migrations moved to the one-shot migrate service (ADR 0007)."""
    script = backend_file("docker-entrypoint.sh")

    assert "alembic" not in script
    assert script.rstrip().splitlines()[-1].startswith("exec uvicorn app.main:create_app")


@pytest.mark.parametrize("name", sorted(APP_SERVICES))
def test_app_services_run_as_numeric_non_root_user(services: dict[str, Service], name: str) -> None:
    users = re.findall(r"^USER (\S+)$", dockerfile(services, name), flags=re.MULTILINE)

    assert users, f"{name}: Dockerfile has no USER"
    assert NON_ROOT_NUMERIC.fullmatch(users[-1]), users[-1]
    assert services[name]["user"] == users[-1]


def test_integration_compose_images_are_pinned_by_digest() -> None:
    for name, service in load(TEST_COMPOSE)["services"].items():
        assert DIGEST.search(service["image"]), name


# The query models the API loads (ADR 0008).
BACKEND_MODEL_SETTINGS = (
    "EMBEDDING_DENSE_MODEL",
    "EMBEDDING_SPARSE_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "RERANKER_MODEL",
    "INFERENCE_WORKERS",
    "INFERENCE_ONNX_THREADS",
)


def test_backend_gets_the_model_settings_from_env_and_reads_the_baked_store(
    services: dict[str, Service],
) -> None:
    environment: dict[str, str] = services["backend"]["environment"]

    for name in BACKEND_MODEL_SETTINGS:
        assert str(environment[name]).startswith(f"${{{name}:-"), name
    assert environment["EMBEDDING_CACHE_DIR"] == MODEL_STORE
    assert {k: environment[k] for k in TELEMETRY_ENV} == TELEMETRY_ENV


# Settings the API never reads: ingestion's corpus, chunking and index-retention knobs.
BACKEND_UNUSED_PREFIXES = ("corpus_", "chunk_", "index_")
BACKEND_FIXED = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "QDRANT_URL": "http://qdrant:6333",
    "EMBEDDING_CACHE_DIR": MODEL_STORE,
}


def test_backend_gets_every_setting_it_reads(services: dict[str, Service]) -> None:
    """Every API setting, retrieval knobs included, is passed through from .env."""
    environment: dict[str, str] = services["backend"]["environment"]
    fields = set(Settings.model_fields)
    from_env = {
        name.upper() for name in fields if not name.startswith(BACKEND_UNUSED_PREFIXES)
    } - set(BACKEND_FIXED)

    assert set(environment) == from_env | set(BACKEND_FIXED) | set(TELEMETRY_ENV)
    assert {k: environment[k] for k in BACKEND_FIXED} == BACKEND_FIXED
    assert [k for k in from_env if not str(environment[k]).startswith(f"${{{k}:")] == []
    assert {"RETRIEVAL_MODE", "SEARCH_ENDPOINT_ENABLED", "QUERY_MAX_CHARS"} <= from_env


def test_backend_compose_defaults_equal_the_settings_defaults(
    services: dict[str, Service],
) -> None:
    """A default that differs from the code would change behavior only under compose."""
    environment: dict[str, str] = services["backend"]["environment"]
    defaults = {}
    for key, value in environment.items():
        match = re.fullmatch(r"\$\{(\w+):-(.*)\}", str(value))
        if match and match.group(1) == key:
            defaults[key.lower()] = match.group(2)

    from_compose = Settings(**defaults)  # type: ignore[arg-type]  # strings, parsed by pydantic
    reference = Settings(postgres_password=from_compose.postgres_password)
    differing = [
        name for name in defaults if getattr(from_compose, name) != getattr(reference, name)
    ]

    assert "retrieval_mode" in defaults
    assert differing == []
