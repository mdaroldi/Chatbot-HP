"""Hardening invariants of docker-compose.yml.

Trivy's config scanner has no docker-compose support, so these rules are pinned
here instead of being checked only by hand.
"""

import pathlib
import re
from typing import Any

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
COMPOSE = REPO / "docker-compose.yml"
TEST_COMPOSE = REPO / "docker-compose.test.yml"

SERVICES = {"backend", "frontend", "postgres", "qdrant"}
APP_SERVICES = {"backend", "frontend"}
DATA_SERVICES = {"postgres", "qdrant"}
BIND = "${BIND_ADDRESS:-127.0.0.1}"

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


def test_stack_has_exactly_the_four_services(services: dict[str, Service]) -> None:
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


def test_only_api_and_frontend_publish_ports_on_the_bind_address(
    services: dict[str, Service],
) -> None:
    published = {name for name, service in services.items() if service.get("ports")}

    assert published == APP_SERVICES
    assert services["backend"]["ports"] == [f"{BIND}:${{API_PORT:-8000}}:8000"]
    assert services["frontend"]["ports"] == [f"{BIND}:${{FRONTEND_PORT:-8501}}:8501"]
    for name in DATA_SERVICES:
        assert "expose" not in services[name]


def test_data_network_is_internal_and_only_backend_bridges_it(
    compose: dict[str, Any], services: dict[str, Service]
) -> None:
    assert compose["networks"]["data"]["internal"] is True
    assert compose["networks"]["edge"].get("internal") is not True
    assert set(services["backend"]["networks"]) == {"edge", "data"}
    assert set(services["frontend"]["networks"]) == {"edge"}
    for name in DATA_SERVICES:
        assert set(services[name]["networks"]) == {"data"}


@pytest.mark.parametrize("name", sorted(DATA_SERVICES))
def test_third_party_images_are_pinned_by_digest(services: dict[str, Service], name: str) -> None:
    assert "@sha256:" in services[name]["image"]


def test_postgres_password_is_required_and_never_defaulted(
    services: dict[str, Service],
) -> None:
    for name in ("backend", "postgres"):
        required = "${POSTGRES_PASSWORD:?set it in .env}"
        assert services[name]["environment"]["POSTGRES_PASSWORD"] == required
    assert "POSTGRES_PASSWORD" not in services["frontend"]["environment"]
    assert "env_file" not in services["frontend"]


def test_qdrant_telemetry_is_disabled(services: dict[str, Service]) -> None:
    assert services["qdrant"]["environment"]["QDRANT__TELEMETRY_DISABLED"] == "true"


def test_app_services_use_an_init_process(services: dict[str, Service]) -> None:
    for name in APP_SERVICES:
        assert services[name]["init"] is True


@pytest.mark.parametrize("name", sorted(DATA_SERVICES))
def test_data_services_have_healthchecks(services: dict[str, Service], name: str) -> None:
    assert services[name]["healthcheck"]["test"]


def test_startup_order_waits_for_healthy_dependencies(services: dict[str, Service]) -> None:
    backend_deps = services["backend"]["depends_on"]
    frontend_deps = services["frontend"]["depends_on"]

    assert {name: dep["condition"] for name, dep in backend_deps.items()} == {
        "postgres": "service_healthy",
        "qdrant": "service_healthy",
    }
    assert {name: dep["condition"] for name, dep in frontend_deps.items()} == {
        "backend": "service_healthy",
    }


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


@pytest.mark.parametrize("name", sorted(APP_SERVICES))
def test_app_services_run_as_numeric_non_root_user(services: dict[str, Service], name: str) -> None:
    users = re.findall(r"^USER (\S+)$", dockerfile(services, name), flags=re.MULTILINE)

    assert users, f"{name}: Dockerfile has no USER"
    assert NON_ROOT_NUMERIC.fullmatch(users[-1]), users[-1]
    assert services[name]["user"] == users[-1]


def test_integration_compose_images_are_pinned_by_digest() -> None:
    for name, service in load(TEST_COMPOSE)["services"].items():
        assert DIGEST.search(service["image"]), name
