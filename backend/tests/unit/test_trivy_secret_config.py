"""The Trivy secret-scan exception stays as narrow as it was proved to be.

Trivy's built-in `gcp-service-account` rule matches the literal `"type":
"service_account"`, which the Anthropic SDK writes in docstrings of its beta types.
A plain allow rule would not do: Trivy ORs an allow rule's `path` and `regex`, so a
regex would silence the literal in every file. Instead the built-in rule is replaced
by the same rule with one path exemption, the SDK's package directory. Proved by
running Trivy 0.76.1 on synthetic files: the literal outside the package is still
reported, and other rules (an AWS key) still fire inside it. This test pins the file
so a later edit cannot widen the exemption unnoticed.
"""

import pathlib
import re
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG = ROOT / "trivy-secret.yaml"
MAKEFILE = ROOT / "Makefile"
SDK_FILE = "/opt/venv/lib/python3.12/site-packages/anthropic/types/beta/agent_create_params.py"


def config() -> dict[str, Any]:  # Any: parsed YAML
    data: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return data


def test_only_the_gcp_rule_is_disabled_and_nothing_is_allowed_globally() -> None:
    data = config()

    assert set(data) == {"disable-rules", "rules"}
    assert data["disable-rules"] == ["gcp-service-account"]


def test_the_gcp_rule_returns_with_one_exemption_the_sdk_package() -> None:
    (rule,) = config()["rules"]

    assert rule["severity"] == "CRITICAL"
    assert re.search(rule["regex"], '{"type": "service_account", "private_key": "..."}')
    (allow,) = rule["allow-rules"]
    assert set(allow) == {"id", "description", "path"}  # no regex: it would apply anywhere
    assert re.search(allow["path"], SDK_FILE)
    for elsewhere in (
        "/opt/venv/lib/python3.12/site-packages/anthropic_other/x.py",
        "/opt/venv/lib/python3.12/site-packages/google/auth/x.py",
        "/app/app/main.py",
        "/app/key.json",
    ):
        assert not re.search(allow["path"], elsewhere), elsewhere


def test_the_image_scan_uses_the_config() -> None:
    recipe = MAKEFILE.read_text(encoding="utf-8").split("scan-images:", 1)[1].split("\n\n", 1)[0]

    assert "--secret-config /src/trivy-secret.yaml" in recipe
