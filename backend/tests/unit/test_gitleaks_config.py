"""The gitleaks exemption stays as narrow as ADR 0006 records it.

The behavior was proved by running gitleaks (commit 9b report); this pins the file
so a later edit cannot widen the exemption unnoticed.
"""

import pathlib
import tomllib

GITLEAKS = pathlib.Path(__file__).resolve().parents[3] / ".gitleaks.toml"


def test_gitleaks_keeps_the_default_rules_and_needs_a_version_with_target_rules() -> None:
    config = tomllib.loads(GITLEAKS.read_text(encoding="utf-8"))

    assert config["extend"] == {"useDefault": True}
    assert config["minVersion"] == "v8.30.1"
    assert set(config) == {"minVersion", "extend", "allowlists"}


def test_the_only_exemption_is_digests_in_the_model_pins_for_one_rule() -> None:
    [allowlist] = tomllib.loads(GITLEAKS.read_text(encoding="utf-8"))["allowlists"]

    assert {k: v for k, v in allowlist.items() if k != "description"} == {
        "targetRules": ["generic-api-key"],
        "condition": "AND",
        "paths": [r"^backend/models\.toml$"],
        "regexTarget": "secret",
        "regexes": [r"^[0-9a-f]{64}$"],
    }
