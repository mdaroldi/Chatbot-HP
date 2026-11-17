"""`make sbom`: a local CycloneDX bill of materials per image, never committed or uploaded.

It reuses the pinned Trivy image the scans use, so it adds no tool. The output lists
every package with its version and license, which a reviewer can diff between
releases; it stays local because CI uploads no artifacts (test_ci_workflow).
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
MAKEFILE = ROOT / "Makefile"
GITIGNORE = ROOT / ".gitignore"
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def recipe(target: str) -> list[str]:
    lines = MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{target}:"))
    body = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        body.append(line.strip())
    return body


def test_sbom_is_a_phony_documented_target() -> None:
    text = MAKEFILE.read_text()

    assert re.search(r"^sbom: .*## ", text, re.MULTILINE)
    phony = text.split(".PHONY:", 1)[1].split("\n\n", 1)[0]
    assert re.search(r"\bsbom\b", phony)


def test_sbom_uses_the_pinned_trivy_image_in_cyclonedx_for_every_image() -> None:
    body = " ".join(recipe("sbom"))

    assert "$(TRIVY_IMAGE)" in body
    assert "--format cyclonedx" in body
    assert "for image in $(IMAGES)" in body
    assert "|| exit 1" in body


def test_sbom_writes_into_a_git_ignored_directory_only() -> None:
    body = " ".join(recipe("sbom"))
    ignored = GITIGNORE.read_text().splitlines()

    assert '"$(CURDIR)/sbom":/out' in body
    assert "--output /out/" in body
    assert "/sbom/" in ignored


def test_no_workflow_runs_sbom() -> None:
    assert all("make sbom" not in path.read_text() for path in WORKFLOWS)
