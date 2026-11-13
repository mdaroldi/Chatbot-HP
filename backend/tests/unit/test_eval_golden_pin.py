"""The frozen golden set (ADR 0009, Phase 2 commit 11b).

Every default and threshold of Phase 2 is chosen on this file, so it must not change
without a new version, a new ADR record and this pin moving with them.
"""

import hashlib
import pathlib

from app.evaluation.cli import GOLDEN_FILE
from app.evaluation.golden import load_golden
from app.ingestion.corpus import load_corpus

EVAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "eval"
FROZEN_VERSION = "1.0.1"
FROZEN_SHA256 = "91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416"


def test_the_golden_set_is_the_frozen_file() -> None:
    path = EVAL_DIR / GOLDEN_FILE

    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_SHA256


def test_the_frozen_file_loads_with_its_recorded_version_and_hash() -> None:
    golden = load_golden(EVAL_DIR / GOLDEN_FILE, {d.doc_id for d in load_corpus()})

    assert (golden.version, golden.sha256) == (FROZEN_VERSION, FROZEN_SHA256)
    assert len(golden.questions) == 70
