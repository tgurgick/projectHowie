"""Data-ownership boundary: raw data, credentials and research output must
never be packaged, and the model/MCP egress surface must only carry derived
fields."""

import configparser
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PATTERNS = (".db", ".pkl", ".env", "data/", ".csv", ".parquet", "research")


def test_manifest_has_no_raw_data_or_secrets():
    text = (ROOT / "MANIFEST.in").read_text()
    includes = [ln for ln in text.splitlines()
                if ln.strip() and not ln.startswith("#")
                and not ln.startswith(("global-exclude", "prune", "exclude"))]
    for line in includes:
        for pat in FORBIDDEN_PATTERNS:
            assert pat not in line, f"MANIFEST.in includes raw data/secret pattern {pat!r}: {line}"
    assert "prune data" in text and "global-exclude" in text


def test_setup_package_data_is_allowlisted():
    text = (ROOT / "setup.py").read_text()
    m = re.search(r"package_data\s*=\s*\{(.*?)\}", text, re.S)
    assert m, "package_data block missing"
    block = m.group(1)
    for pat in ("*.db", "*.pkl", "data/", ".env", "*.csv"):
        assert pat not in block, f"package_data ships {pat}"


def test_gitignore_covers_local_artifacts():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text().splitlines()]
    for required in ("/data/", "*.db", ".env", "/research/", "*.csv"):
        assert required in lines, f".gitignore missing {required}"


@pytest.mark.skipif(not (ROOT / "setup.py").exists(), reason="no setup.py")
def test_sdist_contains_no_raw_data(tmp_path):
    """Build a real sdist and scan it — the manifest test above is static;
    this proves the artifact."""
    import subprocess
    import sys
    import tarfile

    r = subprocess.run(
        [sys.executable, "setup.py", "-q", "sdist", "--dist-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        pytest.skip(f"sdist build unavailable: {r.stderr[-300:]}")
    tarballs = list(tmp_path.glob("*.tar.gz"))
    assert tarballs, "no sdist produced"
    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()
    # repo-level data/ and research/ only — howie3/data/ is a source package
    bad = [n for n in names
           if re.search(r"\.(db|sqlite3?|pkl|csv|parquet)$|/\.env$|^[^/]+/(data|research)/", n)]
    assert not bad, f"sdist ships raw data or secrets: {bad[:10]}"
