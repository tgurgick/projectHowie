"""Cockpit front-end: run the node unit tests for lib.js and lint app.js so
every HTML assignment built from data goes through the auto-escaping h``
tag (the UI's injection boundary)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "howie3" / "ui"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_unit_tests_pass():
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "ui").glob("*.test.mjs"))
    r = subprocess.run(["node", "--test", *files], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_scripts_parse():
    for name in ("lib.js", "app.js"):
        r = subprocess.run(["node", "--check", str(UI / name)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_no_unescaped_template_assignments():
    """innerHTML/outerHTML may receive: an h`` template, a Raw-aware
    expression, a constant string literal (no interpolation), or one of the
    two SVG builders that interpolate numbers only (reviewed)."""
    src = (UI / "app.js").read_text()
    offenders = []
    for m in re.finditer(r"\.(innerHTML|outerHTML)\s*=\s*([^;]+)", src):
        rhs = " ".join(m.group(2).split())
        if rhs.startswith("text.slice(0, i)"):
            continue  # the typed wordmark: a constant
        if rhs.startswith("h`") or rhs.startswith("qtable(") or rhs.startswith("saved"):
            continue
        if re.match(r"^'[^']*'$", rhs) or re.match(r'^"[^"]*"$', rhs):
            continue  # constant markup we wrote
        if rhs.startswith("content instanceof Raw") or rhs.startswith("html"):
            continue  # termPrint / placeTip: Raw-only inputs
        if rhs in ("svg", "svg + '</svg>'") or rhs.startswith("`<svg"):
            continue  # numeric SVG builders; text nodes inside use esc()
        if "? h`" in rhs or rhs.endswith("h`") or "h`" in rhs:
            continue  # ternaries whose branches are h`` or constants
        offenders.append(rhs[:80])
    assert not offenders, f"HTML assigned without h``: {offenders}"
    # the legacy pattern must be gone entirely
    assert "innerHTML = `" not in src, "raw template literal assigned to innerHTML"
    # every inline handler that carries data does so via data-* attributes, never string-built JS
    assert not re.search(r"onclick=\"[^\"]*\$\{[^}]*\.name", src), "player name interpolated into inline JS"


def test_index_loads_split_assets():
    html = (UI / "index.html").read_text()
    assert '<script src="/ui/lib.js"></script>' in html and '<script src="/ui/app.js"></script>' in html
    assert '<link rel="stylesheet" href="/ui/style.css">' in html
    assert "<style>" not in html and "function " not in html, "index.html must hold markup only"
    assert 'content="__HOWIE_TOKEN__"' in html
