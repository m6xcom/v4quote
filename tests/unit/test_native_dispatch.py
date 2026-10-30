"""``v4quote.amm.native`` dispatch layer: build fingerprint, why the Python backend was chosen
handle counters and the handle cache identity rule.

The fingerprint algorithm and the backend selection run without the extension (fake modules, subprocesses with a
stand-in ``v4quote_native``); the comparisons with the real build need it (skipped otherwise)."""
from __future__ import annotations

import gc
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from v4quote.amm import native
from v4quote.amm.v3 import pool as v3m
from v4quote.amm.v3.tick_bitmap import TickBitmapSnapshot
from v4quote.amm.v3.pool import V3PoolState

REPO = Path(__file__).resolve().parents[2]
B = "0x" + "b2" * 20
C = "0x" + "c3" * 20
Q96 = 1 << 96


def addr(n: int) -> str:
    return "0x" + f"{n:040x}"


def v3_full_range(a, t0, t1, liquidity, *, fee=500, ts=10, word_range=None, sqrt_price=Q96, tick=0) -> V3PoolState:
    hi = (887272 // ts) * ts
    ticks = TickBitmapSnapshot.from_ticks(ts, {-hi: liquidity, hi: -liquidity}, word_range)
    return V3PoolState(a, t0, t1, fee, ts, sqrt_price, tick, liquidity, ticks)


# ------------------------------------------------------------------------------------------------ fingerprint
def _crate(root: Path, lib: bytes = b"pub fn f() {}\n") -> Path:
    (root / "src" / "sub").mkdir(parents=True)
    (root / "Cargo.toml").write_bytes(b"[package]\nname = \"x\"\n")
    (root / "Cargo.lock").write_bytes(b"version = 4\n")
    (root / "build.rs").write_bytes(b"fn main() {}\n")
    (root / "src" / "lib.rs").write_bytes(lib)
    (root / "src" / "sub" / "m.rs").write_bytes(b"// m\n")
    return root


def _manual(root: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        d = (root / f).read_bytes().replace(b"\r\n", b"\n")
        h.update(f.encode() + b"\n" + str(len(d)).encode() + b"\n" + d + b"\n")
    return h.hexdigest()


def test_source_hash_algorithm(tmp_path):
    root = _crate(tmp_path / "a")
    want = _manual(root, ["Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs", "src/sub/m.rs"])
    assert native.source_hash(root) == want
    # CRLF checkout (Windows autocrlf) hashes like LF
    crlf = _crate(tmp_path / "b", lib=b"pub fn f() {}\r\n")
    assert native.source_hash(crlf) == want
    # a non-.rs file under src/ is not part of the build inputs
    (root / "src" / "notes.txt").write_bytes(b"x")
    assert native.source_hash(root) == want
    # any change to the hashed files changes it, including a new module in a sub-directory
    (root / "src" / "lib.rs").write_bytes(b"pub fn f() { }\n")
    assert native.source_hash(root) != want
    (root / "src" / "lib.rs").write_bytes(b"pub fn f() {}\n")
    (root / "src" / "sub" / "n.rs").write_bytes(b"")
    assert native.source_hash(root) not in (None, want)
    # not a crate checkout: nothing to compare with
    assert native.source_hash(tmp_path / "missing") is None
    (tmp_path / "a" / "Cargo.lock").unlink()
    assert native.source_hash(tmp_path / "a") is None


def test_check_build_reasons(tmp_path):
    root = _crate(tmp_path / "c")
    h = native.source_hash(root)
    assert native.check_build(SimpleNamespace(__source_hash__=h), root) == (None, "ok")
    reason, check = native.check_build(SimpleNamespace(__source_hash__="0" * 64), root)
    assert reason.startswith("stale_build: ") and check == "mismatch" and h[:12] in reason
    reason, check = native.check_build(SimpleNamespace(), root)  # built before the fingerprint existed
    assert reason.startswith("stale_build: ") and check == "mismatch"
    reason, check = native.check_build(SimpleNamespace(__source_hash__=h), tmp_path / "nowhere")
    assert reason is None and check.startswith("skipped")


def test_real_build_fingerprint_matches_the_repository():
    cn = pytest.importorskip("v4quote_native")
    h = native.source_hash(REPO / "native")
    assert h is not None and native.SOURCE_DIR == REPO / "native"
    assert cn.__source_hash__ == h, "v4quote_native is stale: rebuild (uv sync --locked --extra native)"
    assert cn.__build__ == f"{cn.__version__}+{h}"
    if os.environ.get(native.ENV, "1").strip() != "0":
        assert native.ACTIVE and native.BUILD == cn.__build__ and native.BUILD_CHECK == "ok"


# ------------------------------------------------------------------------------------------------ reasons
_PROBE = ("import logging, sys\n"
          "{pre}\n"
          "from v4quote.amm import native\n"
          "b = native.backend()\n"
          "print(b, native.ACTIVE, native.describe()['reason'])\n")


def _probe(tmp_path: Path, pre: str = "", env: dict | None = None, path: Path | None = None) -> tuple[str, str]:
    e = dict(os.environ)
    e.pop(native.ENV, None)
    e.update(env or {})
    if path is not None:
        e["PYTHONPATH"] = os.pathsep.join([str(path), e.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, "-c", _PROBE.format(pre=pre)], env=e, capture_output=True, text=True,
                       check=True, cwd=tmp_path)
    return r.stdout.strip(), r.stderr


def test_reason_not_installed(tmp_path):
    out, err = _probe(tmp_path, "sys.modules['v4quote_native'] = None  # as if the extra was not synced")
    assert out.startswith("python False not_installed: ")
    assert "v4quote math backend: python (not_installed: " in err  # WARNING (the last-resort handler shows it)


def test_reason_broken_build_keeps_the_error_text(tmp_path):
    fake = tmp_path / "fake"
    (fake / "v4quote_native").mkdir(parents=True)
    (fake / "v4quote_native" / "__init__.py").write_text("raise ImportError('DLL load failed: ' + 'x' * 500)\n")
    out, err = _probe(tmp_path, path=fake)
    assert out.startswith("python False broken_build: ImportError: DLL load failed: xxx")
    assert len(out.split("broken_build: ImportError: ", 1)[1]) == 200  # str(e)[:200]
    assert "v4quote math backend: python (broken_build: ImportError: DLL load failed" in err
    # a missing dependency of the extension is a broken build, not "not installed"
    (fake / "v4quote_native" / "__init__.py").write_text("import _no_such_dependency_of_core_native\n")
    out, _ = _probe(tmp_path, path=fake)
    assert out.startswith("python False broken_build: ModuleNotFoundError: ")


def test_reason_stale_build_selects_python(tmp_path):
    fake = tmp_path / "fake"
    (fake / "v4quote_native").mkdir(parents=True)
    (fake / "v4quote_native" / "__init__.py").write_text("__version__ = '0.0.9'\n__source_hash__ = '0' * 64\n")
    out, err = _probe(tmp_path, path=fake)
    assert out.startswith("python False stale_build: v4quote_native was built from sources 000000000000")
    assert "v4quote math backend: python (stale_build" in err
    (fake / "v4quote_native" / "__init__.py").write_text("__version__ = '0.0.9'\n")  # pre-fingerprint build
    out, _ = _probe(tmp_path, path=fake)
    assert out.startswith("python False stale_build: v4quote_native has no __source_hash__")


def test_reason_disabled_is_not_a_warning(tmp_path):
    out, err = _probe(tmp_path, env={native.ENV: "0"})
    assert out == "python False disabled: V4QUOTE_NATIVE=0"
    assert "v4quote math backend" not in err  # INFO: below the last-resort handler's WARNING level


# ------------------------------------------------------------------------------------------------ handles
@pytest.fixture
def active():
    if not native.ACTIVE:
        pytest.skip("native backend not active (not installed, stale or V4QUOTE_NATIVE=0)")


def _pool(liq: int = 10**22) -> v3m.V3PoolState:
    return v3_full_range(addr(2), B, C, liq)


def test_handle_counters(active):
    before = dict(native._stats)
    st = _pool()
    assert native.handle(st) is not None and native.handle(st) is not None  # second call: cache hit, no count
    assert native._stats["handles"] == before["handles"] + 1

    class Sub(v3m.V3PoolState):
        pass

    sub = Sub(**{f: getattr(st, f) for f in st.__dataclass_fields__})
    assert native.handle(sub) is None
    assert native._stats["no_handle"] == before["no_handle"] + 1
    d = native.describe()
    assert d["handles"] == native._stats["handles"] and d["no_handle"] == native._stats["no_handle"]


def test_cache_hit_requires_the_same_object(active):
    """A cache entry under an object's id is used only while its weak reference points at that very object: an id
    recycled before the entry was dropped (or any stale entry) is a miss, never another state's handle."""
    a, b = _pool(10**22), _pool(3 * 10**21)
    ha = native.handle(a)
    native._by_id[id(b)] = native._by_id[id(a)]  # b's id now maps to a's entry (a recycled-id stand-in)
    hb = native.handle(b)
    assert hb is not None and hb is not ha
    assert native._by_id[id(b)][0]() is b
    amt = 10**20
    assert native.quote(b, 0, 1, amt)[0] == v3m.quote_exact_input(b, True, amt, want_tick=False).amount_out
    assert native.quote(a, 0, 1, amt)[0] == v3m.quote_exact_input(a, True, amt, want_tick=False).amount_out
    # the same for the shared tick-data cache
    t2 = TickBitmapSnapshot.from_ticks(a.ticks.tick_spacing, dict(a.ticks.liquidity_net))
    native._ticks_by_id[id(t2)] = native._ticks_by_id[id(a.ticks)]
    native._ticks(t2)
    assert native._ticks_by_id[id(t2)][0]() is t2
    # the entry goes with the state
    k = id(b)
    del hb, b
    gc.collect()
    assert k not in native._by_id


def test_new_state_objects_get_new_handles(active):
    """The supported way to change a state (a new object: dataclasses.replace, a new TickBitmapSnapshot) never sees an
    old handle; results follow the new values."""
    st = _pool()
    amt = 10**21
    q0 = native.quote(st, 0, 1, amt)
    hi = max(st.ticks.liquidity_net)
    thin = TickBitmapSnapshot.from_ticks(st.ticks.tick_spacing, {-hi: 10**20, hi: -10**20})
    st2 = replace(st, liquidity=10**20, ticks=thin)
    q1 = native.quote(st2, 0, 1, amt)
    assert native.handle(st2) is not native.handle(st)
    assert q1[0] == v3m.quote_exact_input(st2, True, amt, want_tick=False).amount_out != q0[0]


def test_no_in_place_writes_to_snapshot_mappings():
    """RULE (native.py): states and snapshots are never mutated in place - a handle built before such a
    write would keep the old values. Tripwire over the Python sources of the package."""
    pat = re.compile(r"\.(words|liquidity_net)\s*(\[[^\]]*\]\s*(=(?!=)|[+\-*/|&^]=)|\.(update|pop|popitem|clear|setdefault)\()"
                     r"|del\s+[\w.\[\]]*\.(words|liquidity_net)\[")
    hits = []
    for pkg in ("src",):
        for p in (REPO / pkg).rglob("*.py"):
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{p.relative_to(REPO)}:{n}: {line.strip()}")
    assert not hits, "in-place write to a snapshot mapping (make a new snapshot instead):\n" + "\n".join(hits)
