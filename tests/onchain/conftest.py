"""Fixtures for on-chain integration tests (marker ``onchain``).

* RPC URLs come only from the environment (<CHAIN>_RPC_URL, e.g. AVALANCHE_RPC_URL; a repo-root .env is loaded if present).
  Tests for a chain are skipped when its variable is not set.
* Every comparison for a chain uses ONE pinned block (latest - PIN_LAG at session start), so the
  state snapshot and the quoter eth_call see identical chain state.
* A comparison table (v4quote vs on-chain, diff in wei) is printed at the end of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from v4quote.chain.rpc import CHAINS, get_web3, rpc_url

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PIN_LAG = 3  # blocks behind head, so that every RPC node behind a load balancer has the block

try:  # optional convenience; never overrides variables already set in the environment
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def load_pools_config() -> dict:
    return yaml.safe_load((FIXTURES / "pools.yaml").read_text(encoding="utf-8"))


POOLS_CONFIG = load_pools_config()


def pools_of_type(*types: str) -> list:
    out = []
    for p in POOLS_CONFIG.get("pools") or []:
        if p["type"] in types:
            marks = []
            if rpc_url(p["chain"]) is None:
                marks.append(pytest.mark.skip(reason=f"{CHAINS[p['chain']].rpc_env} not set"))
            out.append(pytest.param(p, id=p["id"], marks=marks))
    return out


def token_address(chain: str, symbol_or_address: str) -> str:
    if symbol_or_address.startswith("0x"):
        return symbol_or_address
    return POOLS_CONFIG["tokens"][chain][symbol_or_address]


@dataclass
class Comparison:
    pool_id: str
    block: int
    case: str
    ours: object
    chain: object
    extra: str = ""

    @property
    def diff(self) -> str:
        if isinstance(self.ours, int) and isinstance(self.chain, int):
            return str(self.ours - self.chain)
        return "0" if self.ours == self.chain else "MISMATCH"


@dataclass
class Report:
    rows: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_REPORT = Report()


@pytest.fixture(scope="session")
def report() -> Report:
    return _REPORT


class ChainCtx:
    def __init__(self, name: str):
        self.name = name
        self.w3 = get_web3(name)
        self.block = self.w3.eth.block_number - PIN_LAG
        self.timestamp = int(self.w3.eth.get_block(self.block)["timestamp"])


_CTX: dict[str, ChainCtx] = {}


@pytest.fixture(scope="session")
def chain_ctx():
    def get(name: str) -> ChainCtx:
        if rpc_url(name) is None:
            pytest.skip(f"{CHAINS[name].rpc_env} not set")
        if name not in _CTX:
            _CTX[name] = ChainCtx(name)
        return _CTX[name]

    return get


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _REPORT.rows and not _REPORT.notes:
        return
    tr = terminalreporter
    tr.section("v4quote vs on-chain quoter (same pinned block)")
    for name, ctx in _CTX.items():
        tr.write_line(f"{name}: block {ctx.block} (timestamp {ctx.timestamp})")
    tr.write_line(f"{'pool':44} {'case':46} {'core':>30} {'on-chain':>30} {'diff':>6}  extra")
    for r in _REPORT.rows:
        tr.write_line(f"{r.pool_id:44} {r.case:46} {str(r.ours):>30} {str(r.chain):>30} {r.diff:>6}  {r.extra}")
    mismatches = [r for r in _REPORT.rows if r.diff not in ("0",)]
    tr.write_line(f"comparisons: {len(_REPORT.rows)}, non-zero diffs: {len(mismatches)}")
    for n in _REPORT.notes:
        tr.write_line(f"note: {n}")
