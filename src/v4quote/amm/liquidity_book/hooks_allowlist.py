"""Allowlist of LB v2.2 hooks contracts whose pools v4quote may simulate.

An entry pins the hook by (chain, address), binds it to the one pair it was
verified with (``pair``), and pins the on-chain codehash and, for proxies, the
implementation address and implementation codehash. Any change on chain (upgrade,
beacon switch, redeploy, the hook reused by another pair) makes the pool
``UnsupportedPool`` again. For LFJ rewarder clones the pool is also rejected while an
extra rewarder is set or the hook is unlinked (``pair.LBPairState.unsupported_reasons``).

``execution_allowed`` is a policy flag for callers that send transactions: simulating a pool is
permitted by being in the list, sending transactions through it is not, unless this flag is True. All entries
are False.

Evidence: a replay of historical ``swap()`` calls of each pair (hooks executed) against the model,
``tests/onchain/test_lb_swap_replay.py``, plus the codehash / pair links re-checked by
``tests/onchain/test_address_links.py``. Entries whose on-chain swap replay is still pending are listed in
``REPLAY_PENDING``: their pools are simulated, but results through them should be treated as unverified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

Verification = Literal["source-verified", "empirical"]
ProxyKind = Literal["lfj-immutable-clone", "eip1967-beacon", "eip1967"]

_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KNOWN_CHAINS = ("monad", "avalanche")
EVIDENCE = "tests/onchain/test_lb_swap_replay.py"  # where the entries are verified against the chain


@dataclass(frozen=True)
class HookApproval:
    """Allowlist entry for one hooks contract (hex strings compared case-insensitively).

    Validation (raises ValueError): ``pair`` is a mandatory address (the LBPair whose
    ``getLBHooksParameters()`` points at this hook); codehashes are
    32-byte hex; ``verification`` is "source-verified" or "empirical"; for a proxy
    (``proxy_kind`` set) both ``implementation`` and ``implementation_codehash`` are
    mandatory, for a non-proxy both must be absent.
    """

    codehash: str
    implementation: Optional[str] = None
    implementation_codehash: Optional[str] = None
    proxy_kind: Optional[ProxyKind] = None
    verification: Verification = "empirical"
    execution_allowed: bool = False
    pinned_block: Optional[int] = None
    added_date: Optional[str] = None
    evidence: Optional[str] = None
    pair: Optional[str] = None  # mandatory (validated below); last only because the fields above are positional

    def __post_init__(self) -> None:
        if not (isinstance(self.pair, str) and _ADDR_RE.match(self.pair)):
            raise ValueError(f"allowlist entry requires the pair address, got {self.pair!r}")
        if not _HASH_RE.match(self.codehash or ""):
            raise ValueError(f"invalid codehash {self.codehash!r}")
        if self.verification not in ("source-verified", "empirical"):
            raise ValueError(f"invalid verification {self.verification!r}")
        if not isinstance(self.execution_allowed, bool):
            raise ValueError("execution_allowed must be bool")
        if self.proxy_kind is not None:
            if self.proxy_kind not in ("lfj-immutable-clone", "eip1967-beacon", "eip1967"):
                raise ValueError(f"invalid proxy_kind {self.proxy_kind!r}")
            if not (self.implementation and _ADDR_RE.match(self.implementation)):
                raise ValueError("proxy entry requires a valid implementation address")
            if not (self.implementation_codehash and _HASH_RE.match(self.implementation_codehash)):
                raise ValueError("proxy entry requires implementation_codehash")
        elif self.implementation is not None or self.implementation_codehash is not None:
            raise ValueError("implementation fields set but proxy_kind is None")
        if self.added_date is not None and not _DATE_RE.match(self.added_date):
            raise ValueError(f"invalid added_date {self.added_date!r}")


HooksAllowlist = Mapping[tuple[str, str], HookApproval]  # key: (chain, hook address)


def validate_allowlist(allowlist: HooksAllowlist) -> None:
    pairs: set[tuple[str, str]] = set()
    for (chain, hook), approval in allowlist.items():
        if chain not in KNOWN_CHAINS:
            raise ValueError(f"unknown chain {chain!r}")
        if not _ADDR_RE.match(hook):
            raise ValueError(f"invalid hook address {hook!r}")
        if not isinstance(approval, HookApproval):
            raise ValueError("allowlist values must be HookApproval")
        key = (chain, str(approval.pair).lower())
        if key in pairs:
            raise ValueError(f"pair {approval.pair} on {chain} has more than one allowlisted hook")
        pairs.add(key)


# Codehashes re-read on-chain when the entries were added (2026-09-21):
# Avalanche block 95854756/95854758, Monad blocks 106830370/106830372.
# AUSD pair hooks added 2026-09-23 at Avalanche block 95951189.
# USDt/USDC and sAVAX/WAVAX pair hooks added 2026-09-29 at Avalanche block 96389826.
# ``pair``: the pair each hook was verified and replayed with (tests/fixtures/pools.yaml / route_pools.yaml);
# re-checked on chain by tests/onchain/test_address_links.py::test_default_hooks_allowlist_matches_chain
# (pair.getLBHooksParameters() points at the hook).
_LFJ_HOOK_IMPL = "0x2235CDFe1172aA5eD7cEC6E1713f93eeaBFbc295"

DEFAULT_HOOKS_ALLOWLIST: HooksAllowlist = {
    # LFJ WAVAX/USDC bs10 (0x864d4e5e...16ea). Implementation LBHooksSimpleRewarder verified on routescan.
    ("avalanche", "0xBA83987a28b1C7514576e15D58c134b8F0E00Fa9"): HookApproval(
        pair="0x864d4e5Ee7318e97483DB7EB0912E09F161516EA",
        codehash="0x419f4e8dcd4dd32a0220cb95d64b152d3331fc5a8fa7b3d30ac6d7ac256a17b7",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0x0e088a5b6645a261bcdcb2efad8789e25f5b423a129dbd4188869a8361e73ba1",
        proxy_kind="lfj-immutable-clone",
        verification="source-verified",
        execution_allowed=False,
        pinned_block=95854756,
        added_date="2026-09-21",
        evidence=EVIDENCE,
    ),
    # LFJ WMON/USDC bs10 (0x5e60bc3f...fe22). Monad implementation bytecode differs from the verified
    # Avalanche one -> empirical (swap replay) only.
    ("monad", "0xd0B3Ff65058609743877637e689B30abdb97364a"): HookApproval(
        pair="0x5E60BC3F7a7303BC4dfE4dc2220bdC90bc04fE22",
        codehash="0x908dd8930d84a8f5bae05ee104f1dbae9c34d3f08640c27b2cb1cc279b875769",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0xef6f52a0e9e2cf26dd81f5456c6071f58dafbb06017c411f9f02b7a93a452ea1",
        proxy_kind="lfj-immutable-clone",
        verification="empirical",
        execution_allowed=False,
        pinned_block=106830370,
        added_date="2026-09-21",
        evidence=EVIDENCE,
    ),
    # LFJ WMON/USDC bs5 (0x5AFD3EC8...7620).
    ("monad", "0x8b0D325fEa6F13BB6D62F825137ABd4cF6457367"): HookApproval(
        pair="0x5AFD3EC861f6104af26e8755aBcc1f876de77620",
        codehash="0xcd9425f235e9de88175cbd19c472d9a3c5ff6d73ed0c4db2169fa77adda3a180",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0xef6f52a0e9e2cf26dd81f5456c6071f58dafbb06017c411f9f02b7a93a452ea1",
        proxy_kind="lfj-immutable-clone",
        verification="empirical",
        execution_allowed=False,
        pinned_block=106830372,
        added_date="2026-09-21",
        evidence=EVIDENCE,
    ),
    # Pharaoh DLMM WAVAX/USDC bs5 (0x8aC5707f...3379). BeaconProxy; beacon 0xd28467eD...548C.
    ("avalanche", "0x138ce0008F7feBdcFfdf27C0aD156584eCa7D5B6"): HookApproval(
        pair="0x8aC5707f8D4BDe1d771d34c7AfD81c3922b73379",
        codehash="0xa57cb656b9262cee1544dfe70bbd760543cd40bff89a2a1c06f29f888d1ed741",
        implementation="0xC997575204290FF7106aB8b2BCFa7e7dEA43D783",
        implementation_codehash="0x67524de60f5a73e3f9ec5d78fea749d2133b323ef1fa340bf151866b0258fb51",
        proxy_kind="eip1967-beacon",
        verification="empirical",
        execution_allowed=False,
        pinned_block=95854758,
        added_date="2026-09-21",
        evidence=EVIDENCE,
    ),
    # LFJ AUSD/USDC bs1 (0x8573f981...9cee). Same implementation codehash as the WAVAX/USDC hook above;
    # clone bytecode identical to it except the embedded pair address. Allowlisted for simulation 2026-09-27.
    ("avalanche", "0xE104964852BE626EE27762712E4DE521066859C9"): HookApproval(
        pair="0x8573F98175D816d520248B5fACF40D309B1c9ceE",
        codehash="0xdaf577bdcbb29359151d7d9f5d0f0e3bf7bf2aa7cc5a7ff09aea9efe49c29dbf",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0x0e088a5b6645a261bcdcb2efad8789e25f5b423a129dbd4188869a8361e73ba1",
        proxy_kind="lfj-immutable-clone",
        verification="source-verified",
        execution_allowed=False,
        pinned_block=95951189,
        added_date="2026-09-23",
        evidence=EVIDENCE,
    ),
    # LFJ AUSD/USDt bs1 (0xcec37728...631a). Same as above. Allowlisted for simulation 2026-09-27.
    ("avalanche", "0x72ed0b6AcB5c585873b3F644F99FC167c7601256"): HookApproval(
        pair="0xcEC377285AbF370FDf872625D2742252656d631a",
        codehash="0x1131a0dabd90112bc1cbbd9cdc68110d54374178e8572eaa2dde5fdd27f79006",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0x0e088a5b6645a261bcdcb2efad8789e25f5b423a129dbd4188869a8361e73ba1",
        proxy_kind="lfj-immutable-clone",
        verification="source-verified",
        execution_allowed=False,
        pinned_block=95951189,
        added_date="2026-09-23",
        evidence=EVIDENCE,
    ),
    # LFJ USDt/USDC bs1 (0x2823299a...ef5d). Clone of the same verified LBHooksSimpleRewarder as 0xBA83...0Fa9;
    # clone bytecode identical except the embedded pair address. Approved for simulation 2026-09-29.
    ("avalanche", "0x90014c5E78Bd8a3E2bbBaA8e5a0Aee2C1D2131CD"): HookApproval(
        pair="0x2823299af89285fF1a1abF58DB37cE57006FEf5D",
        codehash="0x5fd2e48584442dd82aaa6ef7be5ab8a9fa699c9c456a560545800042892b483a",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0x0e088a5b6645a261bcdcb2efad8789e25f5b423a129dbd4188869a8361e73ba1",
        proxy_kind="lfj-immutable-clone",
        verification="source-verified",
        execution_allowed=False,
        pinned_block=96389826,
        added_date="2026-09-29",
        evidence=EVIDENCE,
    ),
    # LFJ sAVAX/WAVAX bs5 (0x883ea72c...c5c0). Same as above. Approved for simulation 2026-09-29.
    ("avalanche", "0xdAe382F3542dd8a6b42F0F1F71B3f7227a4a3622"): HookApproval(
        pair="0x883eA72c2A46F7AcB3820855344C43666c6cc5c0",
        codehash="0x68d59a1ef8d61856d4a2c036b4603529a848af31cc7efabdae1d2c7fba8dd928",
        implementation=_LFJ_HOOK_IMPL,
        implementation_codehash="0x0e088a5b6645a261bcdcb2efad8789e25f5b423a129dbd4188869a8361e73ba1",
        proxy_kind="lfj-immutable-clone",
        verification="source-verified",
        execution_allowed=False,
        pinned_block=96389826,
        added_date="2026-09-29",
        evidence=EVIDENCE,
    ),
    # Pharaoh DLMM USDt/USDC bs1 (0x16888ee5...255c). BeaconProxy of the same beacon 0xd28467eD...548C and the same
    # DLMMRewarder implementation as 0x138c...D5B6 (identical proxy codehash); pair = same unverified implementation
    # 0xF41253C1...d710 as 0x8aC5...3379. Rewards were active when added (2026-09-29).
    ("avalanche", "0x32E68C4b8Edc6d1F16E99EbA826B292FDC18daDC"): HookApproval(
        pair="0x16888EE58E706C253b821EAA637ae9b2cFeA255C",
        codehash="0xa57cb656b9262cee1544dfe70bbd760543cd40bff89a2a1c06f29f888d1ed741",
        implementation="0xC997575204290FF7106aB8b2BCFa7e7dEA43D783",
        implementation_codehash="0x67524de60f5a73e3f9ec5d78fea749d2133b323ef1fa340bf151866b0258fb51",
        proxy_kind="eip1967-beacon",
        verification="empirical",
        execution_allowed=False,
        pinned_block=96389826,
        added_date="2026-09-29",
        evidence=EVIDENCE,
    ),
}
validate_allowlist(DEFAULT_HOOKS_ALLOWLIST)

# Pairs (chain, pair address lower case) whose hooks were allowlisted WITHOUT an on-chain swap replay of their own
# (criterion: >= 24 replayed swaps per pair, 0 wei mismatches). Consumers should treat results through them as
# unverified. Remove a pair here once its replay passed.
# 2026-09-29: the replay of all three 2026-09-29 pairs passed (24 of 24 swaps each, 0 mismatches).
REPLAY_PENDING: frozenset[tuple[str, str]] = frozenset()
