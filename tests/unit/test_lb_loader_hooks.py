"""load_lb_pair on a hooked LB v2.2 pair (in-memory chain): extra rewarder, isLinked, pair binding.

For an LFJ ImmutableClone hook the loader reads ``getExtraHooksParameters()`` and ``isLinked()`` in the same
Multicall3 as the bins (same block), as optional calls: a revert makes the fact unknown (None), never fails the load.
``unsupported_reasons`` then rejects the pool while an extra rewarder is set, the hook is unlinked or a fact is unknown.
The real ``v4quote.chain.multicall`` encode/decode path runs end to end (tests/_fake_chain.py).
"""

from __future__ import annotations

import pytest

from v4quote.amm.liquidity_book import HookApproval, get_swap_out
from v4quote.chain.loaders import (
    EIP1967_BEACON_SLOT,
    EIP1967_IMPLEMENTATION_SLOT,
    LFJ_IMMUTABLE_CLONE_PREFIX,
    PROXY_EIP1967,
    PROXY_EIP1967_BEACON,
    PROXY_LFJ_CLONE,
    _codehash,
    load_lb_pair,
)
from v4quote.chain.multicall import multicall
from v4quote.errors import UnsupportedPool
from tests._fake_chain import FakeW3

BLOCK = 1_000
PAIR = "0x00000000000000000000000000000000000000a1"
HOOK = "0x00000000000000000000000000000000000000b2"
IMPL = "0x00000000000000000000000000000000000000c3"
EXTRA = "0x00000000000000000000000000000000000000e5"
ACTIVE = 1 << 23
FLAGS = 0x151
HOOK_CODE = LFJ_IMMUTABLE_CLONE_PREFIX + bytes.fromhex(IMPL[2:]) + bytes.fromhex("5af43d82803e903d91603c57fd5bf3") \
    + bytes.fromhex(PAIR[2:])
IMPL_CODE = bytes.fromhex("6080604052")
PROXY_CODE = bytes.fromhex("60806040")


def _bytes32(x: int) -> bytes:
    return x.to_bytes(32, "big")


def _chain(*, extra: int | None = 0, linked: bool | None = True, clone: bool = True) -> FakeW3:
    w3 = FakeW3(BLOCK)
    e = w3.eth
    e.returns(PAIR, "getTokenX()", ("address",), "0x" + "11" * 20)
    e.returns(PAIR, "getTokenY()", ("address",), "0x" + "22" * 20)
    e.returns(PAIR, "getBinStep()", ("uint16",), 10)
    e.returns(PAIR, "getActiveId()", ("uint24",), ACTIVE)
    e.returns(PAIR, "getStaticFeeParameters()", ("uint16",) * 4 + ("uint24", "uint16", "uint24"),
              8_000, 30, 600, 5_000, 40_000, 1_000, 350_000)
    e.returns(PAIR, "getVariableFeeParameters()", ("uint24", "uint24", "uint24", "uint40"), 0, 0, ACTIVE, 0)
    e.returns(PAIR, "getLBHooksParameters()", ("bytes32",), _bytes32((FLAGS << 160) | int(HOOK, 16)))
    bins = {ACTIVE - 1: (0, 10**20), ACTIVE: (10**20, 10**20), ACTIVE + 1: (10**20, 0)}
    e.on(PAIR, "getBin(uint24)", lambda i: (("uint128", "uint128"), bins.get(i, (0, 0))))
    e.on(PAIR, "getNextNonEmptyBin(bool,uint24)", lambda for_y, i: (("uint24",), (0 if for_y else (1 << 24) - 1,)))
    if clone:
        e.code[HOOK.lower()] = HOOK_CODE
    else:  # EIP-1967 proxy: not an LFJ clone, so no LFJ link reads are expected
        e.code[HOOK.lower()] = PROXY_CODE
        e.set_slot(HOOK, EIP1967_IMPLEMENTATION_SLOT, int(IMPL, 16))
    e.code[IMPL.lower()] = IMPL_CODE
    e.on(HOOK, "getExtraHooksParameters()", lambda: None if extra is None else (("bytes32",), (_bytes32(extra),)))
    e.on(HOOK, "isLinked()", lambda: None if linked is None else (("bool",), (linked,)))
    return w3


def _allow(proxy_kind: str = PROXY_LFJ_CLONE, pair: str = PAIR) -> dict:
    code = HOOK_CODE if proxy_kind == PROXY_LFJ_CLONE else PROXY_CODE
    return {("avalanche", HOOK): HookApproval(_codehash(code), IMPL, _codehash(IMPL_CODE), proxy_kind=proxy_kind,
                                              pair=pair)}


def _load(w3: FakeW3):
    return load_lb_pair(w3, PAIR, BLOCK, bins_each_side=2, chain="avalanche")


def test_link_facts_are_read_in_the_bins_multicall_as_optional_calls():
    w3 = _chain()
    s = _load(w3)
    assert s.hooks_extra_parameters == 0 and s.hooks_linked is True
    assert s.hooks_implementation.lower() == IMPL
    # 2 multicalls: pair parameters, then bins + edges + the two hook reads together (no extra round trip)
    assert len(w3.eth.batches) == 2
    second = w3.eth.batches[1]
    assert second[-2:] == ["getExtraHooksParameters()", "isLinked()"]
    assert second.count("getBin(uint24)") == 5
    flags = w3.eth.allow[-len(second):]
    assert flags == [False] * (len(second) - 2) + [True, True]
    assert s.unsupported_reasons(_allow()) == []
    assert get_swap_out(s, 10**18, True, 2_000, hooks_allowlist=_allow())[1] > 0


def test_extra_rewarder_set_makes_the_pool_unsupported():
    s = _load(_chain(extra=(FLAGS << 160) | int(EXTRA, 16)))
    assert s.hooks_extra_parameters == (FLAGS << 160) | int(EXTRA, 16)
    with pytest.raises(UnsupportedPool, match=f"extra rewarder {EXTRA}"):
        get_swap_out(s, 10**18, True, 2_000, hooks_allowlist=_allow())


def test_unlinked_hook_makes_the_pool_unsupported():
    s = _load(_chain(linked=False))
    assert s.hooks_linked is False
    with pytest.raises(UnsupportedPool, match=r"isLinked\(\) == false"):
        get_swap_out(s, 10**18, True, 2_000, hooks_allowlist=_allow())


def test_reverting_link_reads_load_but_are_unsupported():
    s = _load(_chain(extra=None, linked=None))  # both calls revert: the load itself succeeds
    assert s.hooks_extra_parameters is None and s.hooks_linked is None
    assert s.bins  # bins loaded
    reasons = s.unsupported_reasons(_allow())
    assert any("getExtraHooksParameters() unknown" in r for r in reasons)
    assert any("isLinked() unknown" in r for r in reasons)


def test_non_lfj_proxy_hook_gets_no_lfj_reads():
    w3 = _chain(clone=False)
    s = _load(w3)
    assert all(sig not in ("getExtraHooksParameters()", "isLinked()") for _, sig, _ in w3.eth.calls)
    assert s.hooks_extra_parameters is None and s.hooks_linked is None
    assert s.unsupported_reasons(_allow(PROXY_EIP1967)) == []


def test_pair_binding_checked_on_loaded_state():
    s = _load(_chain())
    other = "0x00000000000000000000000000000000000000f6"
    reasons = s.unsupported_reasons(_allow(pair=other))
    assert reasons and "allowlisted for pair" in reasons[0]


def test_multicall_per_call_allow_failure():
    w3 = _chain(extra=None)
    from v4quote.chain.multicall import Call, CallFailed

    calls = [Call(PAIR, "getBinStep()", out=("uint16",)), Call(HOOK, "getExtraHooksParameters()", out=("bytes32",))]
    assert multicall(w3, calls, BLOCK, allow_failure=[False, True]) == [10, None]
    with pytest.raises(CallFailed):
        multicall(w3, calls, BLOCK)
    with pytest.raises(ValueError):
        multicall(w3, calls, BLOCK, allow_failure=[True])


# ------------------------------------------------------------------ beacon hooks
BEACON = "0x00000000000000000000000000000000000000d4"


@pytest.fixture(autouse=True)
def _beacon_resolution(monkeypatch):
    """resolve_hook reads beacon.implementation() with a plain eth_call, which the fake chain (Multicall3 only) does
    not serve: for the beacon tests the hook resolves to a BeaconProxy of IMPL (the slot layout is tested on chain)."""
    import v4quote.chain.loaders as loaders

    real = loaders.resolve_hook

    def resolve(w3, hook, block):
        if w3.eth.storage.get((hook.lower(), EIP1967_BEACON_SLOT)):
            return loaders.HookCodeInfo(_codehash(PROXY_CODE), loaders.to_checksum_address(IMPL), _codehash(IMPL_CODE),
                                        PROXY_EIP1967_BEACON)
        return real(w3, hook, block)

    monkeypatch.setattr(loaders, "resolve_hook", resolve)


def _beacon_chain(*, lb_pair: str | None = PAIR, linked: bool | None = True) -> FakeW3:
    w3 = _chain(clone=False)
    e = w3.eth
    e.set_slot(HOOK, EIP1967_IMPLEMENTATION_SLOT, 0)
    e.set_slot(HOOK, EIP1967_BEACON_SLOT, int(BEACON, 16))
    e.on(HOOK, "getLBPair()", lambda: None if lb_pair is None else (("address",), (lb_pair,)))
    e.on(HOOK, "isLinked()", lambda: None if linked is None else (("bool",), (linked,)))
    return w3


def test_beacon_hook_pair_and_link_are_read_in_the_bins_multicall():
    w3 = _beacon_chain()
    s = _load(w3)
    assert s.hooks_pair.lower() == PAIR and s.hooks_linked is True and s.hooks_extra_parameters is None
    assert len(w3.eth.batches) == 2
    second = w3.eth.batches[1]
    assert second[-2:] == ["getLBPair()", "isLinked()"]
    assert w3.eth.allow[-len(second):] == [False] * (len(second) - 2) + [True, True]
    assert all(sig != "getExtraHooksParameters()" for _, sig, _ in w3.eth.calls)
    assert s.unsupported_reasons(_allow(PROXY_EIP1967_BEACON)) == []


@pytest.mark.parametrize(("kw", "needle"), [
    ({"lb_pair": "0x00000000000000000000000000000000000000f7"}, "serves pair"),
    ({"lb_pair": None}, r"getLBPair\(\) unknown"),
    ({"linked": False}, r"isLinked\(\) == false"),
    ({"linked": None}, r"isLinked\(\) unknown"),
])
def test_beacon_hook_on_another_pair_unlinked_or_unknown_is_unsupported(kw, needle):
    s = _load(_beacon_chain(**kw))
    with pytest.raises(UnsupportedPool, match=needle):
        get_swap_out(s, 10**18, True, 2_000, hooks_allowlist=_allow(PROXY_EIP1967_BEACON))
