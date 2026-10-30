"""LFJ Liquidity Book math: vectors from lfj-gg/joe-v2 v2.2.0 tests plus behavioural checks of the pair logic."""

import pytest

from v4quote.errors import InsufficientState, MathRevert, UnsupportedPool
from v4quote.amm.liquidity_book import FeeParameters, LBPairState, get_price_from_id, get_swap_in, get_swap_out, swap
from v4quote.amm.liquidity_book.math import (
    SCALE,
    get_base,
    get_fee_amount,
    get_fee_amount_from,
    pow128,
    shift_div_round_down,
)
from v4quote.amm.liquidity_book import HookApproval
from v4quote.amm.liquidity_book import hooks_allowlist as hl
from v4quote.amm.liquidity_book.pair import HOOKS_AFTER_SWAP_FLAG, HOOKS_BEFORE_SWAP_FLAG, _simulate_out

ID_ONE = 1 << 23  # price 1


def _to_128(x_e18: int) -> int:
    return shift_div_round_down(x_e18, 128, 10**18)  # PriceHelper.convertDecimalPriceTo128x128


def test_price_at_id_one_is_one():
    for bs in (1, 5, 10, 20, 25, 100):
        assert get_price_from_id(ID_ONE, bs) == SCALE


def test_price_vectors_from_joe_v2_price_helper_test():
    # test/libraries/PriceHelper.t.sol::test_Price
    cases = [
        (8574931, 1, 123456789 * 10**18),
        (8252553, 1, 1234567890000),  # 0.00000123456789e18
        (8392773, 100, 10**36),
        (8389042, 1000, 10**36),
    ]
    for bin_id, bs, price_e18 in cases:
        expected = _to_128(price_e18)
        assert get_price_from_id(bin_id - 1, bs) <= expected
        assert get_price_from_id(bin_id + 1, bs) >= expected


def test_pow_vector_from_joe_v2():
    # test/libraries/math/Uint128x128Math.t.sol::test_Pow: 1.0001^100000 ~= 22015.456048527954 (rel 1e-6)
    res = pow128(((10**18 + 10**14) << 128) // 10**18, 100_000)  # _toUint128x128(1.0001e18)
    res_e18 = (res * 10**18) >> 128
    expected = 22015456048527954000000
    assert abs(res_e18 - expected) * 10**6 <= expected


def test_get_base():
    assert get_base(1) == SCALE + (1 << 128) // 10_000
    assert get_base(25) == SCALE + (25 << 128) // 10_000


def test_price_is_monotonic_and_inverse_symmetric():
    for bs in (1, 10, 100):
        prev = 0
        for d in range(-50, 51, 5):
            p = get_price_from_id(ID_ONE + d, bs)
            assert p > prev
            prev = p
        # p(+d) * p(-d) ~= 1 in 128.128 (within rounding of the two inversions)
        up, down = get_price_from_id(ID_ONE + 777, bs), get_price_from_id(ID_ONE - 777, bs)
        assert abs(((up * down) >> 128) - SCALE) < SCALE >> 100


def test_fee_helpers():
    fee = 3 * 10**15  # 0.3%
    assert get_fee_amount_from(10**18, fee) == 3 * 10**15
    assert get_fee_amount_from(1, fee) == 1  # rounds up
    # fee added on top: amount * f / (1 - f), rounded up
    amount = 997 * 10**15
    assert get_fee_amount(amount, fee) == (amount * fee + (10**18 - fee) - 1) // (10**18 - fee)
    with pytest.raises(MathRevert):
        get_fee_amount(1, 10**17 + 1)  # > MAX_FEE


def _params(**kw) -> FeeParameters:
    base = dict(
        base_factor=5000, filter_period=30, decay_period=600, reduction_factor=5000,
        variable_fee_control=40000, protocol_share=1000, max_volatility_accumulator=350000,
        volatility_accumulator=0, volatility_reference=0, id_reference=ID_ONE, time_of_last_update=1_000,
    )
    base.update(kw)
    return FeeParameters(**base)


def test_volatility_accumulator_filter_and_decay():
    p = _params(volatility_accumulator=100_000, volatility_reference=20_000, id_reference=ID_ONE - 3)
    # dt < filter period: references untouched, only the timestamp moves
    q = p.update_references(ID_ONE, 1_010)
    assert (q.id_reference, q.volatility_reference, q.time_of_last_update) == (ID_ONE - 3, 20_000, 1_010)
    # filter <= dt < decay: idRef := activeId, volRef := volAcc * R / 10000
    q = p.update_references(ID_ONE, 1_100)
    assert (q.id_reference, q.volatility_reference) == (ID_ONE, 50_000)
    # dt >= decay period: volRef := 0
    q = p.update_references(ID_ONE, 1_000 + 600)
    assert q.volatility_reference == 0
    # accumulator: volRef + |id - idRef| * 10000, capped
    assert p.update_volatility_accumulator(ID_ONE).volatility_accumulator == 20_000 + 3 * 10_000
    assert p.update_volatility_accumulator(ID_ONE + 100).volatility_accumulator == 350_000
    with pytest.raises(MathRevert):
        p.update_references(ID_ONE, 999)  # time going backwards


def test_total_fee_formula():
    p = _params(volatility_accumulator=12_345)
    bs = 25
    base = 5000 * bs * 10**10
    var = ((12_345 * bs) ** 2 * 40000 + 99) // 100
    assert p.total_fee(bs) == base + var


PAIR_ADDR = "0x000000000000000000000000000000000000005a"


def _pair(bins, active=ID_ONE, bs=10, lo=None, hi=None, hooks=0, version="2.2", hook_facts=None, address=PAIR_ADDR,
          **kw):
    ids = list(bins)
    return LBPairState(
        address=address, token_x="X", token_y="Y", bin_step=bs, active_id=active,
        params=_params(**kw), bins=bins,
        loaded_range=(lo if lo is not None else min(ids) - 5, hi if hi is not None else max(ids) + 5),
        exhausted_below=lo is None, exhausted_above=hi is None, version=version, hooks_parameters=hooks,
        chain="avalanche", **(hook_facts or {}),
    )


def _ladder(n=10, x=10**20, y=10**20):
    """Y liquidity below the active bin, X above, both in the active bin."""
    bins = {ID_ONE: (x, y)}
    for k in range(1, n + 1):
        bins[ID_ONE - k] = (0, y)
        bins[ID_ONE + k] = (x, 0)
    return bins


def test_swap_out_crosses_bins_and_matches_swap():
    pair = _pair(_ladder())
    ts = 2_000
    for swap_for_y in (True, False):
        for amt in (10**12, 10**20, 5 * 10**20):
            left, out, fee = get_swap_out(pair, amt, swap_for_y, ts)
            r = _simulate_out(pair, amt, swap_for_y, ts, {})
            assert left == 0 and out > 0 and fee > 0
            if amt > 10**20:  # more than one bin's worth of liquidity
                assert r.bins_crossed >= 5
            res, new_state = swap(pair, amt, swap_for_y, ts)
            assert (res.amount_out, res.fee) == (out, fee)
            # bins conserve value: sum of output-token reserves drops by exactly amount_out
            out_idx = 1 if swap_for_y else 0
            before = sum(b[out_idx] for b in pair.bins.values())
            after = sum(b[out_idx] for b in new_state.bins.values())
            assert before - after == out


def test_swap_in_roundtrip():
    pair = _pair(_ladder())
    ts = 2_000
    for swap_for_y in (True, False):
        for want in (10**9, 10**19, 3 * 10**20):
            amount_in, out_left, fee = get_swap_in(pair, want, swap_for_y, ts)
            assert out_left == 0
            left, out, _ = get_swap_out(pair, amount_in, swap_for_y, ts)
            assert left == 0 and out >= want


def test_insufficient_liquidity_behaviour():
    pair = _pair(_ladder(n=2))
    left, out, _ = get_swap_out(pair, 10**30, True, 2_000)
    assert left > 0  # getSwapOut reports unfilled input
    with pytest.raises(MathRevert):
        swap(pair, 10**30, True, 2_000)  # swap reverts (OutOfLiquidity)


def test_timestamp_changes_variable_fee():
    # volatility reference decays with time -> different fee for the same trade
    pair = _pair(_ladder(), volatility_accumulator=200_000, volatility_reference=150_000, id_reference=ID_ONE)
    fee_soon = get_swap_out(pair, 10**19, True, 1_010)[2]  # dt < filter: volRef stays 150000
    fee_later = get_swap_out(pair, 10**19, True, 5_000)[2]  # dt >= decay: volRef -> 0
    assert fee_soon > fee_later


def test_unloaded_bins_raise():
    bins = _ladder(n=3)
    pair = _pair(bins, lo=ID_ONE - 3, hi=ID_ONE + 3)
    with pytest.raises(InsufficientState):
        get_swap_out(pair, 10**30, True, 2_000)


HOOK = 0x1234 << 20
HOOK_ADDR = "0x" + format(HOOK, "040x")
H_A, H_B, H_C = "0x" + "aa" * 32, "0x" + "bb" * 32, "0x" + "cc" * 32
IMPL = "0x00000000000000000000000000000000000000Ab"
FACTS = dict(hooks_codehash=H_A, hooks_implementation=IMPL, hooks_implementation_codehash=H_B,
             hooks_extra_parameters=0, hooks_linked=True)
APPROVAL = HookApproval(H_A.upper().replace("0X", "0x"), IMPL.lower(), H_B, proxy_kind="lfj-immutable-clone",
                        pair=PAIR_ADDR.lower())
ALLOW = {("avalanche", HOOK_ADDR): APPROVAL}


def test_hooked_pool_rejected_by_default_and_unknown_hooks():
    pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts=FACTS)
    with pytest.raises(UnsupportedPool):
        get_swap_out(pair, 10**18, True, 2_000)  # default allowlist is empty
    with pytest.raises(UnsupportedPool):
        get_swap_out(pair, 10**18, True, 2_000, hooks_allowlist={("monad", HOOK_ADDR): APPROVAL})
    assert get_swap_out(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)[1] > 0


def test_hook_flags_outside_0x151_rejected_even_if_allowlisted():
    for extra in (HOOKS_AFTER_SWAP_FLAG, 1 << 162, 1 << 163, 1 << 165, 1 << 169):  # AFTER_SWAP, flash loans, ...
        pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | extra | HOOK, hook_facts=FACTS)
        with pytest.raises(UnsupportedPool):
            get_swap_out(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)
    ok = _pair(_ladder(), hooks=(0x151 << 160) | HOOK, hook_facts=FACTS)
    assert get_swap_out(ok, 10**18, True, 2_000, hooks_allowlist=ALLOW)[1] > 0


def test_hook_codehash_or_implementation_change_rejected():
    for change in (
        {"hooks_codehash": H_C},
        {"hooks_implementation": "0x00000000000000000000000000000000000000Ac"},
        {"hooks_implementation_codehash": H_C},
        {"hooks_codehash": None},
        {"hooks_extra_parameters": 1},
        {"hooks_linked": False},
    ):
        pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts={**FACTS, **change})
        for fn in (get_swap_out, get_swap_in):
            with pytest.raises(UnsupportedPool):
                fn(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)
        with pytest.raises(UnsupportedPool):
            swap(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)
    # proxy on chain but the allowlist entry is a non-proxy entry (pins no implementation)
    pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts=FACTS)
    with pytest.raises(UnsupportedPool):
        get_swap_out(pair, 10**18, True, 2_000,
                     hooks_allowlist={("avalanche", HOOK_ADDR): HookApproval(H_A, pair=PAIR_ADDR)})


# ------------------------------------------------------------------ LFJ extra rewarder, isLinked
EXTRA_REWARDER = (0x151 << 160) | 0x00000000000000000000000000000000000000E7


def _reasons(**facts):
    pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts={**FACTS, **facts})
    return pair, pair.unsupported_reasons(ALLOW)


def test_lfj_clone_without_extra_rewarder_and_linked_is_supported():
    pair, reasons = _reasons()
    assert reasons == []
    assert get_swap_out(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)[1] > 0


def test_lfj_clone_with_extra_rewarder_is_unsupported():
    pair, reasons = _reasons(hooks_extra_parameters=EXTRA_REWARDER)
    assert len(reasons) == 1
    assert "extra rewarder 0x00000000000000000000000000000000000000e7" in reasons[0]
    assert "flags 0x151" in reasons[0]
    for fn in (get_swap_out, get_swap_in):
        with pytest.raises(UnsupportedPool, match="extra rewarder"):
            fn(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)
    with pytest.raises(UnsupportedPool, match="extra rewarder"):
        swap(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)


def test_lfj_clone_not_linked_is_unsupported():
    pair, reasons = _reasons(hooks_linked=False)
    assert reasons == [f"hook {HOOK_ADDR} is not linked to its pair (isLinked() == false)"]
    with pytest.raises(UnsupportedPool, match="isLinked"):
        get_swap_out(pair, 10**18, True, 2_000, hooks_allowlist=ALLOW)


def test_lfj_clone_with_unknown_link_facts_is_unsupported():
    _, reasons = _reasons(hooks_extra_parameters=None, hooks_linked=None)
    assert len(reasons) == 2
    assert "getExtraHooksParameters() unknown" in reasons[0] and "isLinked() unknown" in reasons[1]


def test_link_facts_not_required_for_plain_eip1967_entries():
    plain = HookApproval(H_A, IMPL, H_B, proxy_kind="eip1967", pair=PAIR_ADDR)
    pair = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK,
                 hook_facts={**FACTS, "hooks_extra_parameters": None, "hooks_linked": None})
    assert pair.unsupported_reasons({("avalanche", HOOK_ADDR): plain}) == []


def test_beacon_entry_requires_the_hooks_own_pair_and_link():
    """Beacon proxies share a codehash; getLBPair() / isLinked() bind the hook."""
    beacon = {("avalanche", HOOK_ADDR): HookApproval(H_A, IMPL, H_B, proxy_kind="eip1967-beacon", pair=PAIR_ADDR)}
    base = {**FACTS, "hooks_extra_parameters": None}
    ok = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts={**base, "hooks_pair": PAIR_ADDR.lower()})
    assert ok.unsupported_reasons(beacon) == []  # extra rewarder is an LFJ fact: not required here
    for facts, needle in (({"hooks_pair": None}, "getLBPair() unknown"),
                          ({"hooks_pair": "0x00000000000000000000000000000000000000Bd"}, "serves pair"),
                          ({"hooks_pair": PAIR_ADDR, "hooks_linked": False}, "isLinked() == false"),
                          ({"hooks_pair": PAIR_ADDR, "hooks_linked": None}, "isLinked() unknown")):
        st = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts={**base, **facts})
        reasons = st.unsupported_reasons(beacon)
        assert any(needle in r for r in reasons), (facts, reasons)
        with pytest.raises(UnsupportedPool):
            get_swap_out(st, 10**18, True, 2_000, hooks_allowlist=beacon)


# ------------------------------------------------------------------ entry bound to its pair
def test_allowlisted_hook_on_another_pair_is_unsupported():
    other = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts=FACTS,
                  address="0x00000000000000000000000000000000000000Bc")
    reasons = other.unsupported_reasons(ALLOW)
    assert reasons == [f"hook {HOOK_ADDR} is allowlisted for pair {APPROVAL.pair}, "
                       f"not for 0x00000000000000000000000000000000000000Bc"]
    with pytest.raises(UnsupportedPool, match="allowlisted for pair"):
        get_swap_out(other, 10**18, True, 2_000, hooks_allowlist=ALLOW)
    # the address comparison is case-insensitive (entry lower case, state checksum-like)
    assert _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts=FACTS).unsupported_reasons(ALLOW) == []


def test_allowlist_rejects_two_hooks_for_one_pair():
    from v4quote.amm.liquidity_book import validate_allowlist

    other_hook = "0x" + format(0x5678 << 20, "040x")
    with pytest.raises(ValueError, match="more than one allowlisted hook"):
        validate_allowlist({("avalanche", HOOK_ADDR): APPROVAL, ("avalanche", other_hook): APPROVAL})
    validate_allowlist({("avalanche", HOOK_ADDR): APPROVAL, ("monad", other_hook): APPROVAL})  # other chain: fine


def test_hook_approval_validation():
    HookApproval(H_A, pair=PAIR_ADDR)  # plain contract
    for bad in (
        dict(codehash=H_A, pair=None),  # no pair
        dict(codehash=H_A, pair="0xpair"),
        dict(codehash="0xaa"),
        dict(codehash=H_A, proxy_kind="eip1967-beacon"),  # proxy without implementation fields
        dict(codehash=H_A, proxy_kind="eip1967-beacon", implementation=IMPL),  # missing impl codehash
        dict(codehash=H_A, proxy_kind="eip1967-beacon", implementation_codehash=H_B),  # missing impl address
        dict(codehash=H_A, implementation=IMPL, implementation_codehash=H_B),  # impl fields without proxy_kind
        dict(codehash=H_A, verification="trusted"),
        dict(codehash=H_A, execution_allowed="no"),
        dict(codehash=H_A, added_date="21.09.2026"),
    ):
        with pytest.raises(ValueError):
            HookApproval(**{"pair": PAIR_ADDR, **bad})


def test_default_allowlist_contents():
    from v4quote.amm.liquidity_book import DEFAULT_HOOKS_ALLOWLIST, validate_allowlist

    validate_allowlist(DEFAULT_HOOKS_ALLOWLIST)
    assert len(DEFAULT_HOOKS_ALLOWLIST) == 9
    for (chain, hook), a in DEFAULT_HOOKS_ALLOWLIST.items():
        assert a.execution_allowed is False
        assert a.proxy_kind and a.implementation and a.implementation_codehash
        assert a.pinned_block and a.added_date in ("2026-09-21", "2026-09-23", "2026-09-29")
        assert a.evidence == hl.EVIDENCE
    by_hook = {h.lower(): a for (_, h), a in DEFAULT_HOOKS_ALLOWLIST.items()}
    # every entry is bound to the pair it was verified with (see the fixtures), checksummed
    from eth_utils import to_checksum_address

    pairs = {h: a.pair for h, a in by_hook.items()}
    assert pairs == {
        "0xba83987a28b1c7514576e15d58c134b8f0e00fa9": "0x864d4e5Ee7318e97483DB7EB0912E09F161516EA",
        "0xd0b3ff65058609743877637e689b30abdb97364a": "0x5E60BC3F7a7303BC4dfE4dc2220bdC90bc04fE22",
        "0x8b0d325fea6f13bb6d62f825137abd4cf6457367": "0x5AFD3EC861f6104af26e8755aBcc1f876de77620",
        "0x138ce0008f7febdcffdf27c0ad156584eca7d5b6": "0x8aC5707f8D4BDe1d771d34c7AfD81c3922b73379",
        "0xe104964852be626ee27762712e4de521066859c9": "0x8573F98175D816d520248B5fACF40D309B1c9ceE",
        "0x72ed0b6acb5c585873b3f644f99fc167c7601256": "0xcEC377285AbF370FDf872625D2742252656d631a",
        # simulation only (allowlisted 2026-09-29)
        "0x90014c5e78bd8a3e2bbbaa8e5a0aee2c1d2131cd": "0x2823299af89285fF1a1abF58DB37cE57006FEf5D",
        "0xdae382f3542dd8a6b42f0f1f71b3f7227a4a3622": "0x883eA72c2A46F7AcB3820855344C43666c6cc5c0",
        "0x32e68c4b8edc6d1f16e99eba826b292fdc18dadc": "0x16888EE58E706C253b821EAA637ae9b2cFeA255C",
    }
    assert all(to_checksum_address(v) == v for v in pairs.values())
    # LFJ Avalanche clones of the verified LBHooksSimpleRewarder: WAVAX/USDC, AUSD/USDC, AUSD/USDt
    lfj_avax = ("0xba83987a28b1c7514576e15d58c134b8f0e00fa9", "0xe104964852be626ee27762712e4de521066859c9",
                "0x72ed0b6acb5c585873b3f644f99fc167c7601256", "0x90014c5e78bd8a3e2bbbaa8e5a0aee2c1d2131cd",
                "0xdae382f3542dd8a6b42f0f1f71b3f7227a4a3622")
    for h in lfj_avax:
        assert by_hook[h].verification == "source-verified"
        assert by_hook[h].implementation_codehash == by_hook[lfj_avax[0]].implementation_codehash
    for h in ("0xd0b3ff65058609743877637e689b30abdb97364a", "0x8b0d325fea6f13bb6d62f825137abd4cf6457367",
              "0x138ce0008f7febdcffdf27c0ad156584eca7d5b6", "0x32e68c4b8edc6d1f16e99eba826b292fdc18dadc"):
        assert by_hook[h].verification == "empirical"
    # Pharaoh: both beacon hooks share the proxy codehash, beacon implementation and its codehash
    ph_old = by_hook["0x138ce0008f7febdcffdf27c0ad156584eca7d5b6"]
    ph_new = by_hook["0x32e68c4b8edc6d1f16e99eba826b292fdc18dadc"]
    assert (ph_new.codehash, ph_new.implementation, ph_new.implementation_codehash, ph_new.proxy_kind) == (
        ph_old.codehash, ph_old.implementation, ph_old.implementation_codehash, "eip1967-beacon")
    # the three 2026-09-29 pairs are simulation-only; their replay passed on 2026-09-29
    from v4quote.amm.liquidity_book import REPLAY_PENDING

    new = {h: a for h, a in by_hook.items() if a.added_date == "2026-09-29"}
    assert len(new) == 3 and all(a.pinned_block == 96389826 for a in new.values())
    assert all(not a.execution_allowed for a in new.values())
    assert not REPLAY_PENDING & {("avalanche", a.pair.lower()) for a in new.values()}


def test_execution_permitted_policy():
    from dataclasses import replace

    from v4quote.amm.liquidity_book import execution_permitted

    hooked = _pair(_ladder(), hooks=HOOKS_BEFORE_SWAP_FLAG | HOOK, hook_facts=FACTS)
    assert get_swap_out(hooked, 10**18, True, 2_000, hooks_allowlist=ALLOW)[1] > 0  # simulation allowed
    assert execution_permitted(hooked, ALLOW) is False  # execution not
    allow_exec = {k: replace(v, execution_allowed=True) for k, v in ALLOW.items()}
    assert execution_permitted(hooked, allow_exec) is True
    assert execution_permitted(hooked) is False  # unknown hook
    assert execution_permitted(_pair(_ladder())) is True  # no hooks at all


def test_swap_emits_per_bin_records():
    pair = _pair(_ladder())
    res, _ = swap(pair, 5 * 10**20, True, 2_000)
    assert len(res.bin_swaps) == res.bins_crossed >= 5
    assert sum(b.amount_out for b in res.bin_swaps) == res.amount_out
    assert sum(b.amount_in + b.protocol_fees for b in res.bin_swaps) == 5 * 10**20
    assert [b.bin_id for b in res.bin_swaps] == sorted((b.bin_id for b in res.bin_swaps), reverse=True)
