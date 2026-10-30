"""LogScanner: rate limits vs range errors, chunk regrowth, reply validation, confirmations, truncation.

Everything runs against an in-memory fake ``eth`` (no network, no real sleeping).
"""

from __future__ import annotations

import pytest
from web3.exceptions import Web3RPCError

from v4quote.chain.logs import (
    BlockNotConfirmed,
    LogScanner,
    LogsReplyInvalid,
    classify_rpc_error,
)
from v4quote.chain.rpc import RpcTransportError

ADDR = "0x" + "11" * 20
OTHER = "0x" + "22" * 20
T0 = b"\xaa" * 32
T1 = b"\xbb" * 32
TX = b"\x01" * 32


def mk_log(block: int, idx: int = 0, *, address: str = ADDR, topics=(T0,), removed: bool = False) -> dict:
    return {"blockNumber": block, "logIndex": idx, "transactionIndex": 0, "transactionHash": TX,
            "address": address, "topics": list(topics), "data": b"", "removed": removed}


def rpc_err(code: int, message: str) -> Web3RPCError:
    err = {"code": code, "message": message}
    return Web3RPCError(repr(err), rpc_response={"jsonrpc": "2.0", "id": 1, "error": err})


class FakeEth:
    """Serves logs from ``chain`` (list of raw logs). ``script`` is a queue of overrides per request:
    an Exception to raise, a list to return verbatim, or None for the honest answer. ``max_span`` makes
    it reject wider ranges with a range error (like a public RPC)."""

    def __init__(self, chain=(), *, head: int = 10_000, max_span: int | None = None, script=()) -> None:
        self.chain = list(chain)
        self.block_number = head
        self.max_span = max_span
        self.script = list(script)
        self.filters: list[dict] = []

    def honest(self, flt: dict) -> list:
        addrs = flt.get("address")
        addrs = None if addrs is None else {a.lower() for a in ([addrs] if isinstance(addrs, str) else addrs)}
        return [lg for lg in self.chain
                if flt["fromBlock"] <= lg["blockNumber"] <= flt["toBlock"]
                and (addrs is None or lg["address"].lower() in addrs)]

    def get_logs(self, flt: dict) -> list:
        self.filters.append(dict(flt))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if step is not None:
                return step(flt) if callable(step) else step
        if self.max_span is not None and flt["toBlock"] - flt["fromBlock"] + 1 > self.max_span:
            raise rpc_err(-32602, "block range too large, max 100 blocks")
        return self.honest(flt)


class FakeW3:
    def __init__(self, eth: FakeEth) -> None:
        self.eth = eth


def scanner(eth: FakeEth, chunk: int = 100, **kw) -> tuple[LogScanner, list[float]]:
    sleeps: list[float] = []
    kw.setdefault("max_retries", 4)
    sc = LogScanner(FakeW3(eth), chunk, min_interval_s=0, sleep=sleeps.append, **kw)
    return sc, sleeps


def spans(eth: FakeEth) -> list[int]:
    return [f["toBlock"] - f["fromBlock"] + 1 for f in eth.filters]


# ---------------------------------------------------------------- (a) classification


@pytest.mark.parametrize("code,msg,status,kind", [
    (-32005, "limit exceeded", None, "rate"),
    (-32005, "request rate limit exceeded", None, "rate"),
    (-32005, "query returned more than 10000 results", None, "range"),  # Infura / thirdweb reuse -32005
    (-32029, "too many requests, request limit exceeded", None, "rate"),
    (429, "Too Many Requests", None, "rate"),
    (-32000, "exceeded the quota usage", None, "rate"),
    (-32000, "compute units per second capacity exceeded", None, "rate"),
    (-32000, "request timeout, rate limit", None, "rate"),
    (None, "HTTPError: 429 Client Error: Too Many Requests", 429, "rate"),
    (-32602, "block range too large", None, "range"),
    (-32000, "too many results: the result count exceeds the limit, please narrow the query block range", None, "range"),
    (-32602, "Query returned more than 20000 results. Try with this block range [0x1, 0x2].", None, "range"),
    (-32602, "Log response size exceeded.", None, "range"),
    (None, "HTTPError: 413 Payload Too Large", 413, "range"),
    (-32000, "execution aborted (timeout = 5s)", None, "range"),
    (-32000, "header not found", None, "other"),
    (-32000, "limit", None, "other"),  # bare "limit" / "exceed" no longer mean "range too large"
    (-32000, "exceeded", None, "other"),
])
def test_classify(code, msg, status, kind):
    assert classify_rpc_error(code, msg, status) == kind


@pytest.mark.parametrize("err", [
    rpc_err(-32005, "limit exceeded"),
    rpc_err(-32029, "rate limited"),
    RpcTransportError("RPC eth_getLogs failed: HTTPError: 429 Client Error: Too Many Requests", status=429),
])
def test_rate_limit_backs_off_without_shrinking(err):
    chain = [mk_log(b) for b in (5, 150, 260)]
    eth = FakeEth(chain, script=[err, err, err])
    sc, sleeps = scanner(eth, 100)
    got = sc.get_logs(ADDR, None, 1, 300)
    assert [lg.block_number for lg in got] == [5, 150, 260]
    assert sc.chunk == 100 and sc.range_errors == 0
    assert sc.rate_limited == 3
    # 3 retries of the same chunk, the first-chunk truncation probe (upper half), then 2 more chunks
    assert spans(eth) == [100] * 4 + [50] + [100] * 2
    assert len(sleeps) == 3 and sleeps[0] < sleeps[1] < sleeps[2]  # exponential (jitter is +-25%)


def test_rate_limit_exhausted_raises_transport_error():
    eth = FakeEth(script=[rpc_err(-32029, "slow down")] * 10)
    sc, sleeps = scanner(eth, 100, max_retries=2)
    with pytest.raises(RpcTransportError) as ei:
        sc.get_logs(ADDR, None, 1, 50)
    assert ei.value.status == 429 and len(eth.filters) == 3 and len(sleeps) == 2


def test_range_error_shrinks_then_grows_back():
    chain = [mk_log(b) for b in range(1, 2001, 7)]
    eth = FakeEth(chain, max_span=100)
    sc, sleeps = scanner(eth, 400, grow_after=3)
    got = sc.get_logs(ADDR, None, 1, 2000)
    assert [lg.block_number for lg in got] == [lg["blockNumber"] for lg in chain]
    assert not sleeps  # range errors are not backed off
    s = spans(eth)
    assert s[:3] == [400, 200, 100]
    assert 200 in s[3:], "chunk never tried to grow back"
    assert max(s[3:]) <= 400


def test_chunk_regrows_to_initial_after_transient_range_errors():
    chain = [mk_log(b) for b in range(1, 3001, 10)]
    too_big = rpc_err(-32000, "too many results")
    eth = FakeEth(chain, script=[too_big, too_big])
    sc, _ = scanner(eth, 400, grow_after=2)
    sc.get_logs(ADDR, None, 1, 3000)
    assert sc.chunk == 400
    assert spans(eth)[:3] == [400, 200, 100]


def test_generic_error_is_retried_not_shrunk():
    eth = FakeEth([mk_log(3)], script=[rpc_err(-32000, "header not found")])
    sc, sleeps = scanner(eth, 100)
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, 10)] == [3]
    assert sc.chunk == 100 and len(sleeps) == 1


# ---------------------------------------------------------------- (b) validation


def test_log_outside_range_is_rejected_and_retried():
    eth = FakeEth([mk_log(5)], script=[[mk_log(5), mk_log(500)]])
    sc, _ = scanner(eth, 100)
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, 100)] == [5]
    assert sc.invalid_replies == 1


def test_persistently_bad_reply_raises():
    eth = FakeEth(script=[[mk_log(500)]] * 10)
    sc, _ = scanner(eth, 100, max_retries=2)
    with pytest.raises(LogsReplyInvalid):
        sc.get_logs(ADDR, None, 1, 100)


def test_wrong_address_is_rejected():
    eth = FakeEth(script=[[mk_log(5, address=OTHER)]] * 10)
    sc, _ = scanner(eth, 100, max_retries=1)
    with pytest.raises(LogsReplyInvalid, match="address"):
        sc.get_logs(ADDR, None, 1, 100)
    # a list filter accepts any of its members
    eth = FakeEth(script=[[mk_log(5, address=OTHER), mk_log(6)]])
    sc, _ = scanner(eth, 100)
    assert len(sc.get_logs([ADDR, OTHER], None, 1, 100)) == 2


def test_topic_filter_is_enforced():
    t0_hex = "0x" + T0.hex()
    eth = FakeEth(script=[[mk_log(5, topics=(T1,))]] * 10)
    sc, _ = scanner(eth, 100, max_retries=1)
    with pytest.raises(LogsReplyInvalid, match="topic"):
        sc.get_logs(ADDR, [t0_hex], 1, 100)
    # None is a wildcard, a nested list is OR, a missing position does not match
    ok = [mk_log(5, 0, topics=(T0, T1)), mk_log(5, 1, topics=(T1, T1))]
    eth = FakeEth(script=[ok])
    sc, _ = scanner(eth, 100)
    assert len(sc.get_logs(ADDR, [None, [t0_hex, "0x" + T1.hex()]], 1, 100)) == 2
    eth = FakeEth(script=[[mk_log(5, topics=(T0,))]] * 10)
    sc, _ = scanner(eth, 100, max_retries=0)
    with pytest.raises(LogsReplyInvalid):
        sc.get_logs(ADDR, [t0_hex, "0x" + T1.hex()], 1, 100)


def test_removed_logs_are_dropped():
    eth = FakeEth(script=[[mk_log(5, 0), mk_log(6, 0, removed=True), mk_log(7, 0)]])
    sc, _ = scanner(eth, 100)
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, 100)] == [5, 7]
    assert sc.removed_dropped == 1


def test_confirmation_depth():
    chain = [mk_log(b) for b in (10, 90, 95, 99)]
    eth = FakeEth(chain, head=100)
    sc, _ = scanner(eth, 1000, confirmations=10)
    with pytest.raises(BlockNotConfirmed):
        sc.get_logs(ADDR, None, 1, 95)
    assert eth.filters == []  # nothing requested
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, None)] == [10, 90]  # None -> head - 10
    assert eth.filters[-1]["toBlock"] == 90


def test_confirmation_depth_includes_boundary():
    eth = FakeEth([mk_log(90)], head=100)
    sc, _ = scanner(eth, 1000, confirmations=10)
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, 90)] == [90]


# ---------------------------------------------------------------- truncation heuristics


def test_reply_at_known_cap_is_split():
    chain = [mk_log(b, i) for b in range(1, 201) for i in range(25)]  # 5000 logs in 200 blocks
    eth = FakeEth(chain)
    sc, _ = scanner(eth, 200)
    got = sc.get_logs(ADDR, None, 1, 200)
    assert len(got) == 5000
    assert spans(eth)[:2] == [200, 100] and sc.truncations == 1


def test_sparse_reply_is_probed_and_split():
    chain = [mk_log(b) for b in range(1, 5001)]  # 1 log per block

    def truncated(flt):  # provider answers "ok" but keeps only the first 3 blocks (Tenderly-like)
        return [lg for lg in chain if flt["fromBlock"] <= lg["blockNumber"] <= flt["fromBlock"] + 2]

    # chunk 1 + its first-chunk probe honest (learns density), chunk 2 cut
    eth = FakeEth(chain, script=[None, None, truncated])
    sc, _ = scanner(eth, 800)
    got = sc.get_logs(ADDR, None, 1, 5000)
    assert [lg.block_number for lg in got] == list(range(1, 5001))
    assert sc.truncations == 1


def test_genuinely_empty_range_passes_probe():
    chain = [mk_log(b) for b in range(1, 801)] + [mk_log(4500)]
    eth = FakeEth(chain)
    sc, _ = scanner(eth, 800)
    got = sc.get_logs(ADDR, None, 1, 4800)
    assert len(got) == 801 and sc.truncations == 0
    assert sc.chunk == 800
    assert len(eth.filters) > 6  # sparse chunks after a dense one were probed, and accepted


def test_truncation_check_can_be_disabled():
    chain = [mk_log(b, i) for b in range(1, 101) for i in range(10)]  # exactly 1000 logs
    eth = FakeEth(chain)
    sc, _ = scanner(eth, 100, truncation_check=False)
    assert len(sc.get_logs(ADDR, None, 1, 100)) == 1000 and len(eth.filters) == 1


# ---------------------------------------------------------------- first chunk, sparse events


def test_first_chunk_truncation_like_tenderly_is_recovered():
    # observed once on Avalanche: 100k blocks, 32,276 logs, Tenderly
    # answered "ok" with 114 of them; 20k-block chunks were complete.
    n_logs, window = 32_276, 100_000
    chain = [mk_log(1 + i * window // n_logs, i) for i in range(n_logs)]

    def tenderly(flt):
        got = [lg for lg in chain if flt["fromBlock"] <= lg["blockNumber"] <= flt["toBlock"]]
        return got[:114] if flt["toBlock"] - flt["fromBlock"] + 1 > 20_000 else got

    eth = FakeEth(chain, head=200_000, script=[tenderly] * 10_000)
    sc, _ = scanner(eth, window)
    got = sc.get_logs(ADDR, None, 1, window)
    assert len(got) == n_logs and [lg.log_index for lg in got] == list(range(n_logs))
    assert sc.truncations >= 1 and sc.chunk <= 20_000
    assert spans(eth)[:2] == [100_000, 50_000]  # the very first reply is probed (no density known yet)


def test_sparse_event_truncation_on_large_chunk_is_detected():
    # 1 event per 500 blocks: expected 40 per 20k chunk, below the old density heuristic's floor of 100
    chain = [mk_log(b) for b in range(500, 200_001, 500)]

    def cut(flt):
        return [lg for lg in chain if flt["fromBlock"] <= lg["blockNumber"] <= flt["toBlock"]][:3]

    eth = FakeEth(chain, head=300_000, script=[None] * 4 + [cut])  # chunk 3 is cut, after density is known
    sc, _ = scanner(eth, 20_000)
    got = sc.get_logs(ADDR, None, 1, 200_000)
    assert [lg.block_number for lg in got] == list(range(500, 200_001, 500))
    assert sc.truncations == 1


def test_probe_overhead():
    chain = [mk_log(b) for b in range(1, 100_001, 3)]
    # small chunks: only the first chunk of the filter is probed
    eth = FakeEth(chain, head=200_000)
    sc, _ = scanner(eth, 100)
    sc.get_logs(ADDR, None, 1, 1000)
    assert len(eth.filters) == 10 + 1 and sc.probes == 1
    # a second filter gets its own first-chunk probe; the first one does not get another
    sc.get_logs(ADDR, [None], 1001, 2000)
    sc.get_logs(ADDR, None, 2001, 3000)
    assert sc.probes == 2
    # chunks above probe_above: every chunk is probed (+1 request of half the span each)
    eth = FakeEth(chain, head=200_000)
    sc, _ = scanner(eth, 20_000, probe_above=10_000)
    sc.get_logs(ADDR, None, 1, 100_000)
    assert spans(eth) == [20_000, 10_000] * 5 and sc.probes == 5


def test_cross_check_recovers_truncation_the_probe_misses():
    chain = [mk_log(b) for b in range(1, 4001)]

    def keeps_last_half(flt):  # cut from the other end: the upper-half probe is a subset of the reply
        span = flt["toBlock"] - flt["fromBlock"] + 1
        lo = flt["fromBlock"] + (span // 2 if span > 1000 else 0)
        return [lg for lg in chain if lo <= lg["blockNumber"] <= flt["toBlock"]]

    eth = FakeEth(chain, script=[keeps_last_half] * 1000)
    ref, _ = scanner(FakeEth(chain), 500)
    sc, _ = scanner(eth, 2000, cross_check=ref)
    got = sc.get_logs(ADDR, None, 1, 4000)
    assert [lg.block_number for lg in got] == list(range(1, 4001))
    assert sc.truncations >= 1 and sc.cross_checked >= 1 and sc.chunk <= 1000


def test_cross_check_mismatch_raises():
    from v4quote.chain.logs import LogsCrossCheckMismatch

    eth = FakeEth([mk_log(5), mk_log(7)])
    ref, _ = scanner(FakeEth([mk_log(5)]), 100)
    sc, _ = scanner(eth, 100, cross_check=ref)
    with pytest.raises(LogsCrossCheckMismatch, match="1 missing there"):
        sc.get_logs(ADDR, None, 1, 100)


# ---------------------------------------------------------------- gate scans must be cross-checked


def test_scan_ranges_carry_cross_checked_flag_and_gate_scan_requires_it():
    from v4quote.chain.logs import CrossCheckRequired

    chain = [mk_log(b) for b in (5, 150, 250)]
    plain, _ = scanner(FakeEth(chain), 100)
    res = plain.scan(ADDR, None, 1, 300)
    assert [(r.from_block, r.to_block, r.cross_checked) for r in res.ranges] == [
        (1, 100, False), (101, 200, False), (201, 300, False)]
    assert [lg.block_number for lg in res.logs] == [5, 150, 250] and not res.cross_checked
    assert res.unchecked_ranges() == [(1, 100), (101, 200), (201, 300)]
    with pytest.raises(CrossCheckRequired):
        plain.gate_scan(ADDR, None, 1, 300)  # nothing is read without a second endpoint
    with pytest.raises(CrossCheckRequired):
        scanner(FakeEth(chain), 100, require_cross_check=True)
    ref, _ = scanner(FakeEth(chain), 100)
    sc, _ = scanner(FakeEth(chain), 100, cross_check=ref, require_cross_check=True)
    res = sc.gate_scan(ADDR, None, 1, 300)
    assert res.cross_checked and res.unchecked_ranges() == [] and len(res.logs) == 3
    assert all(r.cross_checked for r in sc.scan(ADDR, None, 1, 300).ranges)


def test_for_gate_always_cross_checks(clean_env):
    from v4quote.chain.logs import CrossCheckRequired
    from v4quote.chain.rpc import RpcNotConfigured

    clean_env.setenv("MONAD_LOGS_RPC_URL", "http://127.0.0.1:9/logs")
    with pytest.raises(RpcNotConfigured):
        LogScanner.for_gate("monad")  # no second endpoint -> no gate scanner
    with pytest.raises(CrossCheckRequired):
        LogScanner.for_gate("monad", cross_check=False)
    clean_env.setenv("MONAD_LOGS_CROSSCHECK_RPC_URL", "http://127.0.0.1:9/other")
    sc = LogScanner.for_gate("monad", min_interval_s=0)
    assert sc.require_cross_check and sc.cross_check is not None
    assert sc.confirmations == sc.cross_check.confirmations == 3


def test_lagging_cross_check_endpoint_is_waited_for_not_a_mismatch():
    from v4quote.chain.logs import CrossCheckEndpointLagging

    chain = [mk_log(b) for b in (5, 95)]
    ref_eth = FakeEth([lg for lg in chain if lg["blockNumber"] <= 50], head=60)  # 40 blocks behind

    heads = iter([60, 80, 110])  # catches up (and gets the logs) after two waits

    class Catching(FakeEth):
        @property
        def block_number(self):
            h = next(heads)
            if h >= 100:
                self.chain = list(chain)
            return h

        @block_number.setter
        def block_number(self, v):
            pass

    ref, _ = scanner(Catching([lg for lg in chain if lg["blockNumber"] <= 50]), 100, confirmations=10)
    sc, sleeps = scanner(FakeEth(chain, head=110), 100, confirmations=10, cross_check=ref)
    res = sc.scan(ADDR, None, 1, 100)
    assert res.cross_checked and [lg.block_number for lg in res.logs] == [5, 95]
    assert sc.cross_lag_waits == 2 and len(sleeps) == 2  # backoff, no LogsCrossCheckMismatch

    ref2, _ = scanner(ref_eth, 100, confirmations=10)  # never catches up
    sc2, _ = scanner(FakeEth(chain, head=110), 100, confirmations=10, cross_check=ref2, max_retries=2)
    with pytest.raises(CrossCheckEndpointLagging):
        sc2.get_logs(ADDR, None, 1, 100)
    assert ref_eth.filters == []  # the lagging endpoint's logs were never compared


# ---------------------------------------------------------------- for_chain defaults


@pytest.fixture
def clean_env(monkeypatch):
    for chain in ("MONAD", "AVALANCHE"):
        for suffix in ("_RPC_URL", "_LOGS_RPC_URL", "_LOGS_RPC_URL_CHUNK", "_LOGS_CROSSCHECK_RPC_URL",
                       "_LOGS_CROSSCHECK_RPC_URL_CHUNK"):
            monkeypatch.delenv(chain + suffix, raising=False)
    return monkeypatch


def test_for_chain_confirmation_defaults(clean_env):
    clean_env.setenv("MONAD_RPC_URL", "http://127.0.0.1:9/monad")
    clean_env.setenv("AVALANCHE_RPC_URL", "http://127.0.0.1:9/avax")
    m = LogScanner.for_chain("monad")
    a = LogScanner.for_chain("avalanche")
    assert m.confirmations == 3 and a.confirmations == 2
    assert (m.chunk, a.chunk) == (100, 2048)
    assert LogScanner.for_chain("monad", confirmations=0).confirmations == 0  # explicit override
    assert LogScanner.for_chain("avalanche", confirmations=12).confirmations == 12


def test_for_chain_dedicated_endpoint_chunk_and_cross_check(clean_env):
    from v4quote.chain.rpc import RpcNotConfigured

    clean_env.setenv("AVALANCHE_LOGS_RPC_URL", "http://127.0.0.1:9/logs")
    sc = LogScanner.for_chain("avalanche")
    assert sc.chunk == 20_000 and sc.cross_check is None
    with pytest.raises(RpcNotConfigured):
        LogScanner.for_chain("avalanche", cross_check=True)  # no second endpoint
    clean_env.setenv("AVALANCHE_RPC_URL", "http://127.0.0.1:9/logs")
    with pytest.raises(RpcNotConfigured):
        LogScanner.for_chain("avalanche", cross_check=True)  # the same endpoint is not a cross-check
    clean_env.setenv("AVALANCHE_RPC_URL", "http://127.0.0.1:9/public")
    sc = LogScanner.for_chain("avalanche", cross_check=True, min_interval_s=0)
    assert sc.cross_check is not None and sc.cross_check.chunk == 2048
    assert sc.cross_check.confirmations == sc.confirmations == 2 and sc.cross_check.min_interval_s == 0  # N-B
    clean_env.setenv("AVALANCHE_LOGS_RPC_URL_CHUNK", "5000")
    assert LogScanner.for_chain("avalanche").chunk == 5000


# ---------------------------------------------------------------- masking


SHORT_Q_URL = "https://node.example/rpc?v=1&apikey=QUERYSECRET999"


@pytest.mark.parametrize("code,message,kind", [
    (-32000, "rate limit exceeded: 10 requests/s, 1000 per minute", "rate"),
    (-32005, "limit exceeded, retry in 1s", "rate"),
    (-32005, "query exceeds 10000 results", "range"),  # was "<rpc>0000 results" -> misread as rate
])
def test_short_query_value_does_not_corrupt_classification(code, message, kind):
    from v4quote.chain.rpc import mask_rpc_error, mask_secret

    assert mask_secret(message, SHORT_Q_URL) == message
    assert "QUERYSECRET999" not in mask_secret("key=QUERYSECRET999", SHORT_Q_URL)
    err = mask_rpc_error(rpc_err(code, message), SHORT_Q_URL)  # what MaskedHTTPProvider does to the reply
    eth = FakeEth([mk_log(5)], script=[err])
    sc, _ = scanner(eth, 100)
    assert [lg.block_number for lg in sc.get_logs(ADDR, None, 1, 100)] == [5]
    if kind == "rate":
        assert sc.rate_limited == 1 and sc.range_errors == 0 and sc.chunk == 100
    else:
        assert sc.range_errors == 1 and sc.rate_limited == 0 and sc.chunk == 50


@pytest.mark.parametrize("logger_name", ["urllib3.connectionpool", "web3.providers.HTTPProvider"])
def test_exc_info_and_stack_info_never_print_the_key(logger_name, caplog):
    import io
    import logging

    from v4quote.chain.rpc import web3_from_url

    url = "https://node.example/v2/PATHSECRET_abc123?apikey=QUERYSECRET999"
    web3_from_url(url)  # registers the URL and installs the redaction filter (no network)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    lg = logging.getLogger(logger_name)
    lg.addHandler(handler)
    try:
        try:
            try:
                raise OSError(f"connection reset by {url}")
            except OSError as inner:
                raise ConnectionError("Max retries exceeded with url: /v2/PATHSECRET_abc123?apikey=QUERYSECRET999") from inner
        except ConnectionError as exc:
            with caplog.at_level(logging.WARNING, logger=logger_name):
                lg.warning("Retrying (%r) after connection broken by %r", "Retry(total=2)", exc,
                           exc_info=True, stack_info=True)
    finally:
        lg.removeHandler(handler)
    for out in (buf.getvalue(), caplog.text):
        assert "PATHSECRET_abc123" not in out and "QUERYSECRET999" not in out and "node.example" not in out
        assert "Traceback" in out and "Stack (most recent call last)" in out and "<rpc>" in out
    rec = caplog.records[-1]
    assert rec.exc_info is None and rec.stack_info is None
