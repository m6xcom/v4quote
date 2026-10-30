"""eth_getLogs scanning with range batching, adaptive chunk size, rate-limit backoff and reply validation.

Public RPCs cap ``eth_getLogs`` differently (Avalanche api.avax.network 2048 blocks, Monad
rpc.monad.xyz 100 blocks, rpc1.monad.xyz ~50k blocks for an address filter). A separate endpoint for
log scanning can be given in ``MONAD_LOGS_RPC_URL`` / ``AVALANCHE_LOGS_RPC_URL``; otherwise the regular
``*_RPC_URL`` endpoint is used with a conservative chunk. URLs are only ever read from the environment
and the Web3 instance is built through ``MaskedHTTPProvider`` (no URL in errors or logs).

Error handling:

* rate limits (HTTP 429, JSON-RPC -32029 / 429, -32005 without a range/size message, "rate limit" and
  similar messages) are classified FIRST and retried with exponential backoff + jitter; the chunk is
  left alone;
* range / result-size errors halve the chunk; after ``grow_after`` consecutive successes the chunk
  doubles again, up to the initial value;
* every reply is validated: block number inside the requested range, address and topics match the
  filter. ``removed=true`` logs are dropped. A reply that fails validation is retried and, if it keeps
  failing, raises ``LogsReplyInvalid``;
* silent truncation (a reply that is "ok" but incomplete: Tenderly returned 114 of 32,276 logs for a
  100k-block range at head in one measurement) is
  caught by: a result count equal to a well-known provider cap; and one probe request for the upper
  half of the range, which is ALWAYS sent for the first chunk of a filter and for every chunk wider
  than ``probe_above`` blocks, and otherwise only when the reply is far sparser than the density seen
  so far for that filter. Any log in the probe missing from the full reply means the reply was cut:
  the chunk is halved and the range re-read;
* ``cross_check=<LogScanner on another endpoint>`` re-reads every accepted chunk there and compares
  the (block, logIndex) sets. Before comparing, the second endpoint's confirmed head must have reached the
  chunk: a lagging endpoint is waited for with backoff (``CrossCheckEndpointLagging`` only once the retries
  are exhausted), never reported as a mismatch. ``for_chain(..., cross_check=True)`` gives the second scanner
  the same ``confirmations``;
* verified backfills (data a decision depends on) MUST be cross-checked: ``LogScanner.for_gate(chain)``
  builds a scanner with ``require_cross_check=True`` (it refuses to be built, or to scan, without a second
  endpoint) and ``gate_scan()`` returns a ``ScanResult``. Every scan result is a sequence of ``ScannedRange``
  (``iter_ranges`` / ``scan``), each carrying ``cross_checked: bool``, so a consumer can store it per range
  and reject unchecked ranges;
* ``confirmations=N`` refuses to scan past ``head - N``; ``for_chain`` defaults it per chain
  (``DEFAULT_CONFIRMATIONS``).

Probing cost: +1 request (covering half the span) per chunk wider than ``probe_above`` and for the
first chunk of every filter. For R blocks with chunk C: R/C extra requests when C > probe_above
(2x requests, 1.5x blocks read), otherwise 1 extra request per filter per scanner plus the
density-triggered probes. A cross-check adds the second endpoint's own requests for the same R blocks.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from eth_utils import to_checksum_address
from web3 import Web3
from web3.exceptions import Web3RPCError

from v4quote.chain.rpc import CHAINS, RpcNotConfigured, RpcTransportError, mask_rpc_error, web3_from_url

log = logging.getLogger(__name__)  # messages never contain the endpoint URL

LOGS_ENV = {"monad": "MONAD_LOGS_RPC_URL", "avalanche": "AVALANCHE_LOGS_RPC_URL"}
DEFAULT_CHUNK = {"monad": 100, "avalanche": 2048}
# Default chunk for a dedicated ``*_LOGS_RPC_URL`` (override with ``*_LOGS_RPC_URL_CHUNK``). Consistency
# runs: 20k blocks matched the reference on Tenderly (Avalanche; 100k was silently cut to
# 114 of 32,276 logs) and on originstake (Monad); rpc1.monad.xyz also passed at 50k. 20k is proven on all.
LOGS_CHUNK_WITH_DEDICATED_ENDPOINT = 20_000
# Optional second endpoint for the cross-check; falls back to the chain's regular ``*_RPC_URL``.
CROSSCHECK_ENV = {"monad": "MONAD_LOGS_CROSSCHECK_RPC_URL", "avalanche": "AVALANCHE_LOGS_CROSSCHECK_RPC_URL"}
# Reorg margin used by ``LogScanner.for_chain`` unless ``confirmations=`` is passed explicitly.
DEFAULT_CONFIRMATIONS = {"monad": 3, "avalanche": 2}
# Chunks wider than this are always probed for silent truncation, whatever the observed density.
PROBE_ABOVE_BLOCKS = 10_000

# Throughput limits. Checked before the range patterns.
_RATE_LIMIT_CODES = (-32029, 429)
_AMBIGUOUS_CODES = (-32005,)  # EIP-1474 "limit exceeded": Infura/thirdweb use it for both kinds
_RATE_LIMIT_MSGS = (
    "rate limit", "rate-limit", "ratelimit", "too many request", "exceeded the quota", "quota exceeded",
    "throttl", "request limit", "requests per second", "rps limit", "compute units", "capacity exceeded",
    "over capacity", "credits",
)
# "Range / result set too large": split and retry. Deliberately no bare "limit" / "exceed" / "size".
_RANGE_MSGS = (
    "block range", "range too", "range is too", "range exceeds", "range limit", "blocks range",
    "too many results", "too many blocks", "too many logs", "too large", "response size", "result count",
    "results limit", "more than", "10000 results", "narrow the query", "query timeout", "timed out",
    "timeout", "try with this block range",
)
_RANGE_CODES_HTTP = (413,)

# Result counts at which some providers silently cut the reply instead of erroring.
TRUNCATION_CAPS = frozenset({1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000})


@dataclass(frozen=True)
class Log:
    block_number: int
    tx_hash: str
    tx_index: int
    log_index: int
    address: str
    topics: tuple[bytes, ...]
    data: bytes

    @property
    def topic0(self) -> bytes:
        return self.topics[0] if self.topics else b""


class LogsReplyInvalid(RuntimeError):
    """The provider kept returning logs that do not match the request (wrong block, address or topic)."""


class LogsCrossCheckMismatch(RuntimeError):
    """The cross-check endpoint returned logs this endpoint does not have (or vice versa) for one range."""


class BlockNotConfirmed(ValueError):
    """``to_block`` is newer than ``head - confirmations``."""


class CrossCheckRequired(RuntimeError):
    """A gate scan was requested on a scanner without a cross-check endpoint (or a range was not checked)."""


class CrossCheckEndpointLagging(RuntimeError):
    """The cross-check endpoint's confirmed head stayed below the chunk after every backoff retry."""


@dataclass(frozen=True)
class ScannedRange:
    """One accepted block range ``[from_block, to_block]`` of a scan with its logs (block order).

    ``cross_checked`` is True only if these exact blocks were re-read on the second endpoint and matched;
    a consumer can store it per range and reject ranges where it is False.
    """

    from_block: int
    to_block: int
    logs: tuple[Log, ...]
    cross_checked: bool


@dataclass(frozen=True)
class ScanResult:
    """``[from_block, to_block]`` scanned as contiguous ``ranges`` (each with its own ``cross_checked`` flag)."""

    from_block: int
    to_block: int
    ranges: tuple[ScannedRange, ...]

    @property
    def logs(self) -> list[Log]:
        return [lg for r in self.ranges for lg in r.logs]

    @property
    def cross_checked(self) -> bool:
        """True only if every range was cross-checked (an empty scan is not)."""
        return bool(self.ranges) and all(r.cross_checked for r in self.ranges)

    def unchecked_ranges(self) -> list[tuple[int, int]]:
        return [(r.from_block, r.to_block) for r in self.ranges if not r.cross_checked]


class _RangeTooLarge(Exception):
    pass


class _SilentlyTruncated(_RangeTooLarge):
    """An "ok" reply was proven incomplete (probe or cross-check): the chunk no longer regrows past half."""


class _RateLimited(Exception):
    pass


class _BadReply(Exception):
    pass


def _logs_endpoint(chain: str) -> tuple[str, int]:
    env = LOGS_ENV[chain]
    url = os.environ.get(env, "").strip()
    if url:
        return url, int(os.environ.get(env + "_CHUNK", str(LOGS_CHUNK_WITH_DEDICATED_ENDPOINT)))
    base = os.environ.get(CHAINS[chain].rpc_env, "").strip()
    if not base:
        raise RpcNotConfigured(f"neither {env} nor {CHAINS[chain].rpc_env} is set")
    return base, DEFAULT_CHUNK[chain]


def _crosscheck_endpoint(chain: str, primary_url: str) -> tuple[str, int]:
    env = CROSSCHECK_ENV[chain]
    url = os.environ.get(env, "").strip()
    if url:
        chunk = int(os.environ.get(env + "_CHUNK", str(LOGS_CHUNK_WITH_DEDICATED_ENDPOINT)))
    else:
        url, chunk = os.environ.get(CHAINS[chain].rpc_env, "").strip(), DEFAULT_CHUNK[chain]
    if not url or url == primary_url:
        raise RpcNotConfigured(
            f"cross-check needs a second endpoint: set {env} or {CHAINS[chain].rpc_env} "
            f"to an endpoint other than the log-scanning one")
    return url, chunk


def logs_web3(chain: str) -> tuple[Web3, int]:
    """(Web3 for log scanning, max blocks per request) for ``chain``; URL only from the environment."""
    url, chunk = _logs_endpoint(chain)
    return web3_from_url(url), chunk


def _to_bytes(v) -> bytes:
    if isinstance(v, str):
        s = v[2:] if v[:2] in ("0x", "0X") else v
        return bytes.fromhex(s)
    return bytes(v)


def _to_log(raw) -> Log:
    return Log(
        block_number=int(raw["blockNumber"]),
        tx_hash="0x" + _to_bytes(raw["transactionHash"]).hex(),
        tx_index=int(raw["transactionIndex"]),
        log_index=int(raw["logIndex"]),
        address=to_checksum_address(raw["address"]),
        topics=tuple(_to_bytes(t) for t in raw["topics"]),
        data=_to_bytes(raw["data"]),
    )


def _rpc_error_parts(exc: Web3RPCError) -> tuple[int | None, str]:
    code, msg = None, ""
    resp = getattr(exc, "rpc_response", None)
    err = resp.get("error") if isinstance(resp, dict) else None
    if isinstance(err, dict):
        c = err.get("code")
        code = c if isinstance(c, int) else None
        msg = f"{err.get('message', '')} {err.get('data', '')}"
    return code, (msg + " " + str(exc)).lower()


def classify_rpc_error(code: int | None, msg: str, status: int | None = None) -> str:
    """'rate' | 'range' | 'other' for a failed eth_getLogs. Rate limits are recognised first."""
    msg = msg.lower()
    ranged = any(s in msg for s in _RANGE_MSGS)
    if status == 429 or code in _RATE_LIMIT_CODES:
        return "rate"
    if code in _AMBIGUOUS_CODES:
        return "range" if ranged else "rate"
    if any(s in msg for s in _RATE_LIMIT_MSGS):
        return "rate"
    if status in _RANGE_CODES_HTTP or ranged:
        return "range"
    return "other"


def _norm_topic_spec(topics: Sequence | None) -> list[frozenset[bytes] | None] | None:
    if topics is None:
        return None
    spec: list[frozenset[bytes] | None] = []
    for t in topics:
        if t is None:
            spec.append(None)
        elif isinstance(t, (list, tuple)):
            spec.append(frozenset(_to_bytes(x) for x in t))
        else:
            spec.append(frozenset({_to_bytes(t)}))
    return spec


class LogScanner:
    """Stateful scanner: adapts the chunk size to the provider and paces requests.

    ``min_interval_s`` is the minimum delay between two requests (rate limit courtesy). Rate limits and
    transport errors are retried with exponential backoff (``backoff_base_s`` .. ``backoff_max_s``);
    range/size errors halve the chunk, ``grow_after`` consecutive successes double it (up to the
    initial chunk). ``confirmations`` > 0 makes ``iter_logs`` refuse ranges newer than ``head - N``
    (``to_block=None`` scans up to that confirmed head).
    """

    def __init__(
        self,
        w3: Web3,
        chunk: int,
        *,
        min_interval_s: float = 0.25,
        max_retries: int = 8,
        confirmations: int = 0,
        grow_after: int = 8,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        truncation_check: bool = True,
        probe_above: int = PROBE_ABOVE_BLOCKS,
        cross_check: "LogScanner | None" = None,
        require_cross_check: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if require_cross_check and cross_check is None:
            raise CrossCheckRequired("require_cross_check=True needs a cross_check scanner on a second endpoint")
        self.w3 = w3
        self.chunk = max(1, chunk)
        self.max_chunk = self.chunk
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.confirmations = max(0, confirmations)
        self.grow_after = max(1, grow_after)
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.truncation_check = truncation_check
        self.probe_above = max(1, probe_above)
        self.cross_check = cross_check
        self.require_cross_check = require_cross_check
        self._sleep = sleep
        self._confirmed_head_seen: int | None = None  # highest confirmed head read so far (monotonic cache)
        self._last = 0.0
        self._ok_streak = 0
        # logs per block over accepted replies (EMA), per filter: one filter's density says nothing of another
        self._density: dict[tuple, float] = {}
        # counters (observability)
        self.requests = 0
        self.rate_limited = 0
        self.range_errors = 0
        self.invalid_replies = 0
        self.removed_dropped = 0
        self.truncations = 0
        self.probes = 0
        self.cross_checked = 0
        self.cross_lag_waits = 0

    @classmethod
    def for_chain(
        cls, chain: str, *, confirmations: int | None = None, cross_check: "bool | LogScanner" = False, **kw,
    ) -> "LogScanner":
        """Scanner on the chain's log endpoint; ``confirmations`` defaults to ``DEFAULT_CONFIRMATIONS[chain]``.

        ``cross_check=True`` adds a second scanner on ``*_LOGS_CROSSCHECK_RPC_URL`` (else the regular
        ``*_RPC_URL``), which must be a different endpoint. A ready ``LogScanner`` may be passed instead.
        """
        url, chunk = _logs_endpoint(chain)
        if confirmations is None:
            confirmations = DEFAULT_CONFIRMATIONS[chain]
        if cross_check is True:
            curl, cchunk = _crosscheck_endpoint(chain, url)
            # same reorg margin on both sides: the secondary must not answer from a less final head
            shared = ("min_interval_s", "max_retries", "backoff_base_s", "backoff_max_s", "grow_after",
                      "probe_above", "sleep")
            cross_check = cls(web3_from_url(curl), cchunk, confirmations=confirmations,
                              **{k: v for k, v in kw.items() if k in shared})
        elif cross_check is False:
            cross_check = None
        return cls(web3_from_url(url), chunk, confirmations=confirmations, cross_check=cross_check, **kw)

    @classmethod
    def for_gate(cls, chain: str, *, confirmations: int | None = None, **kw) -> "LogScanner":
        """Scanner for gate backfills: always cross-checked on a second endpoint (``require_cross_check=True``).

        RpcNotConfigured if no distinct second endpoint is configured; the cross-check cannot be switched off.
        Use ``gate_scan`` on it to get a ``ScanResult`` whose every range is cross-checked.
        """
        if kw.pop("cross_check", True) is False:
            raise CrossCheckRequired("gate scans must be cross-checked")
        kw.pop("require_cross_check", None)
        return cls.for_chain(chain, confirmations=confirmations, cross_check=True, require_cross_check=True, **kw)

    # ------------------------------------------------------------------ transport
    def _pace(self) -> None:
        wait = self._last + self.min_interval_s - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()

    def _backoff(self, attempt: int) -> None:
        d = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
        self._sleep(d * (0.75 + 0.5 * random.random()))

    def _url(self) -> str | None:
        uri = getattr(getattr(self.w3, "provider", None), "endpoint_uri", None)
        return str(uri) if uri else None

    def _call_once(self, flt: dict) -> list:
        """One eth_getLogs; raises _RateLimited / _RangeTooLarge or re-raises a masked error."""
        self._pace()
        self.requests += 1
        try:
            return list(self.w3.eth.get_logs(flt))
        except Web3RPCError as exc:
            code, msg = _rpc_error_parts(exc)
            kind = classify_rpc_error(code, msg)
            if kind == "rate":
                raise _RateLimited() from None
            if kind == "range":
                raise _RangeTooLarge() from None
            raise mask_rpc_error(exc, self._url())
        except RpcTransportError as exc:
            msg = str(exc).lower()
            kind = classify_rpc_error(None, msg, exc.status)
            if kind == "rate":
                raise _RateLimited() from None
            if kind == "range" and (exc.status in _RANGE_CODES_HTTP or exc.status is None):
                raise _RangeTooLarge() from None  # 413 or a client-side read timeout on a heavy query
            raise

    def _request(self, flt: dict) -> list:
        """eth_getLogs with backoff for rate limits and transient errors; range errors propagate."""
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(flt)
            except _RangeTooLarge:
                raise
            except _RateLimited:
                self.rate_limited += 1
                log.debug("eth_getLogs rate limited (attempt %d)", attempt + 1)
                if attempt == self.max_retries:
                    raise RpcTransportError("eth_getLogs: still rate limited after retries", status=429) from None
            except (Web3RPCError, RpcTransportError):
                if attempt == self.max_retries:
                    raise
            self._backoff(attempt)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _validate(raw: list, frm: int, to: int, addrs: frozenset[str] | None, spec) -> tuple[list, int]:
        """(logs without removed=true entries, number dropped); _BadReply on any log outside the request."""
        out = []
        dropped = 0
        for r in raw:
            if r.get("removed"):
                dropped += 1
                continue
            bn = int(r["blockNumber"])
            if not frm <= bn <= to:
                raise _BadReply(f"log in block {bn} outside requested [{frm}, {to}]")
            if addrs is not None and str(r["address"]).lower() not in addrs:
                raise _BadReply(f"log from unexpected address {r['address']} in block {bn}")
            if spec is not None:
                topics = [_to_bytes(t) for t in r["topics"]]
                for i, allowed in enumerate(spec):
                    if allowed is None:
                        continue
                    if i >= len(topics) or topics[i] not in allowed:
                        raise _BadReply(f"log in block {bn} does not match topic filter at position {i}")
            out.append(r)
        return out, dropped

    @staticmethod
    def _looks_truncated(n: int, span: int) -> bool:
        return span > 1 and n in TRUNCATION_CAPS

    def _needs_probe(self, key: tuple, n: int, span: int) -> bool:
        if span <= 1:
            return False
        density = self._density.get(key)
        if density is None or span > self.probe_above:
            return True  # first chunk of this filter, or a wide chunk: probe whatever the density
        expected = density * span
        return expected >= 100 and n * 20 < expected

    def _fetch_checked(self, flt: dict, frm: int, to: int, addrs, spec) -> list:
        for attempt in range(self.max_retries + 1):
            raw = self._request(flt)
            try:
                checked, dropped = self._validate(raw, frm, to, addrs, spec)
            except _BadReply as exc:
                self.invalid_replies += 1
                log.warning("eth_getLogs reply rejected: %s", exc)
                if attempt == self.max_retries:
                    raise LogsReplyInvalid(f"eth_getLogs [{frm}, {to}]: {exc}") from None
                self._backoff(attempt)
                continue
            if dropped:
                self.removed_dropped += dropped
                log.warning("eth_getLogs [%d, %d]: dropped %d removed=true logs", frm, to, dropped)
            return checked
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ public
    def confirmed_head(self) -> int:
        head = int(self.w3.eth.block_number) - self.confirmations
        if self._confirmed_head_seen is None or head > self._confirmed_head_seen:
            self._confirmed_head_seen = head
        return head

    def _require_confirmed(self, to_block: int | None) -> int:
        """``to_block`` (the confirmed head for None); BlockNotConfirmed if newer than ``head - confirmations``.
        A block at or below a confirmed head already seen needs no new eth_blockNumber."""
        if to_block is not None and (not self.confirmations or (
                self._confirmed_head_seen is not None and to_block <= self._confirmed_head_seen)):
            return to_block
        head = self.confirmed_head()
        if to_block is None:
            return head
        if to_block > head:
            raise BlockNotConfirmed(f"to_block {to_block} is newer than head - {self.confirmations} = {head}")
        return to_block

    def iter_logs(
        self,
        address: str | Sequence[str] | None,
        topics: Sequence | None,
        from_block: int,
        to_block: int | None,
    ) -> Iterator[Log]:
        """Yield logs in block order for ``[from_block, to_block]`` (inclusive).

        With ``confirmations`` > 0, ``to_block`` must not exceed ``head - confirmations``
        (``BlockNotConfirmed``); ``to_block=None`` means "up to the confirmed head".
        """
        for rng in self.iter_ranges(address, topics, from_block, to_block):
            yield from rng.logs

    def scan(self, address, topics, from_block: int, to_block: int | None) -> ScanResult:
        """All of ``[from_block, to_block]`` as ``ScannedRange``s, each with its ``cross_checked`` flag."""
        ranges = tuple(self.iter_ranges(address, topics, from_block, to_block))
        end = ranges[-1].to_block if ranges else (from_block - 1 if to_block is None else to_block)
        return ScanResult(from_block, end, ranges)

    def gate_scan(self, address, topics, from_block: int, to_block: int | None) -> ScanResult:
        """``scan`` for gate data: CrossCheckRequired unless every range is cross-checked on a second endpoint."""
        if self.cross_check is None:
            raise CrossCheckRequired("gate scans must be cross-checked; "
                                     "use LogScanner.for_gate(chain)")
        res = self.scan(address, topics, from_block, to_block)
        if res.unchecked_ranges():
            raise CrossCheckRequired(f"ranges not cross-checked: {res.unchecked_ranges()}")
        return res

    def iter_ranges(
        self,
        address: str | Sequence[str] | None,
        topics: Sequence | None,
        from_block: int,
        to_block: int | None,
    ) -> Iterator[ScannedRange]:
        """Accepted chunks of ``[from_block, to_block]`` in order, as ``ScannedRange`` (rules as ``iter_logs``)."""
        if self.require_cross_check and self.cross_check is None:
            raise CrossCheckRequired("this scanner requires a cross-check endpoint")
        to_block = self._require_confirmed(to_block)
        addr = None
        addrs = None
        if address is not None:
            addr = to_checksum_address(address) if isinstance(address, str) else [to_checksum_address(a) for a in address]
            addrs = frozenset(a.lower() for a in ([addr] if isinstance(addr, str) else addr))
        spec = _norm_topic_spec(topics)
        key = (addrs, None if spec is None else tuple(spec))
        frm = from_block
        while frm <= to_block:
            to = min(frm + self.chunk - 1, to_block)
            span = to - frm + 1
            flt = {"fromBlock": frm, "toBlock": to}
            if addr is not None:
                flt["address"] = addr
            if topics is not None:
                flt["topics"] = list(topics)
            checked = False
            try:
                raw = self._fetch_checked(flt, frm, to, addrs, spec)
                if self.truncation_check:
                    self._check_truncation(key, raw, flt, frm, to, addrs, spec)
                if self.cross_check is not None:
                    self._cross_check(raw, address, topics, frm, to)
                    checked = True
            except _RangeTooLarge as exc:
                self.range_errors += 1
                self._ok_streak = 0
                if span == 1:
                    raise RuntimeError(f"eth_getLogs rejects even a single block ({frm})") from None
                self.chunk = max(1, span // 2)
                if isinstance(exc, _SilentlyTruncated):
                    self.max_chunk = min(self.max_chunk, self.chunk)
                log.debug("eth_getLogs range too large, chunk -> %d", self.chunk)
                continue
            self._on_success(key, len(raw), span)
            logs = tuple(_to_log(r) for r in sorted(raw, key=lambda x: (int(x["blockNumber"]), int(x["logIndex"]))))
            yield ScannedRange(frm, to, logs, checked)
            frm = to + 1

    def _check_truncation(self, key: tuple, raw: list, flt: dict, frm: int, to: int, addrs, spec) -> None:
        span = to - frm + 1
        if self._looks_truncated(len(raw), span):
            self.truncations += 1
            log.warning("eth_getLogs [%d, %d] returned exactly %d logs (provider cap?), splitting", frm, to, len(raw))
            raise _RangeTooLarge()
        if not self._needs_probe(key, len(raw), span):
            return
        # one probe of the upper half: any log there that the full reply lacks means it was cut
        mid = frm + span // 2
        self.probes += 1
        probe = self._fetch_checked(dict(flt, fromBlock=mid), mid, to, addrs, spec)
        have = {(int(r["blockNumber"]), int(r["logIndex"])) for r in raw}
        if any((int(r["blockNumber"]), int(r["logIndex"])) not in have for r in probe):
            self.truncations += 1
            log.warning("eth_getLogs [%d, %d] reply was truncated (%d logs), splitting", frm, to, len(raw))
            raise _SilentlyTruncated()

    def _wait_for_cross_head(self, to: int) -> None:
        """Wait until the cross-check endpoint's confirmed head reaches ``to``: a lagging second
        endpoint is waited for with exponential backoff, never compared (its missing logs would look like a
        mismatch). CrossCheckEndpointLagging after ``max_retries`` waits."""
        cc = self.cross_check
        if cc._confirmed_head_seen is not None and cc._confirmed_head_seen >= to:
            return
        theirs = None
        for attempt in range(self.max_retries + 1):
            theirs = cc.confirmed_head()
            if theirs >= to:
                return
            self.cross_lag_waits += 1
            log.info("cross-check endpoint confirmed head %d < %d (this endpoint %s), waiting",
                     theirs, to, self._confirmed_head_seen)
            if attempt < self.max_retries:
                self._backoff(attempt)
        raise CrossCheckEndpointLagging(
            f"cross-check endpoint confirmed head {theirs} still below block {to} after {self.max_retries} retries")

    def _cross_check(self, raw: list, address, topics, frm: int, to: int) -> None:
        """Compare an accepted chunk with the second endpoint once its confirmed head has reached the chunk.
        Logs missing here -> split and re-read; logs missing there -> ``LogsCrossCheckMismatch`` (no way to
        tell which side is right)."""
        self._wait_for_cross_head(to)
        other = {(lg.block_number, lg.log_index) for lg in self.cross_check.iter_logs(address, topics, frm, to)}
        have = {(int(r["blockNumber"]), int(r["logIndex"])) for r in raw}
        self.cross_checked += 1
        if have == other:
            return
        missing, extra = len(other - have), len(have - other)
        if extra == 0 and to > frm:
            self.truncations += 1
            log.warning("eth_getLogs [%d, %d]: cross-check found %d missing logs, splitting", frm, to, missing)
            raise _SilentlyTruncated()
        raise LogsCrossCheckMismatch(
            f"eth_getLogs [{frm}, {to}]: {len(have)} logs here vs {len(other)} on the cross-check endpoint "
            f"({missing} missing here, {extra} missing there)")

    def _on_success(self, key: tuple, n: int, span: int) -> None:
        d = n / span
        prev = self._density.get(key)
        self._density[key] = d if prev is None else 0.8 * prev + 0.2 * d
        self._ok_streak += 1
        if self._ok_streak >= self.grow_after and self.chunk < self.max_chunk:
            self.chunk = min(self.max_chunk, self.chunk * 2)
            self._ok_streak = 0
            log.debug("eth_getLogs chunk grows back -> %d", self.chunk)

    def get_logs(self, address, topics, from_block: int, to_block: int | None) -> list[Log]:
        return list(self.iter_logs(address, topics, from_block, to_block))
