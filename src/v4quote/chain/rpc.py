"""RPC connection helpers. Endpoints come ONLY from environment variables.

Every request to the provider goes through ``MaskedHTTPProvider``: any
exception raised while talking to the node is replaced by ``RpcTransportError``
whose message has the endpoint URL (and its path/query, where API keys usually
live) removed. The original exception is not chained, so the URL cannot leak
through tracebacks either. Build Web3 instances only via ``get_web3``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.middleware.base import Web3Middleware
from web3.providers import HTTPProvider

# JSON-RPC level errors (revert data, invalid params) must keep their type so that eth_call can
# recognise contract reverts, but their message can echo the URL / key (some gateways do), so they are
# masked in place: the JSON-RPC response is scrubbed before web3 builds the exception from it.
from web3.exceptions import ContractLogicError, Web3RPCError


@dataclass(frozen=True)
class ChainConfig:
    name: str
    chain_id: int
    rpc_env: str


CHAINS: dict[str, ChainConfig] = {
    "monad": ChainConfig("monad", 143, "MONAD_RPC_URL"),
    "avalanche": ChainConfig("avalanche", 43114, "AVALANCHE_RPC_URL"),
    "gnosis": ChainConfig("gnosis", 100, "GNOSIS_RPC_URL"),  # on-chain tests of the Gnosis pools (archive RPC)
    # on-chain tests of the PulseChain pools (a PUBLIC endpoint, e.g. https://pulsechain-rpc.publicnode.com)
    "pulsechain": ChainConfig("pulsechain", 369, "PULSECHAIN_RPC_URL"),
}


class RpcNotConfigured(RuntimeError):
    pass


class RpcTransportError(RuntimeError):
    """Transport-level failure (HTTP status, timeout, connection). Message never contains the endpoint URL."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# Query values shorter than this are not key-like (``?v=1``); masking them would rewrite ordinary error
# text ("10000 results" -> "<rpc>0000 results") and break error classification.
_MIN_QUERY_SECRET_LEN = 8


def _secret_fragments(url: str) -> list[str]:
    parts = urlsplit(url)
    frags = [url, url.rstrip("/")]
    if parts.path and parts.path != "/":
        frags.append(parts.path)
        frags += [seg for seg in parts.path.split("/") if len(seg) >= 6]
    if parts.query:
        if len(parts.query) >= _MIN_QUERY_SECRET_LEN:
            frags.append(parts.query)
        frags += [v for v in (kv.split("=", 1)[-1] for kv in parts.query.split("&") if kv)
                  if len(v) >= _MIN_QUERY_SECRET_LEN]
    if parts.username or parts.password:
        frags += [p for p in (parts.username, parts.password) if p]
    if parts.netloc:
        frags.append(parts.netloc)
    return sorted({f for f in frags if f}, key=len, reverse=True)


def mask_secret(text: str, url: str) -> str:
    """Remove the endpoint URL and any key-like parts of it from ``text``."""
    for frag in _secret_fragments(url):
        text = text.replace(frag, "<rpc>")
    return text


def _mask_obj(obj: Any, url: str) -> Any:
    if isinstance(obj, str):
        return mask_secret(obj, url)
    if isinstance(obj, dict):
        return {k: _mask_obj(v, url) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_mask_obj(v, url) for v in obj)
    return obj


def mask_rpc_response(response: Any, url: str) -> Any:
    """Scrub the ``error`` member of a JSON-RPC response (results are left untouched)."""
    if isinstance(response, dict) and response.get("error") is not None:
        response = dict(response)
        response["error"] = _mask_obj(response["error"], url)
    return response


def mask_rpc_error(exc: BaseException, url: str | None) -> BaseException:
    """Scrub ``url`` from a Web3RPCError-like exception in place (type and attributes are kept)."""
    if not url:
        return exc
    exc.args = tuple(_mask_obj(a, url) for a in exc.args)
    for attr in ("message", "user_message", "data", "rpc_response"):
        v = getattr(exc, attr, None)
        if v is not None:
            try:
                setattr(exc, attr, _mask_obj(v, url))
            except AttributeError:
                pass
    exc.__cause__ = None
    exc.__context__ = None
    exc.__suppress_context__ = True
    return exc


# Loggers that write the endpoint URL (or its path, where keys live) at DEBUG level.
_URL_LOGGERS = (
    "web3.providers.HTTPProvider",
    "web3.providers.base.BaseProvider",
    "web3.providers.base.JSONBaseProvider",
    "web3._utils.http_session_manager.HTTPSessionManager",
    "web3.manager.RequestManager",
    "urllib3.connectionpool",
    "urllib3.util.retry",  # "Incremented Retry for (url=...)" carries the request path
)
_PARENT_LOGGERS = ("web3.providers", "web3._utils.http_session_manager", "urllib3.connectionpool", "urllib3")
_KNOWN_URLS: set[str] = set()


class _RedactRpcUrlFilter(logging.Filter):
    """Rewrites log records so that no registered RPC URL (or key-like fragment of it) survives."""

    _fmt = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _KNOWN_URLS:
            return True
        try:
            text = record.getMessage()
        except Exception:  # malformed record: drop its args rather than risk leaking them
            text = str(record.msg)
        # exc_info / stack_info are formatted by the handler AFTER filters run, from the live exception, so
        # they are rendered here, masked with the message, and removed from the record.
        extra = []
        if record.exc_info:
            try:
                extra.append(self._fmt.formatException(record.exc_info))
            except Exception:
                extra.append("<exception could not be formatted>")
        elif record.exc_text:
            extra.append(record.exc_text)
        if record.stack_info:
            extra.append(str(record.stack_info))
        if extra:
            text = "\n".join([text, *extra])
        for url in list(_KNOWN_URLS):
            text = mask_secret(text, url)
        record.msg, record.args = text, ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


_FILTER = _RedactRpcUrlFilter()


def _harden_logging() -> None:
    """Redaction filter on every URL-writing logger and level >= INFO on their parents (idempotent)."""
    for name in _URL_LOGGERS:
        lg = logging.getLogger(name)
        if _FILTER not in lg.filters:
            lg.addFilter(_FILTER)
    for name in _PARENT_LOGGERS:
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level < logging.INFO:
            lg.setLevel(logging.INFO)


class MaskedHTTPProvider(HTTPProvider):
    """HTTPProvider whose transport errors, str/repr and log records never reveal the endpoint URL.

    JSON-RPC batch requests are disabled (``make_batch_request`` raises): all reads go through
    Multicall3 instead, and one code path is easier to keep leak-free.
    """

    def __init__(self, endpoint_uri: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(endpoint_uri, *args, **kwargs)
        _KNOWN_URLS.add(str(endpoint_uri))
        _harden_logging()

    def __str__(self) -> str:
        return "MaskedHTTPProvider(<rpc>)"

    def __repr__(self) -> str:
        return "MaskedHTTPProvider(<rpc>)"

    def make_batch_request(self, batch_requests: Any) -> Any:  # noqa: ARG002
        raise NotImplementedError("JSON-RPC batch requests are disabled for MaskedHTTPProvider; use Multicall3")

    def make_request(self, method: Any, params: Any) -> Any:
        url = str(self.endpoint_uri)
        failure: tuple[str, int | None] | None = None
        masked: BaseException | None = None
        try:
            # HTTP 200 + {"error": ...}: web3 turns this into Web3RPCError later (RequestManager), from
            # the response we return, so the error text is scrubbed here.
            return mask_rpc_response(super().make_request(method, params), url)
        except (ContractLogicError, Web3RPCError) as exc:
            masked = mask_rpc_error(exc, url)  # keep the type (eth_call needs revert detection)
        except Exception as exc:  # requests.HTTPError, ConnectionError, Timeout, ...
            status = getattr(getattr(exc, "response", None), "status_code", None)
            failure = (mask_secret(f"{type(exc).__name__}: {exc}", url), status)
        # raised outside the except block so the original exception is neither chained nor in __context__
        if masked is not None:
            raise masked
        raise RpcTransportError(f"RPC {method} failed: {failure[0]}", status=failure[1])


def rpc_url(chain: str) -> str | None:
    cfg = CHAINS[chain]
    url = os.environ.get(cfg.rpc_env, "").strip()
    return url or None


def web3_from_url(url: str, *, timeout: float = 30.0) -> Web3:
    """Web3 over MaskedHTTPProvider with web3's own HTTP retry disabled (callers retry).

    Side effect (deliberate, so that a key in the URL never reaches a log): the first provider built raises the level
    of the ``web3.providers``, ``web3._utils.http_session_manager``, ``urllib3.connectionpool`` and ``urllib3``
    loggers to INFO when they are unset or lower (their DEBUG records carry the endpoint URL) and adds a redaction
    filter to the URL-writing loggers. A level you set to INFO or higher is kept."""
    provider = MaskedHTTPProvider(url, request_kwargs={"timeout": timeout}, exception_retry_configuration=None)
    w3 = Web3(provider)
    # Avalanche C-Chain blocks can carry extraData longer than 32 bytes (e.g. older blocks); without this
    # middleware eth_getBlock fails validation. Harmless for other chains.
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


PINNED_CHAIN_ID_MIDDLEWARE = "pinned_chain_id"


def pin_chain_id(w3: Web3, expected: int | None = None) -> int:
    """Read ``eth_chainId`` ONCE and answer every later ``eth_chainId`` of this Web3 from that value.

    web3.py's ValidationMiddleware asks the node for the chain id on every ``eth_call`` / ``eth_estimateGas`` (twice:
    once for the request and once for the response formatters), so a Multicall3 read costs three round trips instead
    of one; a long-lived reader (e.g. a pool book refreshed every head) pays it on every refresh. The
    chain id of an endpoint does not change while the process runs; a node switched to another chain under a running
    process is caught by the callers' own checks (block hashes, start-up checks).

    ``expected``: raise RuntimeError when the endpoint is another chain (fail closed, before anything is cached).
    Idempotent: a second call keeps the first pinned value and makes no request."""
    if PINNED_CHAIN_ID_MIDDLEWARE in w3.middleware_onion:
        cid = int(w3.middleware_onion[PINNED_CHAIN_ID_MIDDLEWARE].chain_id)
    else:
        cid = int(w3.eth.chain_id)
        if expected is not None and cid != expected:
            raise RuntimeError(f"the RPC endpoint is chainId {cid}, expected {expected}")
        answer = hex(cid)

        class _PinnedChainId(Web3Middleware):
            chain_id = cid

            def wrap_make_request(self, make_request: Any) -> Any:
                def middleware(method: Any, params: Any) -> Any:
                    if method == "eth_chainId":
                        return {"jsonrpc": "2.0", "id": 0, "result": answer}
                    return make_request(method, params)

                return middleware

        w3.middleware_onion.inject(_PinnedChainId, name=PINNED_CHAIN_ID_MIDDLEWARE, layer=0)
    if expected is not None and cid != expected:
        raise RuntimeError(f"the RPC endpoint is chainId {cid}, expected {expected}")
    return cid


def get_web3(chain: str, *, timeout: float = 30.0, verify_chain_id: bool = True) -> Web3:
    """Web3 instance for ``chain`` using the URL from its env var (``CHAINS[chain].rpc_env``, e.g. AVALANCHE_RPC_URL).

    Built with ``web3_from_url``: it sets the web3 / urllib3 transport loggers to INFO (see there) so that DEBUG
    records with the endpoint URL are never emitted."""
    cfg = CHAINS[chain]
    url = rpc_url(chain)
    if url is None:
        raise RpcNotConfigured(f"environment variable {cfg.rpc_env} is not set")
    w3 = web3_from_url(url, timeout=timeout)
    if verify_chain_id:
        got = w3.eth.chain_id
        if got != cfg.chain_id:
            raise RuntimeError(f"{cfg.rpc_env} points to chainId {got}, expected {cfg.chain_id}")
    return w3
