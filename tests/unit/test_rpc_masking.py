"""Transport errors must never reveal the RPC URL / API key (HTTP 429 and 500 from a local fake node)."""

import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from v4quote.chain.loaders import block_timestamp, find_v4_pool_key
from v4quote.chain.multicall import Call, CallFailed, eth_call
from v4quote.chain.rpc import RpcTransportError, get_web3, web3_from_url

SECRET = "FAKEKEY0123456789abcdef"


class _Handler(BaseHTTPRequestHandler):
    status = 429

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = f'{{"error": "rate limited for {self.path}"}}'.encode()
        self.send_response(self.server.status)  # type: ignore[attr-defined]
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(params=[429, 500])
def fake_node(request):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.status = request.param  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v2/{SECRET}?apikey={SECRET}", request.param
    srv.shutdown()


def _full_text(exc: BaseException) -> str:
    return str(exc) + "".join(traceback.format_exception(exc))


def _assert_clean(exc: BaseException, url: str) -> None:
    text = _full_text(exc)
    assert SECRET not in text
    assert url not in text
    assert exc.__cause__ is None
    ctx = exc.__context__
    while ctx is not None:  # nothing in the implicit chain may carry the key either
        assert SECRET not in str(ctx)
        ctx = ctx.__context__


def test_provider_masks_url(fake_node):
    url, status = fake_node
    w3 = web3_from_url(url, timeout=5)
    with pytest.raises(RpcTransportError) as ei:
        w3.eth.block_number
    _assert_clean(ei.value, url)
    assert ei.value.status == status
    with pytest.raises(RpcTransportError) as ei:
        block_timestamp(w3, 1)
    _assert_clean(ei.value, url)
    with pytest.raises(RpcTransportError) as ei:
        find_v4_pool_key(w3, "0x" + "11" * 20, b"\x22" * 32, 1, 10)
    _assert_clean(ei.value, url)


def test_eth_call_masks_url(fake_node):
    url, _ = fake_node
    w3 = web3_from_url(url, timeout=5)
    with pytest.raises(CallFailed) as ei:
        eth_call(w3, Call("0x" + "11" * 20, "getReserves()", (), ("uint112", "uint112", "uint32")), 1, retries=1)
    assert ei.value.reverted is False
    _assert_clean(ei.value, url)


def test_get_web3_from_env_masks_url(fake_node, monkeypatch):
    url, _ = fake_node
    monkeypatch.setenv("MONAD_RPC_URL", url)
    with pytest.raises(RpcTransportError) as ei:
        get_web3("monad")  # chain id check hits the fake node
    _assert_clean(ei.value, url)


def test_only_rpc_module_builds_providers():
    """All provider access must go through v4quote.chain.rpc (MaskedHTTPProvider)."""
    from pathlib import Path

    import v4quote

    roots = [Path(v4quote.__file__).resolve().parent, Path(__file__).resolve().parents[1]]
    offenders = []
    for root in roots:
        for f in root.rglob("*.py"):
            if f.name in ("rpc.py", "test_rpc_masking.py"):
                continue
            text = f.read_text(encoding="utf-8")
            if "HTTPProvider(" in text or "Web3(" in text or "WebsocketProvider" in text:
                offenders.append(str(f))
    assert not offenders, offenders


def test_provider_str_repr_hide_url():
    url = f"https://rpc.example.invalid/v2/{SECRET}?apikey={SECRET}"
    w3 = web3_from_url(url)
    for text in (str(w3.provider), repr(w3.provider), repr(w3)):
        assert SECRET not in text and "rpc.example.invalid" not in text


def test_batch_requests_disabled():
    w3 = web3_from_url(f"https://rpc.example.invalid/v2/{SECRET}")
    with pytest.raises(NotImplementedError) as ei:
        w3.provider.make_batch_request([("eth_blockNumber", [])])
    assert SECRET not in _full_text(ei.value)
    with pytest.raises(Exception) as ei:
        with w3.batch_requests() as batch:
            batch.add(w3.eth.get_block(1))
            batch.execute()
    assert SECRET not in _full_text(ei.value)


def test_debug_logs_do_not_contain_key(fake_node, caplog):
    import logging

    url, _ = fake_node
    w3 = web3_from_url(url, timeout=5)
    caplog.set_level(logging.DEBUG)  # root at DEBUG
    with pytest.raises(RpcTransportError):
        w3.eth.block_number
    assert SECRET not in caplog.text
    # even if someone forces the web3/urllib3 loggers to DEBUG, the redaction filter still applies
    for name in ("web3.providers.HTTPProvider", "web3._utils.http_session_manager.HTTPSessionManager",
                 "urllib3.connectionpool"):
        caplog.set_level(logging.DEBUG, logger=name)
    caplog.clear()
    with pytest.raises(RpcTransportError):
        w3.eth.block_number
    assert caplog.records, "expected DEBUG records from web3/urllib3"
    assert SECRET not in caplog.text
    assert url not in caplog.text


# --- HTTP 200 with a JSON-RPC error that echoes the endpoint path / key -----------------------------

class _EchoErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        import json

        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        err = {"code": -32000, "message": f"backend failure for {self.path} (key /v2/{SECRET})",
               "data": f"upstream http://127.0.0.1{self.path}"}
        body = json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "error": err}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def echo_node():
    srv = HTTPServer(("127.0.0.1", 0), _EchoErrorHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v2/{SECRET}"
    srv.shutdown()


def _assert_rpc_error_clean(exc: BaseException, url: str) -> None:
    _assert_clean(exc, url)
    assert SECRET not in repr(exc)
    for attr in ("message", "user_message", "rpc_response"):
        assert SECRET not in repr(getattr(exc, attr, ""))


def test_jsonrpc_error_over_http_200_is_masked(echo_node, caplog):
    import logging

    from web3.exceptions import Web3RPCError

    caplog.set_level(logging.DEBUG)
    w3 = web3_from_url(echo_node, timeout=5)
    with pytest.raises(Web3RPCError) as ei:  # type preserved, not turned into RpcTransportError
        w3.eth.block_number
    assert type(ei.value) is Web3RPCError
    assert ei.value.rpc_response["error"]["code"] == -32000  # structure preserved for classification
    _assert_rpc_error_clean(ei.value, echo_node)
    assert SECRET not in caplog.text


def test_log_scanner_reraises_masked_rpc_error(echo_node):
    from web3.exceptions import Web3RPCError

    from v4quote.chain.logs import LogScanner

    sc = LogScanner(web3_from_url(echo_node, timeout=5), 100, min_interval_s=0, max_retries=1, sleep=lambda _s: None)
    with pytest.raises(Web3RPCError) as ei:
        sc.get_logs("0x" + "11" * 20, None, 1, 10)
    _assert_rpc_error_clean(ei.value, echo_node)
    assert sc.chunk == 100  # a generic -32000 is neither a range nor a rate-limit error


def test_mask_rpc_error_keeps_type_and_scrubs_attributes():
    from web3.exceptions import ContractLogicError, Web3RPCError

    from v4quote.chain.rpc import mask_rpc_error

    url = f"https://rpc.example.invalid/v2/{SECRET}"
    e = Web3RPCError(f"boom {url}", rpc_response={"error": {"code": 1, "message": f"/v2/{SECRET}"}})
    assert mask_rpc_error(e, url) is e and type(e) is Web3RPCError
    _assert_rpc_error_clean(e, url)
    c = ContractLogicError(f"execution reverted /v2/{SECRET}", data="0x08c379a0")
    mask_rpc_error(c, url)
    assert type(c) is ContractLogicError and c.data == "0x08c379a0"
    _assert_rpc_error_clean(c, url)


def test_urllib3_retry_logger_is_redacted_and_urllib3_parent_capped(caplog):
    """urllib3.util.retry logs "Incremented Retry for (url='/v2/<key>')" at DEBUG: it carries the redaction filter
    and the ``urllib3`` parent is raised to INFO."""
    import logging

    url = f"http://127.0.0.1:1/v2/{SECRET}"
    parent = logging.getLogger("urllib3")
    old_level = parent.level
    parent.setLevel(logging.NOTSET)
    try:
        web3_from_url(url, timeout=1)
        assert parent.level >= logging.INFO
        retry = logging.getLogger("urllib3.util.retry")
        caplog.set_level(logging.DEBUG, logger="urllib3.util.retry")  # even if forced to DEBUG
        caplog.clear()
        retry.debug("Incremented Retry for (url='%s'): %r", f"/v2/{SECRET}", url)
        assert caplog.records
        assert SECRET not in caplog.text and url not in caplog.text
    finally:
        parent.setLevel(old_level)
