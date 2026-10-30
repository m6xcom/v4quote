"""v4quote.chain.rpc.pin_chain_id: one eth_chainId per process instead of two per eth_call (web3's ValidationMiddleware).

A local fake JSON-RPC node counts the methods it receives; the Web3 is the real one of ``web3_from_url``
(MaskedHTTPProvider + the default middleware), so the count is what a long-lived reader sends to its node.
"""

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from v4quote.chain.multicall import Call, eth_call, multicall
from v4quote.chain.rpc import PINNED_CHAIN_ID_MIDDLEWARE, pin_chain_id, web3_from_url

WORD = "0x" + "00" * 31 + "07"


class _Node(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        m = req["method"]
        self.server.seen[m] += 1  # type: ignore[attr-defined]
        if m == "eth_chainId":
            res = hex(self.server.chain_id)  # type: ignore[attr-defined]
        elif m == "eth_blockNumber":
            res = "0x64"
        elif m == "eth_call":
            data = req["params"][0]["data"]
            if data.startswith("0x82ad56cb"):  # Multicall3.aggregate3: answer (true, WORD) for every call
                from eth_abi import decode, encode

                (calls,) = decode(["(address,bool,bytes)[]"], bytes.fromhex(data[10:]))
                res = "0x" + encode(["(bool,bytes)[]"], [[(True, bytes.fromhex(WORD[2:]))] * len(calls)]).hex()
            else:
                res = WORD
        else:
            res = None
        body = json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": res}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def node():
    srv = HTTPServer(("127.0.0.1", 0), _Node)
    srv.seen = Counter()  # type: ignore[attr-defined]
    srv.chain_id = 100  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


POOL = "0x" + "11" * 20


def _reads(w3) -> None:
    for _ in range(3):
        assert eth_call(w3, Call(POOL, "fee()", (), ("uint24",)), 100) == 7
    assert multicall(w3, [Call(POOL, "liquidity()", out=("uint128",))] * 4, 100) == [7] * 4


def test_unpinned_web3_asks_the_chain_id_on_every_eth_call(node):
    srv, url = node
    _reads(web3_from_url(url, timeout=5))
    assert srv.seen["eth_call"] == 4
    assert srv.seen["eth_chainId"] >= 4  # web3 7.x: two per eth_call (request and response formatters)


def test_pinned_web3_asks_once_and_answers_the_rest_itself(node):
    srv, url = node
    w3 = web3_from_url(url, timeout=5)
    assert pin_chain_id(w3, expected=100) == 100
    assert srv.seen["eth_chainId"] == 1
    _reads(w3)
    assert w3.eth.chain_id == 100
    assert srv.seen["eth_call"] == 4 and srv.seen["eth_chainId"] == 1  # nothing more after the start-up read
    assert pin_chain_id(w3, expected=100) == 100 and srv.seen["eth_chainId"] == 1  # idempotent, no request
    with pytest.raises(RuntimeError, match="expected 10"):
        pin_chain_id(w3, expected=10)


def test_pin_refuses_another_chain_before_caching_anything(node):
    srv, url = node
    srv.chain_id = 43114
    w3 = web3_from_url(url, timeout=5)
    with pytest.raises(RuntimeError, match="chainId 43114, expected 100"):
        pin_chain_id(w3, expected=100)
    assert PINNED_CHAIN_ID_MIDDLEWARE not in w3.middleware_onion
    assert pin_chain_id(w3) == 43114 and PINNED_CHAIN_ID_MIDDLEWARE in w3.middleware_onion
