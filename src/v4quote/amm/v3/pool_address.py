"""CREATE2 address derivation for Uniswap-v3-style pools with a parameterised deployer.

address = keccak256(0xff ++ deployer ++ keccak256(abi.encode(token0, token1, key)) ++ init_code_hash)[12:]

* Uniswap v3: deployer = factory, key = uint24 fee.
* PancakeSwap v3: deployer = separate PancakeV3PoolDeployer, key = uint24 fee.
* Ramses / Pharaoh V3: deployer = RamsesV3PoolDeployer, key = int24 tickSpacing.

Presets are verified by tests/onchain/test_address_links.py against factory.getPool.
"""

from __future__ import annotations

from dataclasses import dataclass

from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address


@dataclass(frozen=True)
class V3DeployerPreset:
    name: str
    deployer: str
    init_code_hash: str
    key_type: str  # "uint24" (fee) or "int24" (tickSpacing)


def sort_tokens(a: str, b: str) -> tuple[str, str]:
    a, b = to_checksum_address(a), to_checksum_address(b)
    return (a, b) if int(a, 16) < int(b, 16) else (b, a)


def compute_pool_address(
    deployer: str, token_a: str, token_b: str, key: int, init_code_hash: str, key_type: str = "uint24"
) -> str:
    token0, token1 = sort_tokens(token_a, token_b)
    salt = keccak(abi_encode(["address", "address", key_type], [token0, token1, key]))
    raw = keccak(
        b"\xff" + bytes.fromhex(to_checksum_address(deployer)[2:]) + salt + bytes.fromhex(init_code_hash.removeprefix("0x"))
    )
    return to_checksum_address(raw[12:])


UNISWAP_V3_INIT_CODE_HASH = "0xe34f199b19b2b4f47f68442619d555527d244f78a3297ea89325f843f87b8b54"
PANCAKE_V3_INIT_CODE_HASH = "0x6ce8eb472fa82df5469c6ab6d485f17c3ad13c8cd7af59b3d4a8026c5ce0f7e2"
# RamsesV3 periphery PoolAddress.POOL_INIT_CODE_HASH (verified source of Pharaoh QuoterV2 on Avalanche)
RAMSES_V3_INIT_CODE_HASH = "0x892f127ed4b26ca352056c8fb54585a3268f76f97fdd84d5836ef4bda8d8c685"

PRESETS: dict[tuple[str, str], V3DeployerPreset] = {
    ("monad", "uniswap_v3"): V3DeployerPreset(
        "uniswap_v3", "0x204faca1764b154221e35c0d20abb3c525710498", UNISWAP_V3_INIT_CODE_HASH, "uint24"
    ),
    ("avalanche", "uniswap_v3"): V3DeployerPreset(
        "uniswap_v3", "0x740b1c1de25031C31FF4fC9A62f554A55cdC1baD", UNISWAP_V3_INIT_CODE_HASH, "uint24"
    ),
    ("monad", "pancakeswap_v3"): V3DeployerPreset(
        "pancakeswap_v3", "0x41ff9AA7e16B8B1a8a8dc4f0eFacd93D02d071c9", PANCAKE_V3_INIT_CODE_HASH, "uint24"
    ),
    ("avalanche", "pharaoh_v3"): V3DeployerPreset(
        "pharaoh_v3", "0x6a4113ed0915bCf5E48e758e8f4cEBFFC07C66f9", RAMSES_V3_INIT_CODE_HASH, "int24"
    ),
}
