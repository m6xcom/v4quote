//! ``v4quote_native``: the hot exact-integer AMM math of the Python package ``v4quote`` (src/v4quote/amm),
//! ported to Rust for fast repeated quoting (amount searches, route search). Bit-exact with Python by construction (see ``num``): a call
//! either returns Python's value / Python's revert, or asks the caller to run the Python code (``Fallback``).
//!
//! Kinds: Uniswap v2 ``get_amount_out``, Uniswap v3 exact-input swap over a tick-bitmap snapshot (also used for v4
//! pools), OZ v5 ERC-4626 vault
//! ``deposit`` / ``redeem`` quotes.

#![forbid(unsafe_code)]

pub mod num;
pub mod pool;
pub mod v2;
pub mod v3;
pub mod vault;

/// SHA-256 over the crate's sources (``Cargo.toml``, ``Cargo.lock``, ``build.rs``, ``src/**/*.rs``), computed by
/// ``build.rs`` at build time; ``v4quote.amm.native.source_hash`` computes the same over the repository.
pub const SOURCE_HASH: &str = env!("V4QUOTE_NATIVE_SOURCE_HASH");
/// ``<crate version>+<SOURCE_HASH>``: ``v4quote_native.__build__``.
pub const BUILD: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("V4QUOTE_NATIVE_SOURCE_HASH"));

#[cfg(feature = "python")]
mod py;

/// build.rs compiled into the tests: its SHA-256 against known vectors and its hash of this tree == ``SOURCE_HASH``.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

#[cfg(test)]
mod tests {
    #[test]
    fn build_script_sha256_vectors_and_tree_hash() {
        use super::build_script::{hex, sha256, source_hash};
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let m = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; // 56 bytes: two-block padding
        assert_eq!(hex(&sha256(m)), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        let big = vec![b'a'; 1_000_000];
        assert_eq!(hex(&sha256(&big)), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        let (h, files) = source_hash(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        assert_eq!(h, super::SOURCE_HASH);
        assert!(files.iter().any(|f| f == "src/lib.rs") && files.iter().any(|f| f == "build.rs"));
    }

    #[test]
    fn build_fingerprint_is_a_sha256_hex() {
        assert_eq!(super::SOURCE_HASH.len(), 64);
        assert!(super::SOURCE_HASH.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_eq!(super::BUILD, format!("{}+{}", env!("CARGO_PKG_VERSION"), super::SOURCE_HASH));
    }
}
