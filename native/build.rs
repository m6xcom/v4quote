//! Build fingerprint: ``SOURCE_HASH`` = SHA-256 over the crate's sources, exposed by the Python
//! module as ``v4quote_native.__source_hash__`` / ``__build__``. ``v4quote.amm.native`` computes the same hash over the
//! repository's ``native/`` at import and refuses a build made from other sources (``stale_build``: Python math).
//!
//! Hashed files (keep in sync with ``v4quote.amm.native.source_hash``): ``Cargo.toml``, ``Cargo.lock``, ``build.rs`` and
//! every ``*.rs`` under ``src/`` (recursively), in byte order of their ``/``-separated relative paths. Per file:
//! ``path`` ``\n`` ``len`` ``\n`` ``content`` ``\n`` with CRLF normalised to LF (a Windows checkout hashes like Linux).
//! Std only (no build dependencies): a plain SHA-256.

use std::fs;
use std::path::Path;

fn collect(dir: &Path, rel: &str, out: &mut Vec<String>) {
    let mut entries: Vec<_> = fs::read_dir(dir).expect("read src/").map(|e| e.expect("dir entry")).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().into_string().expect("utf-8 file name");
        let path = format!("{rel}/{name}");
        let ft = e.file_type().expect("file type");
        if ft.is_dir() {
            collect(&e.path(), &path, out);
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// ``(hex SHA-256, hashed relative paths)`` of the crate at ``root``.
pub fn source_hash(root: &Path) -> (String, Vec<String>) {
    let mut files = vec!["Cargo.toml".to_string(), "Cargo.lock".to_string(), "build.rs".to_string()];
    collect(&root.join("src"), "src", &mut files);
    files.sort();
    let mut buf = Vec::new();
    for f in &files {
        let raw = fs::read(root.join(f)).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let mut data = Vec::with_capacity(raw.len());
        for (i, &c) in raw.iter().enumerate() {
            if !(c == b'\r' && raw.get(i + 1) == Some(&b'\n')) {
                data.push(c);
            }
        }
        buf.extend_from_slice(f.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(data.len().to_string().as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(&data);
        buf.push(b'\n');
    }
    (hex(&sha256(&buf)), files)
}

fn main() {
    let (h, files) = source_hash(Path::new(env!("CARGO_MANIFEST_DIR")));
    for f in &files {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-changed=src"); // a new / removed file
    println!("cargo:rustc-env=V4QUOTE_NATIVE_SOURCE_HASH={h}");
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// FIPS 180-4 SHA-256 (the result is compared with Python's hashlib by the core tests).
pub fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = msg.to_vec();
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in m.chunks(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([chunk[4 * t], chunk[4 * t + 1], chunk[4 * t + 2], chunk[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (k, v) in h.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}
