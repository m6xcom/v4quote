"""v4quote.chain.bloom against a real Avalanche header: block 87 150 040 (recorded 2026-09-27), 211 logs; the bloom built
from the logs must equal the header's logsBloom bit for bit, every address / topic is a member, and a few addresses
that emitted nothing in that block are proven absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v4quote.chain import bloom

FIX = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "avalanche_87150040_logs_bloom.json")
                 .read_text(encoding="utf-8"))


def _logs():
    return [{"address": a, "topics": t} for a, t in FIX["logs"]]


def test_bloom_of_the_logs_equals_the_real_header_field():
    got = bloom.logs_bloom(_logs())
    assert len(FIX["logs"]) == 211
    assert bloom.to_hex(got) == FIX["logsBloom"].lower()
    assert bloom.parse_bloom(FIX["logsBloom"]) == got


def test_members_and_proven_absence():
    b = bloom.parse_bloom(FIX["logsBloom"])
    for a, topics in FIX["logs"]:
        assert bloom.contains(b, a)
        assert all(bloom.contains(b, t) for t in topics)
    emitters = {a.lower() for a, _ in FIX["logs"]}
    absent = [a for a in ("0x" + f"{k:040x}" for k in range(1, 400)) if a not in emitters]
    proven = [a for a in absent if not bloom.contains(b, a)]
    assert len(proven) > 0.5 * len(absent)  # a real block: most unrelated addresses are excluded by the bloom


def test_empty_bloom_and_parsing():
    assert bloom.logs_bloom([]) == 0
    assert bloom.parse_bloom("0x" + "00" * 256) == 0
    assert not bloom.contains(0, "0x" + "11" * 20)
    assert bloom.parse_bloom(None) is None and bloom.parse_bloom("0x1234") is None and bloom.parse_bloom("0xzz") is None
    one = bloom.add(0, "0x" + "11" * 20)
    assert bin(one).count("1") in (1, 2, 3) and bloom.contains(one, "0x" + "11" * 20)
    assert bloom.parse_bloom(bloom.to_hex(one)) == one


@pytest.mark.parametrize("case", ["0X" + "00" * 256, bytes(256)])
def test_parse_variants(case):
    assert bloom.parse_bloom(case) == 0


def test_masks_agree_with_contains():
    b = bloom.parse_bloom(FIX["logsBloom"])
    items = [a for a, _ in FIX["logs"][:20]] + ["0x" + f"{k:040x}" for k in range(1, 200)]
    for x in items:
        assert bloom.contains_mask(b, bloom.item_mask(x)) == bloom.contains(b, x)
    assert bin(bloom.item_mask(items[0])).count("1") in (1, 2, 3)


def test_address_masks_cover_emitter_and_topic_form():
    a = "0x" + "ab" * 20
    as_emitter, as_topic = bloom.address_masks(a)
    assert as_emitter == bloom.item_mask(a)
    assert as_topic == bloom.item_mask("0x" + "00" * 12 + "ab" * 20)
    transfer_to_a = [{"address": "0x" + "cd" * 20,
                      "topics": ["0x" + "11" * 32, "0x" + "00" * 32, "0x" + "00" * 12 + "ab" * 20]}]
    assert bloom.contains_mask(bloom.logs_bloom(transfer_to_a), as_topic)  # a Transfer to it shows via the topic form
    assert bloom.contains_mask(bloom.logs_bloom([{"address": a, "topics": []}]), as_emitter)
    assert not bloom.contains_mask(0, as_emitter) and not bloom.contains_mask(0, as_topic)
    with pytest.raises(ValueError):
        bloom.address_masks("0x1234")
