"""Tests for the carry study's universe/download pure parts."""
from scripts.carry.universe import (
    FUNDING_PREFIX,
    next_marker,
    read_funding_csv,
    usdt_symbols_from_prefixes,
)


def test_usdt_filter_drops_other_quotes_and_root():
    prefixes = [
        FUNDING_PREFIX,                          # root echo — dropped
        f"{FUNDING_PREFIX}BTCUSDT/",
        f"{FUNDING_PREFIX}1000BONKUSDC/",        # USDC leg — dropped
        f"{FUNDING_PREFIX}ETHBUSD/",             # BUSD leg — dropped
        f"{FUNDING_PREFIX}1000PEPEUSDT/",
    ]
    assert usdt_symbols_from_prefixes(prefixes) == ["BTCUSDT", "1000PEPEUSDT"]


def test_next_marker_only_when_truncated():
    truncated = (
        "<x><IsTruncated>true</IsTruncated>"
        "<NextMarker>data/futures/um/monthly/fundingRate/M/</NextMarker></x>"
    )
    assert next_marker(truncated) == "data/futures/um/monthly/fundingRate/M/"
    assert next_marker("<x><IsTruncated>false</IsTruncated></x>") is None


def test_read_funding_csv_with_header():
    raw = (
        b"calc_time,funding_interval_hours,last_funding_rate\n"
        b"1780272000001,8,0.00005703\n"
        b"1780300800001,8,-0.00004438\n"
    )
    df = read_funding_csv(raw)
    assert df.columns == ["ts_ms", "interval_h", "rate"]
    assert df["ts_ms"].to_list() == [1780272000001, 1780300800001]
    assert df["rate"][1] == -0.00004438


def test_read_funding_csv_headerless_and_two_column_variants():
    headerless = b"1780272000001,8,0.0001\n"
    df = read_funding_csv(headerless)
    assert df["rate"].to_list() == [0.0001]

    two_col = b"1780272000001,0.0002\n"
    df2 = read_funding_csv(two_col)
    assert df2["rate"].to_list() == [0.0002]
    assert df2["interval_h"].to_list() == [None]


def test_spot_audit_hedgeable_mapping():
    from scripts.carry.spot_audit import hedgeable, spot_bases

    info = {"symbols": [
        {"baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
        {"baseAsset": "PEPE", "quoteAsset": "USDT", "status": "TRADING"},
        {"baseAsset": "MOG", "quoteAsset": "USDT", "status": "TRADING"},
        {"baseAsset": "DEAD", "quoteAsset": "USDT", "status": "BREAK"},
        {"baseAsset": "EUR", "quoteAsset": "BTC", "status": "TRADING"},
    ]}
    bases = spot_bases(info)
    assert hedgeable("BTCUSDT", bases)
    assert hedgeable("1000PEPEUSDT", bases)        # unit-converted hedge
    assert hedgeable("1000000MOGUSDT", bases)
    assert not hedgeable("DEADUSDT", bases)        # spot not TRADING
    assert not hedgeable("NOSPOTUSDT", bases)
    assert not hedgeable("1000USDT", bases)        # prefix-only stays unhedged
