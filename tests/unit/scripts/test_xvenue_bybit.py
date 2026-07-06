"""Tests for the Bybit downloader's pure parts."""
from scripts.xvenue.bybit import canonical_key, parse_funding_rows, unique_key_map


def test_canonical_key_multiplier_agnostic():
    assert canonical_key("BTCUSDT") == "BTC"
    assert canonical_key("1000SHIBUSDT") == "SHIB"      # Binance style
    assert canonical_key("SHIB1000USDT") == "SHIB"      # Bybit style
    assert canonical_key("1000000MOGUSDT") == "MOG"
    assert canonical_key("10000LADYSUSDT") == "LADYS"
    assert canonical_key("1INCHUSDT") == "1INCH"        # not a multiplier
    assert canonical_key("PORT3USDT") == "PORT3"        # trailing digit kept


def test_unique_key_map_drops_collisions():
    m = unique_key_map(["BTCUSDT", "1000PEPEUSDT", "PEPEUSDT"])  # PEPE collides
    assert m == {"BTC": "BTCUSDT"}


def test_parse_funding_rows_types():
    df = parse_funding_rows(
        [{"fundingRateTimestamp": "1700000000000", "fundingRate": "-0.0001"},
         {"fundingRateTimestamp": "1700028800000", "fundingRate": "0.0002"}]
    )
    assert df["ts_ms"].to_list() == [1700000000000, 1700028800000]
    assert df["rate"].to_list() == [-0.0001, 0.0002]
