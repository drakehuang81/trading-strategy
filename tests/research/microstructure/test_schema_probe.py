import polars as pl

from research.microstructure.schema_probe import summarize_schema


def test_summarize_schema_reports_columns_and_cadence():
    df = pl.DataFrame({
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "percentage": [1, 1, 1],
        "depth": [10.0, 11.0, 12.0],
        "notional": [1000.0, 1100.0, 1200.0],
    })
    summary = summarize_schema(df, ts_col="timestamp")
    assert summary["n_rows"] == 3
    assert "percentage" in summary["columns"]
    assert summary["median_gap_ms"] == 1000
    assert summary["distinct_percentage"] == [1]
