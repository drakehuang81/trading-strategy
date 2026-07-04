from scripts.recon.sweep_symbols import render_sweep_markdown


def test_render_sweep_markdown_flags_passes_and_insufficient():
    results = [
        {"symbol": "SOLUSDT",
         "own": {"verdict": "REAL-ALPHA candidate", "ic_test": 0.15},
         "cross": {"verdict": "FAILED — OOS", "ic_test": 0.02}},
        {"symbol": "XRPUSDT",
         "own": {"verdict": "INSUFFICIENT DATA (500h)"},
         "cross": {"verdict": "FAILED — OOS, post-cost", "ic_test": -0.01}},
    ]
    md = render_sweep_markdown(results)
    assert "SOLUSDT:own" in md                                  # flagged as pass
    assert "| SOLUSDT | REAL-ALPHA candidate | 0.150 |" in md
    assert "—" in md                                            # no ic_test cell
    assert "NONE" not in md
