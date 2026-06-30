from research.microstructure.report import render_ic_markdown


def test_render_ic_markdown_table():
    ic_by_signal = {"qi": {"fwd_1s": 0.031, "fwd_60s": -0.004}}
    md = render_ic_markdown(ic_by_signal, n_tests=2)
    assert "| signal | fwd_1s | fwd_60s |" in md
    assert "0.031" in md
    assert "tests run: 2" in md


def test_render_ic_markdown_heterogeneous_horizons():
    from research.microstructure.report import render_ic_markdown
    ic = {"qi": {"fwd_1s": 0.03, "fwd_60s": -0.01}, "ofi": {"fwd_1s": 0.05}}
    md = render_ic_markdown(ic, n_tests=3)
    assert "fwd_1s" in md and "fwd_60s" in md
    assert "nan" in md.lower()  # ofi missing fwd_60s -> nan cell


def test_render_ic_markdown_empty_dict():
    from research.microstructure.report import render_ic_markdown
    md = render_ic_markdown({}, n_tests=0)
    assert "no signals" in md.lower()
