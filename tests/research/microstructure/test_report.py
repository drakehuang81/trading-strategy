from research.microstructure.report import render_ic_markdown


def test_render_ic_markdown_table():
    ic_by_signal = {"qi": {"fwd_1s": 0.031, "fwd_60s": -0.004}}
    md = render_ic_markdown(ic_by_signal, n_tests=2)
    assert "| signal | fwd_1s | fwd_60s |" in md
    assert "0.031" in md
    assert "tests run: 2" in md
