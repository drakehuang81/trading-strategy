from decision.sizing import FixedFractionalSizer, SizingPipeline, IdentityModifier


def test_fixed_fractional_honours_risk_budget():
    s = FixedFractionalSizer(fraction=0.0025)
    size = s.size(equity_usdt=10_000, entry=2000, stop_loss=1980)
    assert abs(size - 1.25) < 1e-9


def test_sizing_pipeline_with_identity_is_noop():
    p = SizingPipeline([IdentityModifier()])
    assert p.apply(5.0, consecutive_wins=10, day_pnl_r=-0.5) == 5.0
