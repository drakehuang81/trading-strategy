# Phase 2c STATUS — Pre-Registered Symbol Sweep: ALL FAILED, CLOSURE FIRED (2026-06-29)

**Tests**: 38 passed · **Sweep**: 12 symbols × 2 hypotheses, discovery window 2023-05-16 → 2024-03-30, 7 skipped days total, zero INSUFFICIENT DATA.

## Result

**PASSES: NONE — all 24 (symbol × hypothesis) tests failed the pre-committed gates.** Replication stage never triggered (conditional on discovery passes).

| symbol | own verdict | own ic_test | cross verdict | cross ic_test |
|---|---|---|---|---|
| SOLUSDT | FAILED — OOS, vs-momentum, monotone | 0.061 | FAILED — OOS, vs-controls, post-cost, monotone | 0.036 |
| XRPUSDT | FAILED (all four) | -0.013 | FAILED (all four) | 0.027 |
| DOGEUSDT | FAILED (all four) | -0.003 | FAILED (all four) | 0.022 |
| BNBUSDT | FAILED (all four) | 0.020 | FAILED (all four) | 0.041 |
| ADAUSDT | FAILED (all four) | -0.013 | FAILED (all four) | 0.040 |
| AVAXUSDT | FAILED (all four) | 0.017 | FAILED — OOS, vs-controls, monotone | 0.030 |
| LINKUSDT | FAILED (all four) | -0.005 | FAILED (all four) | 0.040 |
| LTCUSDT | FAILED (all four) | 0.011 | FAILED (all four) | 0.058 |
| DOTUSDT | FAILED (all four) | 0.011 | FAILED (all four) | 0.041 |
| MATICUSDT | FAILED (all four) | 0.012 | FAILED (all four) | 0.062 |
| ATOMUSDT | FAILED (all four) | 0.025 | FAILED (all four) | 0.060 |
| NEARUSDT | FAILED — OOS, vs-momentum, monotone | 0.028 | FAILED — OOS, vs-controls, monotone | 0.059 |

Pattern check: cross ICs cluster at +0.02~0.06 across every alt — the same near-zero family as BTC→ETH (0.042). The "thinner books carry more signal" prior did not materialize at 1h; own-book ICs on alts are indistinguishable from ETH's 0.046.

## Closure (pre-committed in the plan, now in force)

> **The "directional edge from free Binance data" question is CLOSED PERMANENTLY.** No new symbol lists, no new windows, no revisiting. Total evidence: 6 hypothesis families + a 24-test pre-registered sweep, every one refuted under four pre-committed gates.

## What remains live (handoff §7.2 / §7.3)

1. **Paper assistant ops** — run the 1h assistant in paper mode to accumulate Pre-Live-Gate operational history (needs full 3.11 venv + Ollama).
2. **qi maker option-preservation** — extend TickRecorder to record book streams and start recording; the only path that ever makes the maker/HF question testable.

## Commits (2c)

`64a3a57` sweep runner · this STATUS. Sweep caches: `data/orderbook/_sweep/<symbol>/` + shared `_cross/` (gitignored, in worktree recon-phase2b1).
