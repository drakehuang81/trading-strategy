"""Render an IC-by-(signal x horizon) table to markdown."""
from __future__ import annotations


def render_ic_markdown(
    ic_by_signal: dict[str, dict[str, float]], *, n_tests: int
) -> str:
    horizons = list(next(iter(ic_by_signal.values())).keys())
    header = "| signal | " + " | ".join(horizons) + " |"
    sep = "|" + "---|" * (len(horizons) + 1)
    lines = [header, sep]
    for sig, ic in ic_by_signal.items():
        cells = " | ".join(f"{ic[h]:.3f}" for h in horizons)
        lines.append(f"| {sig} | {cells} |")
    lines.append("")
    lines.append(f"_tests run: {n_tests} (multiple-testing guard — see spec §7)_")
    return "\n".join(lines)
