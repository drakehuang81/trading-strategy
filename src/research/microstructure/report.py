"""Render an IC-by-(signal x horizon) table to markdown."""
from __future__ import annotations


def render_ic_markdown(
    ic_by_signal: dict[str, dict[str, float]], *, n_tests: int
) -> str:
    if not ic_by_signal:
        return "_no signals to report_"
    # union of all horizons, preserving first-seen order
    horizons: list[str] = list(
        dict.fromkeys(h for ic in ic_by_signal.values() for h in ic)
    )
    header = "| signal | " + " | ".join(horizons) + " |"
    sep = "|" + "---|" * (len(horizons) + 1)
    lines = [header, sep]
    for sig, ic in ic_by_signal.items():
        cells = " | ".join(f"{ic.get(h, float('nan')):.3f}" for h in horizons)
        lines.append(f"| {sig} | {cells} |")
    lines.append("")
    lines.append(f"_tests run: {n_tests} (multiple-testing guard — see spec §7)_")
    return "\n".join(lines)
