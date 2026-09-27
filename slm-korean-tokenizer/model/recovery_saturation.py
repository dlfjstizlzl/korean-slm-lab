#!/usr/bin/env python3
"""Saturation analysis for embedding-only Stage-1 recovery.

Thresholds are fixed here, in code, before the extended run's results are read,
so the verdict cannot be tuned to the outcome:

  RELATIVE  a 1M-token interval whose improvement is < 25% of the previous
            interval's improvement
  ABSOLUTE  a 1M-token interval whose improvement is < 0.03 NLL/char

Either condition marks embedding-only recovery as approaching saturation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RELATIVE_THRESHOLD = 0.25
ABSOLUTE_THRESHOLD = 0.03
TOKENS_PER_UNIT = 1_000_000


def interval_rates(points: list[tuple[int, float]]) -> list[dict]:
    """Δ NLL/char per 1M tokens for each consecutive pair of measurements."""
    out: list[dict] = []
    for (tokens_a, nll_a), (tokens_b, nll_b) in zip(points, points[1:]):
        span = tokens_b - tokens_a
        if span <= 0:
            continue
        improvement = nll_a - nll_b
        out.append({
            "from_tokens": tokens_a, "to_tokens": tokens_b, "span_tokens": span,
            "from_nll": nll_a, "to_nll": nll_b,
            "improvement": improvement,
            "improvement_per_1m": improvement * TOKENS_PER_UNIT / span,
        })
    return out


def classify(rates: list[dict]) -> dict:
    """Apply the two fixed thresholds to the final interval."""
    if not rates:
        return {"saturating": False, "reason": "no intervals"}
    final = rates[-1]
    absolute = final["improvement_per_1m"]
    previous = rates[-2]["improvement_per_1m"] if len(rates) > 1 else None
    relative = (absolute / previous) if previous else None

    absolute_hit = absolute < ABSOLUTE_THRESHOLD
    relative_hit = relative is not None and relative < RELATIVE_THRESHOLD
    reasons = []
    if absolute_hit:
        reasons.append(f"absolute {absolute:.4f} < {ABSOLUTE_THRESHOLD}")
    if relative_hit:
        reasons.append(f"relative {relative:.3f} < {RELATIVE_THRESHOLD}")
    return {
        "thresholds": {"relative": RELATIVE_THRESHOLD, "absolute": ABSOLUTE_THRESHOLD},
        "final_interval_per_1m": absolute,
        "previous_interval_per_1m": previous,
        "relative_ratio": relative,
        "absolute_criterion_met": absolute_hit,
        "relative_criterion_met": relative_hit,
        "saturating": absolute_hit or relative_hit,
        "reason": "; ".join(reasons) if reasons else "still improving on both criteria",
    }


def snap_to_grid(points: list[tuple[int, float]], grid: list[int]) -> list[tuple[int, float]]:
    """Keep the measured point nearest each grid target.

    The saturation criteria are defined per 1M tokens, so the verdict is read off
    ~1M-token intervals (0 -> 2M -> 3M -> 4M) rather than the finer intermediate
    evaluation points, whose short spans make the per-1M rate noisy.
    """
    chosen: dict[int, float] = {}
    for target in grid:
        tokens, nll = min(points, key=lambda item: abs(item[0] - target))
        chosen[tokens] = nll
    return sorted(chosen.items())


def collect_points(runs: list[dict]) -> list[tuple[int, float]]:
    """Merge run histories into one cumulative-token curve, de-duplicated."""
    seen: dict[int, float] = {}
    for run in runs:
        for entry in run["history"]:
            seen[int(entry["tokens_seen"])] = float(entry["nll_per_character"])
    return sorted(seen.items())


def category_points(runs: list[dict]) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, dict[int, float]] = {}
    for run in runs:
        for entry in run["history"]:
            for category, value in entry.get("by_category", {}).items():
                merged.setdefault(category, {})[int(entry["tokens_seen"])] = float(value)
    return {category: sorted(values.items()) for category, values in merged.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-json", default="outputs/recovery_saturation.json")
    parser.add_argument("--original-nll", type=float, default=1.7892619577377629)
    parser.add_argument("--transplant-nll", type=float, default=2.863406813686428)
    parser.add_argument("--grid-tokens", type=int, nargs="*",
                        default=[0, 2_000_000, 3_000_000, 4_000_000],
                        help="Cumulative-token targets the saturation intervals are read at.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.runs]
    points = collect_points(runs)
    grid_points = snap_to_grid(points, args.grid_tokens) if args.grid_tokens else points
    rates = interval_rates(grid_points)
    verdict = classify(rates)

    gap = args.transplant_nll - args.original_nll
    curve = [
        {
            "tokens": tokens, "nll_per_character": nll,
            "recovery_ratio": (args.transplant_nll - nll) / gap if gap else None,
            "remaining_gap": nll - args.original_nll,
        }
        for tokens, nll in points
    ]

    report = {
        "original_qwen_nll_per_character": args.original_nll,
        "transplant_nll_per_character": args.transplant_nll,
        "recoverable_gap": gap,
        "curve": curve,
        "grid_tokens": args.grid_tokens,
        "interval_points": [{"tokens": t, "nll_per_character": v} for t, v in grid_points],
        "intervals": rates,
        "saturation": verdict,
        "by_category": {
            category: {
                "curve": [{"tokens": t, "nll_per_character": v} for t, v in values],
                "intervals": interval_rates(values),
            }
            for category, values in category_points(runs).items()
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"curve": curve, "intervals": rates, "saturation": verdict},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
