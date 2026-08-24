"""Data-Efficiency curve from committed eval reports — no plotting deps required.

Reads results/reports/*.json (the JSON `eval.py` writes) and renders, per behavior, Spec-adherence
and Robustness vs training-set size N. Emits a self-contained SVG + HTML (always) and a Markdown table
to stdout; also writes a PNG if matplotlib happens to be installed.

    python plot_curve.py                       # scans results/reports
    python plot_curve.py --reports DIR --out results

Point size N is parsed from each report's tag/model_id (e.g. a run tagged 'N500' -> 500). A report
whose label has no size but reads 'base' is drawn as a dashed baseline; 'tuned'/'full' with no N is
treated as the full-size point if --full-n is given.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

SERIES = [("spec_adherence", "Spec-adherence", "#2f81f7"),
          ("robustness", "Robustness", "#3fb950")]


def _size_from(label: str, full_n: int | None):
    if not label:
        return None
    m = re.search(r"[nN]\s*[_-]?\s*(\d{2,6})", label) or re.search(r"(\d{3,6})", label)
    if m:
        return int(m.group(1))
    low = label.lower()
    if full_n and any(w in low for w in ("tuned", "full", "sft")):
        return full_n
    return None


def load(reports_dir: str, full_n: int | None):
    """-> {behavior: {"points": {N: {metric: val}}, "base": {metric: val} or None}}"""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        try:
            r = json.loads(open(path).read())
        except Exception:
            continue
        beh = r.get("behavior", "?")
        m = r.get("metrics", {})
        if not m:
            continue
        label = str(r.get("tag") or r.get("stamp") or r.get("model_id") or r.get("model") or
                    os.path.basename(path))
        entry = out.setdefault(beh, {"points": {}, "base": None})
        if "base" in label.lower() and _size_from(label, None) is None:
            entry["base"] = m
            continue
        N = _size_from(label, full_n)
        if N is None:
            continue
        entry["points"][N] = m
    return out


def _svg_chart(beh: str, entry: dict) -> str:
    pts = sorted(entry["points"].items())
    W, H, PL, PR, PT, PB = 620, 360, 56, 20, 40, 54
    x0, x1, y0, y1 = PL, W - PR, PT, H - PB
    if not pts:
        return f'<svg viewBox="0 0 {W} 80"><text x="10" y="30" fill="#8b949e">No sized points for {beh} yet.</text></svg>'
    ns = [n for n, _ in pts]
    lnmin, lnmax = math.log10(min(ns)), math.log10(max(ns) if max(ns) > min(ns) else min(ns) * 2)
    def sx(n): return x0 + (0 if lnmax == lnmin else (math.log10(n) - lnmin) / (lnmax - lnmin)) * (x1 - x0)
    def sy(v): return y1 - (max(0.0, min(100.0, v)) / 100.0) * (y1 - y0)
    parts = [f'<svg viewBox="0 0 {W} {H}" font-family="system-ui,Segoe UI,Roboto,sans-serif" font-size="12">',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="#0e1116"/>',
             f'<text x="{W/2:.0f}" y="22" fill="#e6edf3" font-size="14" text-anchor="middle" font-weight="700">Data-Efficiency — {beh}</text>']
    for gy in range(0, 101, 25):                                   # y gridlines + labels
        yy = sy(gy)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#232a33"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" fill="#8b949e" text-anchor="end">{gy}%</text>')
    for n in ns:                                                   # x ticks (per data point)
        xx = sx(n)
        parts.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+5}" stroke="#8b949e"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y1+20:.0f}" fill="#8b949e" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" fill="#adbac7" text-anchor="middle">training examples (N, log scale)</text>')
    base = entry.get("base") or {}
    for key, name, color in SERIES:
        if base.get(key) is not None:                              # dashed frontier/base reference
            by = sy(base[key])
            parts.append(f'<line x1="{x0}" y1="{by:.1f}" x2="{x1}" y2="{by:.1f}" stroke="{color}" stroke-dasharray="4 4" opacity="0.5"/>')
        poly = " ".join(f"{sx(n):.1f},{sy(m[key]):.1f}" for n, m in pts if m.get(key) is not None)
        if poly:
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for n, m in pts:
                if m.get(key) is not None:
                    parts.append(f'<circle cx="{sx(n):.1f}" cy="{sy(m[key]):.1f}" r="3.5" fill="{color}"/>')
    ly = PT + 6
    for key, name, color in SERIES:                                # legend
        parts.append(f'<rect x="{x1-150}" y="{ly-9}" width="11" height="11" fill="{color}"/>'
                     f'<text x="{x1-134}" y="{ly}" fill="#e6edf3">{name}</text>')
        ly += 18
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="results/reports")
    ap.add_argument("--out", default="results")
    ap.add_argument("--full-n", type=int, default=None,
                    help="treat a 'tuned'/'full' report with no N in its tag as this size")
    args = ap.parse_args()

    data = load(args.reports, args.full_n)
    if not data:
        print(f"No usable reports in {args.reports}/. Run eval.py (tag runs like --tag N500) and commit "
              f"results/reports/*.json, then re-run this.")
        return

    # Markdown table (paste into BRAINLIFT.md)
    print("\n## Data-Efficiency\n")
    for beh, entry in data.items():
        print(f"### {beh}\n\n| N | Spec-adherence | Robustness | Over-refusal |\n|--:|--:|--:|--:|")
        if entry.get("base"):
            b = entry["base"]
            print(f"| base | {b.get('spec_adherence','?')} | {b.get('robustness','?')} | {b.get('over_refusal','?')} |")
        for n, m in sorted(entry["points"].items()):
            print(f"| {n} | {m.get('spec_adherence','?')} | {m.get('robustness','?')} | {m.get('over_refusal','?')} |")
        print()

    os.makedirs(args.out, exist_ok=True)
    svgs = {beh: _svg_chart(beh, entry) for beh, entry in data.items()}
    for beh, svg in svgs.items():
        open(os.path.join(args.out, f"data_efficiency_{beh}.svg"), "w").write(svg)
    html = ("<!doctype html><meta charset=utf-8><title>Data-Efficiency curve</title>"
            "<body style='background:#0e1116;margin:0;padding:16px'>"
            + "".join(f"<div style='max-width:640px;margin:0 auto 20px'>{s}</div>" for s in svgs.values())
            + "</body>")
    open(os.path.join(args.out, "data_efficiency_curve.html"), "w").write(html)
    print(f"wrote {args.out}/data_efficiency_curve.html + per-behavior .svg")

    try:                                                            # best-effort PNG
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for beh, entry in data.items():
            pts = sorted(entry["points"].items())
            if not pts:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            for key, name, color in SERIES:
                xs = [n for n, m in pts if m.get(key) is not None]
                ys = [m[key] for n, m in pts if m.get(key) is not None]
                ax.plot(xs, ys, "-o", label=name, color=color)
                if (entry.get("base") or {}).get(key) is not None:
                    ax.axhline(entry["base"][key], ls="--", color=color, alpha=0.5)
            ax.set_xscale("log"); ax.set_xlabel("training examples (N)"); ax.set_ylabel("%")
            ax.set_ylim(0, 100); ax.set_title(f"Data-Efficiency — {beh}"); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"data_efficiency_{beh}.png"), dpi=130)
        print(f"wrote {args.out}/data_efficiency_*.png (matplotlib)")
    except Exception as e:
        print(f"(PNG skipped — matplotlib not available: {type(e).__name__}. The SVG/HTML above is complete.)")


if __name__ == "__main__":
    main()
