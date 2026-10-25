"""Slide-ready figures from a finished run (after scripts/train.py or evaluate.py).

    python scripts/plot_results.py                 # runs/full -> runs/full/reports/figures/
    python scripts/plot_results.py quick
    python scripts/plot_results.py runs/archive/20261025-213000 --format png,svg

Figures (each one message; captions and the numbers behind them go to figures/figures.md):
    fig1_ablation          crisis recall (95% CI) and false-crisis rate for the 5 modes      <- the headline
    fig2_confusion_full    4x4 confusion matrix of the shipped design (row %)
    fig3_by_subtype        where the errors are: recall per crisis subtype, false crisis per safe subtype
    fig4_cost              LLM call rate and mean latency per mode (why the cascade is worth it)
    fig5_threshold         classifier p_crisis thresholds and the grey zone (validation set)
    fig6_sessions          session tracker traces from the multi-turn tests (fast up, slow down, crisis hold)
    fig7_data              train / val / test composition by level
    fig8_reliability       calibration before/after temperature scaling (appendix)

Reads only what the run already contains; nothing is re-run. Design rules: one hue per chart with the
shipped mode emphasised, status colours only for the four levels, thin marks, hairline grid.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
import warnings
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from _common import ROOT, RUNS, read_jsonl

warnings.filterwarnings("ignore", message="findfont")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------------------- palette (see dataviz skill)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, DEEMPH = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
LEVEL_COLOR = {0: "#0ca30c", 1: "#fab219", 2: "#ec835a", 3: "#d03b3b"}   # status palette: safe .. crisis
LEVEL_NAME = ["safe", "tense", "overwhelmed", "crisis"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
MODES = ["rules", "classifier", "rules+classifier", "rules+llm", "full"]
MODE_LABEL = {"rules": "rules\nonly", "classifier": "classifier\nonly", "rules+classifier": "rules +\nclassifier",
              "rules+llm": "rules +\nLLM", "full": "full\n(shipped)"}
SHIPPED = "full"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def save(fig, out: Path, name: str, formats: list[str], dpi: int, written: list[str]) -> None:
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    written.append(name)


def bar_labels(ax, bars, values, fmt=pct, dy=0.012, color=INK2):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt(v), ha="center", va="bottom", fontsize=9, color=color)


def rounded_bars(ax, x, heights, colors, width=0.56):
    """Thin bars with a hairline surface gap between neighbours."""
    return ax.bar(x, heights, width=width, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)


# ---------------------------------------------------------------------------- figures
def fig_ablation(res: dict, meta: dict):
    modes = [m for m in MODES if m in res]
    x = np.arange(len(modes))
    colors = [BLUE if m == SHIPPED else DEEMPH for m in modes]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"wspace": 0.32})

    rec = [res[m]["crisis_recall"] for m in modes]
    lo = [res[m]["crisis_recall"] - res[m]["crisis_recall_ci95"][0] for m in modes]
    hi = [res[m]["crisis_recall_ci95"][1] - res[m]["crisis_recall"] for m in modes]
    bars = rounded_bars(a, x, rec, colors)
    a.errorbar(x, rec, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for bb, v, h in zip(bars, rec, hi):
        a.text(bb.get_x() + bb.get_width() / 2, v + h + 0.02, pct(v), ha="center", va="bottom", fontsize=9, color=INK2)
    n3 = sum(res[modes[0]]["confusion"][3])
    a.set_title("Crisis recall  (higher is better)", loc="left")
    a.set_ylim(0, 1.12)
    a.set_yticks([0, .25, .5, .75, 1]); a.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    a.set_xticks(x); a.set_xticklabels([MODE_LABEL[m] for m in modes])

    fc = [res[m]["false_crisis_rate_noncrisis"] for m in modes]
    n0 = int(sum(sum(res[modes[0]]["confusion"][i]) for i in range(3)))
    bars = rounded_bars(b, x, fc, colors)
    bar_labels(b, bars, fc, dy=0.006)
    b.set_ylim(0, max(0.15, max(fc) * 1.6))
    b.set_title("False crisis on non-crisis messages  (lower is better)", loc="left")
    b.set_xticks(x); b.set_xticklabels([MODE_LABEL[m] for m in modes])
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, f"Test set: {meta['test_file']} ({n3} crisis + {n0} non-crisis messages) · LLM: {meta['llm']} · "
             "error bars: 95% Wilson CI · blue = the shipped design (rules -> classifier -> LLM only when needed)",
             fontsize=8.5, color=MUTED)
    return fig


def fig_confusion(m: dict, mode: str, meta: dict):
    cm = np.array(m["confusion"], dtype=float)
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.grid(False)
    ax.imshow(row, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ[2:]), vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            c = int(cm[i, j])
            if c == 0:
                continue
            ax.text(j, i, f"{c}\n{row[i, j]:.0%}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if row[i, j] > 0.55 else INK, fontweight="semibold" if i == j else "normal")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"{i} {n}" for i, n in enumerate(LEVEL_NAME)])
    ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(LEVEL_NAME)])
    ax.set_xlabel("predicted level"); ax.set_ylabel("true level")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix, {mode} mode  (n={int(cm.sum())})")
    # the two cells that matter most
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor=LEVEL_COLOR[3], linewidth=2))
    ax.text(3.52, 2.5, "crisis caught", fontsize=8, color=LEVEL_COLOR[3], va="top", ha="left", clip_on=False)
    missed = int(cm[3, :3].sum())
    ax.text(3.52, 2.78, f"missed: {missed}", fontsize=8.5, color=INK2, va="top", ha="left", clip_on=False)
    fig.text(0.01, -0.02, f"Cell = count and share of the true level's row · {meta['llm']}", fontsize=8.5, color=MUTED)
    return fig


def fig_by_subtype(res: dict, preds_csv: Path, test: list[dict], mode: str):
    """Left: recall per crisis subtype. Right: false-crisis rate per non-crisis subtype."""
    import csv

    with open(preds_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    col = f"pred_{mode}"
    if not rows or col not in rows[0]:
        return None
    by: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        by.setdefault(r["subtype"], []).append((int(r["gold"]), int(r[col])))
    crisis = sorted(((st, v) for st, v in by.items() if v[0][0] == 3),
                    key=lambda kv: (sum(p == 3 for _, p in kv[1]) / len(kv[1]), -len(kv[1])))
    safe = sorted(((st, v) for st, v in by.items() if v[0][0] < 3),
                  key=lambda kv: (-sum(p == 3 for _, p in kv[1]) / len(kv[1]), kv[0]))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.7})
    names = [f"{st}  (n={len(v)})" for st, v in crisis]
    vals = [sum(p == 3 for _, p in v) / len(v) for _, v in crisis]
    y = np.arange(len(crisis))
    a.barh(y, vals, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, v, (_, items) in zip(y, vals, crisis):
        a.text(min(v, 0.98) + 0.015 if v < 0.9 else v - 0.015, yi, f"{sum(p == 3 for _, p in items)}/{len(items)}",
               va="center", ha="left" if v < 0.9 else "right", fontsize=8.5, color=INK2 if v < 0.9 else "#ffffff")
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8.5)
    a.set_xlim(0, 1.0); a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a.set_title(f"Crisis recall by subtype  ({mode} mode)", loc="left")
    a.grid(axis="y", visible=False); a.invert_yaxis()

    names = [f"{st}  (n={len(v)})" for st, v in safe]
    vals = [sum(p == 3 for _, p in v) / len(v) for _, v in safe]
    lv1 = [sum(p >= 1 for _, p in v) / len(v) for _, v in safe]
    y = np.arange(len(safe))
    b.barh(y, vals, height=0.6, color=LEVEL_COLOR[3], edgecolor=SURFACE, linewidth=1.5, zorder=3, label="flagged as crisis")
    for yi, v, (_, items) in zip(y, vals, safe):
        b.text(v + 0.01, yi, f"{sum(p == 3 for _, p in items)}/{len(items)}", va="center", ha="left", fontsize=8.5, color=INK2)
    b.set_yticks(y); b.set_yticklabels(names, fontsize=8.5)
    b.set_xlim(0, max(0.3, max(vals + [0]) * 1.4)); b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_title("False crisis by non-crisis subtype", loc="left")
    b.grid(axis="y", visible=False); b.invert_yaxis()
    fig.text(0.01, -0.03, "Left: of the true crises in each subtype, how many were caught. "
             "Right: of the non-crisis messages in each subtype, how many were wrongly treated as a crisis "
             "(blunt_literal / idiom_negative / special_interest are the autistic-style groups).", fontsize=8.5, color=MUTED, wrap=True)
    return fig


def fig_cost(res: dict, meta: dict):
    modes = [m for m in MODES if m in res]
    x = np.arange(len(modes))
    colors = [BLUE if m == SHIPPED else DEEMPH for m in modes]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"wspace": 0.28})
    calls = [res[m]["llm_call_rate"] for m in modes]
    bars = rounded_bars(a, x, calls, colors)
    bar_labels(a, bars, calls, dy=0.012)
    a.set_ylim(0, 1.12); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a.set_title("Share of messages sent to the LLM")
    lat = [res[m]["latency_ms_mean"] / 1000 for m in modes]
    p95 = [res[m]["latency_ms_p95"] / 1000 for m in modes]
    bars = rounded_bars(b, x, lat, colors)
    bar_labels(b, bars, lat, fmt=lambda v: f"{v:.1f} s" if v >= 0.1 else f"{1000 * v:.0f} ms", dy=max(lat) * 0.02 + 0.01)
    b.scatter(x, p95, marker="_", s=260, color=INK2, zorder=4, linewidths=1.2)
    b.set_ylim(0, max(p95 + [0.1]) * 1.25)
    b.set_title("Mean latency per message  (dash = 95th percentile)")
    b.set_ylabel("seconds")
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[m] for m in modes]); ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, f"LLM: {meta['llm']}. Rules and classifier answer in milliseconds; the LLM is the only slow part, "
             "so sending fewer messages to it is what keeps the app responsive.", fontsize=8.5, color=MUTED)
    return fig


def fig_threshold(model_dir: Path):
    th = load(model_dir / "thresholds.json")
    logits, labels = model_dir / "val_logits.npy", model_dir / "val_labels.npy"
    if not th or not logits.exists():
        return None
    from safety.classifier import softmax

    z, y = np.load(logits), np.load(labels).astype(int)
    p3 = softmax(z / th["temperature"])[:, 3]
    is3 = y == 3
    grid = np.linspace(0.0, 0.995, 200)   # at 1.0 nothing is flagged, which only draws a cliff
    rec = np.array([((p3 >= t) & is3).sum() / max(1, is3.sum()) for t in grid])
    prec = np.array([((p3 >= t) & is3).sum() / max(1, (p3 >= t).sum()) for t in grid])
    flagged = np.array([(p3 >= t).mean() for t in grid])
    lo, hi = th["crisis_lo"], th["crisis_hi"]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    hi_draw = min(hi, 1.0)
    ax.axvspan(lo, hi_draw, color=SEQ[0], zorder=1)
    ax.plot(grid, rec, color=BLUE, linewidth=2, zorder=3, label="crisis recall")
    ax.plot(grid, prec, color=ORANGE, linewidth=2, zorder=3, label="crisis precision")
    ax.plot(grid, flagged, color=AQUA, linewidth=1.4, zorder=3, label="share of messages flagged")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    for t, name in ((lo, f"crisis_lo = {lo:.2f}"), (hi, f"crisis_hi = {hi:.2f}")):
        if t <= 1:
            ax.axvline(t, color=INK2, linewidth=0.9, zorder=2)
            ax.text(t + (0.005 if t < 0.1 else 0), 1.06, name, ha="left" if t < 0.1 else "center", fontsize=8.5, color=INK2)
    if lo >= 0.12:
        ax.text(lo / 2, 0.5, "no crisis flag\nfrom the classifier", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text((lo + hi_draw) / 2, 0.5, "grey zone:\nask the LLM", ha="center", va="center", fontsize=8.5, color=INK2)
    if hi <= 1:
        ax.text((hi + 1) / 2, 0.5, "classifier\ndecides alone", ha="center", va="center", fontsize=8.5, color=INK2)
    else:
        ax.text(0.98, 0.35, "no threshold reaches the precision target:\nevery possible crisis goes to the LLM",
                ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("threshold on p(crisis) after temperature scaling")
    ax.set_title(f"How the classifier's crisis thresholds were set  (validation set, n={len(y)}, {int(is3.sum())} crises)")
    fig.text(0.01, -0.17, f"Temperature {th['temperature']:.2f} (ECE {th['ece_before']:.3f} -> {th['ece_after']:.3f}). "
             f"crisis_lo keeps >= 98% validation recall; crisis_hi is the first threshold with >= 95% precision. "
             f"About {th['val_llm_call_rate']:.0%} of validation messages fall in the grey zone.", fontsize=8.5, color=MUTED)
    return fig


def parse_multiturn(path: Path) -> list[dict]:
    """Turn multiturn_results.md back into conversations: [{id, desc, steps:[{msg, session, action, text}|{event}]}]."""
    if not path.exists():
        return []
    convs, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(PASS|FAIL)\s+(\w+): (.*)$", line)
        if m:
            cur = {"id": m.group(2), "desc": m.group(3), "ok": m.group(1) == "PASS", "steps": []}
            convs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+(ok|FAIL)\s+msg=(\d) session=(\d) (\S+)\s+\| (.*?)(\s+expected .*)?$", line)
        if m:
            cur["steps"].append({"msg": int(m.group(2)), "session": int(m.group(3)), "action": m.group(4), "text": m.group(5).strip()})
            continue
        m = re.match(r"^\s+\[event\] (\w+)", line)
        if m:
            cur["steps"].append({"event": m.group(1)})
    return convs


def fig_sessions(convs: list[dict], want=("mt02", "mt04", "mt05")):
    picked = [c for c in convs if c["id"] in want] or convs[:3]
    if not picked:
        return None
    fig, axes = plt.subplots(len(picked), 1, figsize=(10, 2.1 * len(picked) + 0.6), sharex=False,
                             gridspec_kw={"hspace": 0.75})
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, picked):
        xs, sess, msg, events = [], [], [], []
        t = 0
        for s in c["steps"]:
            if "event" in s:
                events.append((t + 0.5, s["event"]))
                continue
            t += 1
            xs.append(t); sess.append(s["session"]); msg.append(s["msg"])
        ax.step(xs, sess, where="post", color=BLUE, linewidth=2, zorder=3)
        ax.plot(xs[-1:], sess[-1:], color=BLUE, marker="o", markersize=5, zorder=3)
        ax.scatter(xs, msg, s=48, facecolors=[LEVEL_COLOR[v] for v in msg], edgecolors=SURFACE, linewidths=1.5, zorder=4)
        for x_ev, name in events:
            ax.axvline(x_ev, color=INK2, linewidth=0.8, zorder=2)
            ax.text(x_ev, 3.35, {"release_crisis_hold": 'button: "I am safe now"', "user_chose_continue": 'button: "Go back to practice"'}.get(name, name),
                    fontsize=8, color=INK2, ha="center", va="bottom")
        ax.set_ylim(-0.4, 3.9); ax.set_yticks(range(4)); ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(LEVEL_NAME)], fontsize=8.5)
        ax.set_xlim(0.5, max(xs) + 0.5); ax.set_xticks(xs)
        ax.set_xticklabels([textwrap.shorten(s["text"], 16, placeholder="…") for s in c["steps"] if "event" not in s], fontsize=7.5, rotation=0)
        ax.set_title(f"{c['id']}: {c['desc']}", loc="left")
        ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2, label="conversation level (what the app acts on)"),
               plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=6, label="level of each message (coloured by level)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.005), ncol=2)
    fig.text(0.01, -0.01, "The conversation level rises at once, falls only after calm messages, and stays in crisis "
             "until the user presses the button. From data/multiturn_tests.jsonl, run through the whole agent.", fontsize=8.5, color=MUTED)
    return fig


def fig_data(files: dict[str, Path]):
    rows = [(name, read_jsonl(p)) for name, p in files.items() if p.exists()]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 2.6))
    xmax = max(len(d) for _, d in rows)
    for i, (name, data) in enumerate(rows):
        counts = [sum(r["level"] == lv for r in data) for lv in range(4)]
        left = 0
        for lv, c in enumerate(counts):
            ax.barh(i, c, left=left, height=0.55, color=LEVEL_COLOR[lv], edgecolor=SURFACE, linewidth=2, zorder=3)
            if c >= 0.045 * xmax:   # label only segments wide enough to hold the number
                ax.text(left + c / 2, i, str(c), ha="center", va="center", fontsize=8.5,
                        color=INK if lv == 1 else "#ffffff")
            left += c
        ax.text(left + 8, i, f"n = {len(data)}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows]); ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Data by level", loc="left")
    ax.legend(handles=[Patch(color=LEVEL_COLOR[lv], label=f"{lv} {LEVEL_NAME[lv]}") for lv in range(4)],
              loc="lower right", ncol=4, bbox_to_anchor=(1, 1.02))
    fig.text(0.01, -0.06, "train / val: synthetic messages (data/synthetic). test: team-written, never used for training or tuning.",
             fontsize=8.5, color=MUTED)
    return fig


def fig_reliability(model_dir: Path):
    th = load(model_dir / "thresholds.json")
    if not th or not (model_dir / "val_logits.npy").exists():
        return None
    from safety.classifier import softmax

    z, y = np.load(model_dir / "val_logits.npy"), np.load(model_dir / "val_labels.npy").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"wspace": 0.3})
    for ax, T, title, ece in zip(axes, (1.0, th["temperature"]), ("before temperature scaling", f"after (T = {th['temperature']:.2f})"),
                                 (th["ece_before"], th["ece_after"])):
        p = softmax(z / T)
        conf, pred = p.max(1), p.argmax(1)
        edges = np.linspace(0, 1, 11)
        accs, mids, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                accs.append((pred[m] == y[m]).mean()); mids.append((lo + hi) / 2); ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.9, zorder=4)
        ax.bar(mids, accs, width=0.085, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, a_, n in zip(mids, accs, ns):
            ax.text(x, a_ + 0.02, str(n), ha="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
        ax.set_xlabel("confidence (max probability)"); ax.set_ylabel("accuracy in bin")
        ax.set_title(f"{title}   ECE {ece:.3f}")
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, "Bars on the diagonal = the model's confidence means what it says. Numbers = messages per bin (validation set).",
             fontsize=8.5, color=MUTED)
    return fig


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="full", help="full | quick | a run folder (default: full)")
    ap.add_argument("--mode", default=SHIPPED, help="mode for the confusion / subtype figures (default: full)")
    ap.add_argument("--format", default="png", help="png,svg,pdf")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    run = RUNS / args.run if args.run in ("full", "quick") else Path(args.run).resolve()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    reports = run / "reports"
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written, notes = [], []

    ev = load(reports / "eval_results.json")
    if not ev:
        raise SystemExit(f"no {reports / 'eval_results.json'}: run scripts/train.py (or evaluate.py) first")
    res, meta = ev["results"], ev["meta"]
    mode = args.mode if args.mode in res else next(reversed(res))
    if meta["llm"].startswith("mock"):
        notes.append("LLM was `mock`: the rules+llm and full rows are NOT real results; redo with LM Studio before the slides.")
    if meta.get("llm_errors"):
        notes.append(f"LLM calls FAILED during evaluate ({meta['llm_errors']}): the rules+llm and full rows are NOT real results. "
                     "Start LM Studio, then `python scripts/train.py --start-at evaluate`.")

    save(fig_ablation(res, meta), out, "fig1_ablation", formats, args.dpi, written)
    save(fig_confusion(res[mode], mode, meta), out, f"fig2_confusion_{mode.replace('+', '_')}", formats, args.dpi, written)
    test_path = ROOT / "data" / meta["test_file"]
    test = read_jsonl(test_path) if test_path.exists() else []
    if (reports / "predictions.csv").exists():
        f = fig_by_subtype(res, reports / "predictions.csv", test, mode)
        if f:
            save(f, out, "fig3_by_subtype", formats, args.dpi, written)
    else:
        notes.append("no predictions.csv (older evaluate.py): fig3 skipped; re-run evaluate.py")
    save(fig_cost(res, meta), out, "fig4_cost", formats, args.dpi, written)

    model_dir = Path(meta["classifier"]) if meta.get("classifier") else None
    if model_dir and not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if model_dir and model_dir.exists():
        f = fig_threshold(model_dir)
        if f:
            save(f, out, "fig5_threshold", formats, args.dpi, written)
        f = fig_reliability(model_dir)
        if f:
            save(f, out, "fig8_reliability", formats, args.dpi, written)
    else:
        notes.append("no classifier folder recorded in eval_results.json: fig5 / fig8 skipped")

    convs = parse_multiturn(reports / "multiturn_results.md")
    f = fig_sessions(convs)
    if f:
        save(f, out, "fig6_sessions", formats, args.dpi, written)
    else:
        notes.append("no multiturn_results.md: fig6 skipped")

    f = fig_data({"train": ROOT / "data/synthetic/train.jsonl", "val": ROOT / "data/synthetic/val.jsonl", "test": test_path})
    if f:
        save(f, out, "fig7_data", formats, args.dpi, written)

    # captions + the numbers behind them, for the slide notes
    full = res.get(mode, {})
    lo, hi = full.get("crisis_recall_ci95", (float("nan"), float("nan")))
    lines = [f"# Figures for {run}", "", f"Test set {meta['test_file']} (n={meta['n_test']}), classifier `{meta.get('classifier')}`, LLM `{meta['llm']}`.", ""]
    lines += [f"> {n}" for n in notes] + ([""] if notes else [])
    lines += [
        f"- **fig1_ablation** - headline. {mode}: crisis recall {full.get('crisis_recall', 0):.2f} (95% CI {lo:.2f}-{hi:.2f}), "
        f"false crisis on non-crisis messages {full.get('false_crisis_rate_noncrisis', 0):.2f}. Say: each layer earns its place; "
        "rules alone miss indirect crises, the classifier alone over-flags, the cascade gets both.",
        f"- **fig2_confusion_{mode}** - where the {mode} mode is right and wrong, by level. Missed crises: {int(np.array(full.get('confusion', [[0]*4]*4))[3, :3].sum())}.",
        "- **fig3_by_subtype** - which crisis subtypes are missed (indirect wording, no risk words) and which safe groups are over-flagged "
        f"(autistic-style false crisis {full.get('style_false_crisis_rate', 0):.2f}, tense-but-safe {full.get('tense_safe_false_crisis_rate', 0):.2f}).",
        f"- **fig4_cost** - {mode} sends {full.get('llm_call_rate', 0):.0%} of messages to the LLM, mean latency {full.get('latency_ms_mean', 0) / 1000:.1f} s "
        f"(p95 {full.get('latency_ms_p95', 0) / 1000:.1f} s). The cascade is what keeps the app responsive.",
        "- **fig5_threshold** - how crisis_lo / crisis_hi were chosen on the validation set and what the grey zone means (design slide).",
        "- **fig6_sessions** - the session tracker: fast up, slow down, crisis hold until the button (demo slide or backup).",
        "- **fig7_data** - what the models were trained and tested on (one slide, early).",
        "- **fig8_reliability** - appendix: calibration before/after temperature scaling.",
        "", "All modes:", "",
        "| mode | crisis recall (95% CI) | false crisis (non-crisis) | style false crisis | tense-safe false crisis | macro-F1 | LLM calls | mean latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, r in res.items():
        l, h = r["crisis_recall_ci95"]
        lines.append(f"| {m} | {r['crisis_recall']:.2f} ({l:.2f}-{h:.2f}) | {r['false_crisis_rate_noncrisis']:.2f} | "
                     f"{r['style_false_crisis_rate']:.2f} | {r['tense_safe_false_crisis_rate']:.2f} | {r['macro_f1']:.2f} | "
                     f"{r['llm_call_rate']:.2f} | {r['latency_ms_mean'] / 1000:.1f} s |")
    (out / "figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} figures to {out}: " + ", ".join(written))
    for n in notes:
        print("note:", n)
    print(f"captions and numbers: {out / 'figures.md'}")


if __name__ == "__main__":
    main()
