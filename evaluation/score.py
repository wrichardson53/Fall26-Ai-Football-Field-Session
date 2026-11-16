#!/usr/bin/env python3
"""
Step 4 of the evaluation: score the pipeline against the filled-in sheets.

    python evaluation/score.py

Reads evaluation/clips_sheet.csv and evaluation/boxes_sheet.csv (see
evaluation/README.md for how to fill them in) and writes the metrics, split by
camera angle, to evaluation/results.md. Only labeled clips are scored: a clip
counts as labeled once its true_snap_frame is filled in.
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "evaluation"))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from run_pipeline import CACHE, EVAL_DIR, EXCLUDED, paths  # noqa: E402
from staticProcess import _determine_receiver_set, _receiver_set_name  # noqa: E402

CLIP_SHEET = os.path.join(EVAL_DIR, "clips_sheet.csv")
BOX_SHEET = os.path.join(EVAL_DIR, "boxes_sheet.csv")
RESULTS = os.path.join(EVAL_DIR, "results.md")

PLAYER_ROLES = ["OL", "QB", "RB", "TE", "WR", "D"]
OFFENSE = {"OL", "QB", "RB", "TE", "WR"}
NOT_PLAYER = {"REF", "OTHER", "DUP"}
FAILURE_STAGES = ["snap", "detection", "team", "role", "homography", "other"]
# Obvious typos seen in the hand labels.
ALIASES = {"WE": "WR", "ER": "WR"}
GOOD, BAD = {"good", "yes", "ok", "y"}, {"bad", "no", "n"}


def snap_frame_number(text):
    return int(text) if text.strip().lstrip("-").isdigit() else None


def snap_ok(c):
    """True/False when the true snap is known (a frame number within 10 frames,
    or a good/bad verdict), None if unknown."""
    t = c["true_snap_frame"].strip().lower()
    if t in GOOD:
        return True
    if t in BAD:
        return False
    n = snap_frame_number(t)
    if n is None or not c["pred_snap_frame"]:
        return None
    return abs(int(c["pred_snap_frame"]) - n) <= 10


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(n, d):
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def ratio(text):
    """'TRIPS (3x1)' or '3x1' -> (3, 1); None if no ratio."""
    m = re.search(r"(\d)\s*x\s*(\d)", text or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return max(a, b), min(a, b)


def name(r):
    return _receiver_set_name(r[0], r[1], r[0] + r[1]).split(" (")[0]


def fps_of(folder, clip):
    try:
        with open(paths(folder, clip)["det"]) as f:
            return float(json.load(f)["video_info"]["fps"]) or 30.0
    except Exception:
        return 30.0


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def confusion(pairs, labels):
    """pairs of (true, predicted) -> markdown table, rows = true."""
    c = Counter(pairs)
    cols = labels + sorted({p for _, p in pairs} - set(labels))
    rows = [[f"**{t}**"] + [c[(t, p)] or "" for p in cols] for t in labels if any(c[(t, p)] for p in cols)]
    return table(["true \\ predicted"] + cols, rows)


def failure_stage(c, bs, true_role):
    """First stage that broke the WR SET (WRs-only scoring), or None if it is right."""
    if snap_ok(c) is False:
        return "snap"
    t = c["true_wr_set"].strip()
    truth = ratio(c["pred_wr_set"]) if t.lower() == "pred_wr_set" else (ratio(t) if t else ratio(c["pred_wr_only"]))
    pred = ratio(c["pred_wr_only"])
    if truth is not None and pred == truth:
        return None
    given = c["wr_failure_stage"].strip().lower()
    if given:
        return given
    if pred is None:
        return "homography / role assignment failed"
    receivers = [b for b in bs if true_role(b) == "WR"]
    if any(b["pred_role"] == "dropped" for b in receivers):
        return "detection (receiver box dropped)"
    if any(b["pred_role"] == "D" for b in receivers) or             any(b["pred_role"] == "WR" and true_role(b) in ("D", "REF", "OTHER", "DUP") for b in bs):
        return "team (defender/ref counted as WR or WR as defense)"
    if any(b["pred_role"] != true_role(b) and "WR" in (b["pred_role"], true_role(b)) for b in bs):
        return "role (offensive player given the wrong role)"
    if int(c["missed_players"] or 0) > 0:
        return "detection (player missed)"
    return "homography / side split"


def score(clips, boxes, title):
    md = [f"## {title}", ""]
    labeled = [c for c in clips if c["true_snap_frame"].strip()]
    md.append(f"Labeled clips: {len(labeled)} of {len(clips)}. "
              f"Pipeline produced roles for {sum(1 for c in clips if c['status'] == 'ok')} of {len(clips)}.")
    if not labeled:
        return md + ["", "_Nothing labeled yet._", ""]
    names = {c["clip"] for c in labeled}

    # --- Snap detection ---
    errs = []
    for c in labeled:
        n = snap_frame_number(c["true_snap_frame"])
        if c["pred_snap_frame"] and n is not None:
            e = int(c["pred_snap_frame"]) - n
            errs.append((e, 1000 * e / fps_of(c["folder"], c["clip"])))
    verdicts = [snap_ok(c) for c in labeled]
    md += ["", "### Snap detection", ""]
    md.append(table(["metric", "value"], [
        ["predicted snap usable (players set, rest of pipeline unaffected)",
         pct(sum(1 for v in verdicts if v), sum(1 for v in verdicts if v is not None))],
    ]))
    bad = [c["clip"] for c, v in zip(labeled, verdicts) if v is False]
    if bad:
        md += ["", "Unusable snap frames: " + ", ".join(bad)]
    if errs:
        md.append("")
        abs_f = sorted(abs(e) for e, _ in errs)
        md.append(table(["metric", "value"], [
            ["clips with a predicted snap", pct(len(errs), len(labeled))],
            ["mean absolute error", f"{sum(abs_f) / len(abs_f):.1f} frames ({sum(abs(m) for _, m in errs) / len(errs):.0f} ms)"],
            ["median absolute error", f"{abs_f[len(abs_f) // 2]} frames"],
            ["mean signed error (+ = late)", f"{sum(e for e, _ in errs) / len(errs):+.1f} frames"],
            ["within ±3 frames", pct(sum(1 for e in abs_f if e <= 3), len(errs))],
            ["within ±10 frames", pct(sum(1 for e in abs_f if e <= 10), len(errs))],
        ]))

    # --- Player detection (raw detector boxes at the predicted snap) ---
    cb = defaultdict(list)
    for b in boxes:
        if b["clip"] in names:
            cb[b["clip"]].append(b)

    def true_role(b):
        t = b["true_role"].strip().upper()
        return ALIASES.get(t, t) or b["pred_role"]

    # Clips whose snap was unusable have no box labels, so leave them out here.
    det_clips = [c for c in labeled if c["missed_players"].strip()]
    on_field = sum(int(c["players_on_field"] or 0) for c in det_clips)
    missed = sum(int(c["missed_players"] or 0) for c in det_clips)
    det_names = {c["clip"] for c in det_clips}
    all_boxes = [b for name_, bs in cb.items() if name_ in det_names for b in bs]
    false_raw = [b for b in all_boxes if true_role(b) in NOT_PLAYER]
    kept = [b for b in all_boxes if b["pred_role"] != "dropped"]
    false_kept = [b for b in kept if true_role(b) in NOT_PLAYER]
    # Clips where role assignment failed drop every box; only count clips that got roles.
    ok_clips = {c["clip"] for c in labeled if c["status"] == "ok"}
    ok_boxes = [b for b in all_boxes if b["clip"] in ok_clips]
    real_dropped = [b for b in ok_boxes if b["pred_role"] == "dropped" and true_role(b) in PLAYER_ROLES]
    false_ok = [b for b in ok_boxes if true_role(b) in NOT_PLAYER]
    md += ["", "### Player detection", ""]
    md.append(table(["metric", "value"], [
        ["recall (players found / players on field)", pct(on_field - missed, on_field)],
        ["detector precision (real players / boxes)", pct(len(all_boxes) - len(false_raw), len(all_boxes))],
        ["false boxes per clip (refs, sideline, duplicates)", f"{len(false_raw) / max(len(det_clips), 1):.2f}"],
        ["false boxes still shown after role cleanup (clips with roles)", f"{len(false_kept)} of {len(false_ok)}"],
        ["real players wrongly dropped by cleanup (clips with roles)",
         f"{len(real_dropped)} of {sum(1 for b in ok_boxes if true_role(b) in PLAYER_ROLES)}"],
    ]))
    md += ["", f"Over {len(det_clips)} clips with a usable snap frame.",
           "", "False boxes by type: " + ", ".join(f"{k} {v}" for k, v in Counter(true_role(b) for b in false_raw).items())]

    # --- Team and role (kept boxes that are real players) ---
    real = [b for b in kept if true_role(b) in PLAYER_ROLES]
    team_ok = sum(1 for b in real if (b["pred_role"] == "D") == (true_role(b) == "D"))
    off = [b for b in real if true_role(b) in OFFENSE]
    role_ok = sum(1 for b in off if b["pred_role"] == true_role(b))
    yolo_ok = sum(1 for b in off if b["yolo_label"] == true_role(b))
    md += ["", "### Team and position", ""]
    md.append(table(["metric", "value"], [
        ["team correct (offense vs defense), all players", pct(team_ok, len(real))],
        ["offensive role correct (final, after geometry rules)", pct(role_ok, len(off))],
        ["offensive role correct (position model alone, baseline)", pct(yolo_ok, len(off))],
    ]))
    md += ["", "Final roles, offensive players (rows = truth):", "",
           confusion([(true_role(b), b["pred_role"]) for b in off], PLAYER_ROLES)]
    md += ["", "Position model alone, offensive players (`-` = no label):", "",
           confusion([(true_role(b), b["yolo_label"]) for b in off], PLAYER_ROLES)]

    # --- WR SET ---
    md += ["", "### WR SET", "",
           "Truth is the hand-labeled split. The pipeline's WR SET counts WRs and TEs; "
           "the hand labels mostly count WRs only, so both are scored. "
           "End-to-end counts every labeled clip, with an unusable snap or a missing WR SET as wrong.", ""]
    rows, details = [], {}
    for label, key in (("WRs + TEs (as the app writes it)", "pred_wr_set"), ("WRs only", "pred_wr_only")):
        pairs, no_pred, no_truth = [], 0, 0
        for c in labeled:
            pred = ratio(c[key])
            t = c["true_wr_set"].strip()
            truth = ratio(c["pred_wr_set"]) if t.lower() == "pred_wr_set" else (ratio(t) if t else pred)
            if truth is None:
                no_truth += 1
                continue
            if pred is None:
                no_pred += 1
                continue
            pairs.append((truth, pred))
        exact = sum(1 for t, p in pairs if t == p)
        named = sum(1 for t, p in pairs if name(t) == name(p))
        rows.append([label, pct(exact, len(pairs)), pct(named, len(pairs)), pct(exact, len(labeled)), no_pred, no_truth])
        details[label] = pairs
    md.append(table(["receivers counted", "exact split (when produced)", "name (when produced)",
                     "exact split, end-to-end", "no WR SET produced", "unusable snap"], rows))
    for label, pairs in details.items():
        if pairs:
            md += ["", f"{label}, rows = truth:", "",
                   confusion([(f"{t[0]}x{t[1]}", f"{p[0]}x{p[1]}") for t, p in pairs],
                             sorted({f"{t[0]}x{t[1]}" for t, _ in pairs}, reverse=True))]

    fails = Counter(failure_stage(c, cb.get(c["clip"], []), true_role) for c in labeled)
    fails.pop(None, None)
    if fails:
        md += ["", "First failing stage for wrong or missing WR SETs (WRs-only scoring). "
               "Taken from `wr_failure_stage` when filled in, otherwise derived from the box labels:", "",
               table(["stage", "clips"], [[k, v] for k, v in fails.most_common()])]
    return md + [""]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not (os.path.exists(CLIP_SHEET) and os.path.exists(BOX_SHEET)):
        sys.exit("Run evaluation/make_review_pack.py first.")
    clips = [c for c in read(CLIP_SHEET) if c["clip"] not in EXCLUDED]
    boxes = [b for b in read(BOX_SHEET) if b["clip"] not in EXCLUDED]
    # The same WR SET rule, counting wide receivers only (how the sheets were labeled).
    import contextlib
    import io
    for c in clips:
        c["pred_wr_only"] = ""
        if c["pred_wr_set"]:
            with contextlib.redirect_stdout(io.StringIO()):
                c["pred_wr_only"] = _determine_receiver_set(c["clip"], c["folder"], CACHE,
                                                            receiver_roles=("wide_receiver",)) or ""
    md = ["# Pipeline evaluation results", "",
          "Test set: clips chosen at random (seed 370) from the Wide and Sideline folders, "
          "excluding clips used while developing the rules. Generated by `evaluation/score.py`.", ""]
    if EXCLUDED:
        md += ["Removed from the test set after selection:", ""]
        md += [f"- {clip}: {reason}" for clip, reason in sorted(EXCLUDED.items())]
        md += [""]
    md += score(clips, boxes, "All clips")
    for folder in sorted({c["folder"] for c in clips}):
        md += score([c for c in clips if c["folder"] == folder],
                    [b for b in boxes if b["folder"] == folder], f"{folder} only")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
