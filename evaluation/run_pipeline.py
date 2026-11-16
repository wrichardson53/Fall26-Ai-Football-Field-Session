#!/usr/bin/env python3
"""
Step 1 of the evaluation: pick the test clips and run the full processing
pipeline on them (the same commands the app's Process button runs).

    python evaluation/run_pipeline.py            # select clips (first run) + process
    python evaluation/run_pipeline.py --select-only

Clips are chosen at random with a fixed seed from the Wide and Sideline video
folders, excluding the clips used while developing the role/WR SET rules. The
selection is written to evaluation/clips.csv and reused on later runs, so the
test set never changes. Clips whose outputs already exist are skipped.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT, "evaluation")
CLIPS_CSV = os.path.join(EVAL_DIR, "clips.csv")
LOG_DIR = os.path.join(EVAL_DIR, "logs")
CACHE = os.path.join(PROJECT, "cache")
VIDEO_ROOT = os.path.join(os.path.dirname(PROJECT), "videos")

FOLDERS = {"Wide": 25, "Sideline": 25}
SEED = 370
# Clips looked at while tuning the rules; kept out of the test set.
DEV_CLIPS = {f"Wide - Clip {n}" for n in
             ("004", "114", "117", "266", "275", "426", "564", "778", "921", "999", "1012", "1017", "1021")}
# Clips removed from the test set after selection, with the reason. They stay in
# clips.csv so the seeded selection is unchanged, but are not processed or scored.
EXCLUDED = {
    "Sideline - Clip 630": "mislabeled: scoreboard clip filed under Sideline",
    "Sideline - Clip 633": "mislabeled: scoreboard clip filed under Sideline",
}


def select_clips():
    rng = random.Random(SEED)
    rows = []
    for folder, count in FOLDERS.items():
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(VIDEO_ROOT, folder))
                       if f.lower().endswith(".mp4"))
        names = [n for n in names if n not in DEV_CLIPS]
        for name in sorted(rng.sample(names, min(count, len(names)))):
            rows.append({"folder": folder, "clip": name})
    with open(CLIPS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "clip"])
        w.writeheader()
        w.writerows(rows)
    print(f"Selected {len(rows)} clips (seed {SEED}) -> {CLIPS_CSV}")
    return rows


def load_clips(include_excluded=False):
    """The test set; clips in EXCLUDED are left out unless asked for."""
    with open(CLIPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows if include_excluded else [r for r in rows if r["clip"] not in EXCLUDED]


def paths(folder, clip):
    c = os.path.join(CACHE, folder)
    return {
        "video": os.path.join(VIDEO_ROOT, folder, f"{clip}.mp4"),
        "det": os.path.join(c, "players", f"{clip}_detection.json"),
        "pos": os.path.join(c, "positions", f"{clip}_position.json"),
        "snap": os.path.join(c, "snap_detection", f"{clip}_snap_detection.json"),
        "ym": os.path.join(c, "yard_markers", f"{clip}_yard_markers.json"),
        "cor": os.path.join(c, "correspondence", f"{clip}_correspondence.json"),
        "hom": os.path.join(c, "homography", f"{clip}_normalized_positions.json"),
    }


def run_clip(folder, clip):
    p = paths(folder, clip)
    py = sys.executable
    s = lambda name: os.path.join(PROJECT, "scripts", name)
    steps = [
        ("player detection (stride 3)", [py, s("playerDetection.py"), "--video", p["video"], "--output", p["det"], "--stride", "3"]),
        ("snap detection", [py, s("snapDetection.py"), "--player-detections", p["det"], "--output", p["snap"]]),
        ("player detection (snap frame)", [py, s("playerDetection.py"), "--video", p["video"], "--output", p["det"], "--snap-detection", p["snap"]]),
        ("position detection", [py, s("positionDetection.py"), "--video", p["video"], "--output", p["pos"], "--snap-detection", p["snap"]]),
        ("yard marker detection", [py, s("yardMarkerDetection.py"), "--video", p["video"], "--output", p["ym"], "--snap-detection", p["snap"]]),
        ("correspondence points", [py, s("autoCorrespondancePoints.py"), "--detection-json", p["ym"], "--output", p["cor"], "--confidence", "0.7", "--per-frame"]),
        ("homography", [py, s("perFrameHomographyTransform.py"), "--position-detections", p["det"], "--correspondence-points", p["cor"], "--output", p["hom"]]),
        ("static process", [py, s("staticProcess.py"), "--video-name", clip, "--folder-name", folder, "--cache-dir", CACHE]),
    ]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{clip}.log"), "w") as log:
        for name, cmd in steps:
            log.write(f"\n===== {name} =====\n")
            log.flush()
            r = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return f"failed at {name} (exit {r.returncode})"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    clips = load_clips() if os.path.exists(CLIPS_CSV) else select_clips()
    if args.select_only:
        return
    for i, row in enumerate(clips, 1):
        p = paths(row["folder"], row["clip"])
        if os.path.exists(p["hom"]):
            print(f"[{i}/{len(clips)}] {row['clip']}: already processed")
            continue
        t = time.time()
        status = run_clip(row["folder"], row["clip"])
        print(f"[{i}/{len(clips)}] {row['clip']}: {status} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
