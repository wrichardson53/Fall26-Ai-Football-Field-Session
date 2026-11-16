#!/usr/bin/env python3
"""
Step 2 of the evaluation: build what a person needs to label the test clips.

    python evaluation/make_review_pack.py

For every clip in evaluation/clips.csv this writes
  evaluation/images/<clip>.png   snap frame, every player-detector box numbered,
                                 colored by the role the pipeline gave it
and two sheets to fill in (existing answers are kept if the sheets are rebuilt):
  evaluation/clips_sheet.csv     one row per clip  (snap, player count, WR SET)
  evaluation/boxes_sheet.csv     one row per box   (team / role)

Roles are computed exactly as the app does: player-detector boxes, labels
borrowed from the position model, then app/positionRoles.py.
"""

import csv
import json
import os
import sys

import cv2

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "app"))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT, "evaluation"))

import positionRoles as pr  # noqa: E402
import staticProcess as sp  # noqa: E402
from run_pipeline import CACHE, CLIPS_CSV, EVAL_DIR, load_clips, paths  # noqa: E402

IMG_DIR = os.path.join(EVAL_DIR, "images")
CLIP_SHEET = os.path.join(EVAL_DIR, "clips_sheet.csv")
BOX_SHEET = os.path.join(EVAL_DIR, "boxes_sheet.csv")

ROLE = {"oline": "OL", "qb": "QB", "running_back": "RB", "tight_end": "TE",
        "wide_receiver": "WR", "defense": "D", "ref": "REF"}
COLOR = {"OL": (0, 200, 255), "QB": (0, 0, 255), "RB": (255, 0, 255), "TE": (255, 0, 160),
         "WR": (0, 255, 0), "D": (150, 150, 150), "dropped": (255, 255, 255)}

CLIP_FIELDS = ["folder", "clip", "image", "status", "pred_snap_frame", "pred_boxes", "pred_wr_set",
               "true_snap_frame", "players_on_field", "missed_players", "true_wr_set",
               "wr_failure_stage", "notes"]
CLIP_ANSWERS = CLIP_FIELDS[7:]
BOX_FIELDS = ["folder", "clip", "box", "pred_role", "yolo_label", "true_role", "notes"]
BOX_ANSWERS = ["true_role", "notes"]


def load_existing(path, key_fields, answer_fields):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {tuple(r[k] for k in key_fields): {a: r.get(a, "") for a in answer_fields}
                for r in csv.DictReader(f)}


def snap_boxes(p, snap):
    with open(p["det"]) as f:
        frames = json.load(f).get("frames", [])
    for fr in frames:
        if fr.get("frame_number") == snap:
            return fr.get("detections", [])
    return []


def predict_roles(boxes, p, snap):
    """Return {box index: role} using the app's role pipeline (boxes it drops are absent)."""
    pos_dets, _ = sp.get_positions_at_snap(p["pos"], snap)
    hom = sp.get_normalized_positions_at_snap(p["hom"], snap)
    if not hom:
        return {}
    roster = sp._roster_from_player_detections(pos_dets, hom)
    if not pr.assign_roles_by_field_geometry(roster, hom):
        return {}
    roles = {}
    for i, b in enumerate(boxes):
        bb = b["bbox"]
        for r in roster:
            rb = r["bbox"]
            if abs(rb["center_x"] - bb["center_x"]) < 1 and abs(rb["center_y"] - bb["center_y"]) < 1:
                roles[i] = ROLE.get((r.get("class") or "").lower(), r.get("class"))
                break
    return roles


def yolo_labels(boxes, p, snap):
    pos_dets, _ = sp.get_positions_at_snap(p["pos"], snap)
    out = {}
    for i, b in enumerate(boxes):
        bb = b["bbox"]
        best, best_d = "-", pr.MATCH_PX ** 2
        for d in pos_dets:
            db = d["bbox"]
            dist = (db["center_x"] - bb["center_x"]) ** 2 + (db["center_y"] - bb["center_y"]) ** 2
            if dist < best_d:
                best, best_d = ROLE.get(d["class"].lower(), d["class"]), dist
        out[i] = best
    return out


def draw(frame, boxes, roles, banner, offset=(0, 0), scale=1.0, only=None):
    """Draw numbered boxes; offset/scale map frame coordinates onto a zoomed crop,
    and `only` limits drawing to those box indices."""
    ox, oy = offset

    def pt(x, y):
        return int((x - ox) * scale), int((y - oy) * scale)

    for i, b in enumerate(boxes):
        if only is not None and i not in only:
            continue
        bb = b["bbox"]
        role = roles.get(i, "dropped")
        col = COLOR.get(role, (255, 255, 255))
        p1, p2 = pt(bb["x1"], bb["y1"]), pt(bb["x2"], bb["y2"])
        cv2.rectangle(frame, p1, p2, col, 1 if role == "dropped" else 2)
        tag = f"{i + 1} {role if role != 'dropped' else 'x'}"
        org = (p1[0], max(p1[1] - 5, 46))
        cv2.putText(frame, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(frame, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def zoom_on_line(frame, boxes, roles, banner, margin=110, target_width=1400):
    """Enlarged crop around the offensive line, where boxes overlap most."""
    core = [boxes[i]["bbox"] for i, r in roles.items() if r in ("OL", "QB", "TE")]
    if not core:
        return None
    h, w = frame.shape[:2]
    x1 = max(int(min(b["x1"] for b in core)) - margin, 0)
    y1 = max(int(min(b["y1"] for b in core)) - margin, 0)
    x2 = min(int(max(b["x2"] for b in core)) + margin, w)
    y2 = min(int(max(b["y2"] for b in core)) + margin, h)
    scale = min(target_width / max(x2 - x1, 1), 4.0)
    crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    inside = {i for i, b in enumerate(boxes)
              if b["bbox"]["x2"] > x1 and b["bbox"]["x1"] < x2 and b["bbox"]["y2"] > y1 and b["bbox"]["y1"] < y2}
    return draw(crop, boxes, roles, banner + " | zoom on the line", offset=(x1, y1), scale=scale, only=inside)


def main():
    if not os.path.exists(CLIPS_CSV):
        sys.exit("Run evaluation/run_pipeline.py first.")
    os.makedirs(IMG_DIR, exist_ok=True)
    old_clips = load_existing(CLIP_SHEET, ["clip"], CLIP_ANSWERS)
    old_boxes = load_existing(BOX_SHEET, ["clip", "box"], BOX_ANSWERS)
    clip_rows, box_rows = [], []

    for row in load_clips():
        folder, clip = row["folder"], row["clip"]
        p = paths(folder, clip)
        out = {"folder": folder, "clip": clip, "image": "", "status": "ok",
               "pred_snap_frame": "", "pred_boxes": "", "pred_wr_set": ""}
        snap = sp.get_snap_frame(p["snap"]) if os.path.exists(p["snap"]) else None
        if snap is None:
            out["status"] = "no snap detected"
        else:
            out["pred_snap_frame"] = snap
            boxes = snap_boxes(p, snap)
            out["pred_boxes"] = len(boxes)
            roles = predict_roles(boxes, p, snap) if os.path.exists(p["hom"]) else {}
            wr = sp._determine_receiver_set(clip, folder, CACHE) if roles else None
            out["pred_wr_set"] = wr or ""
            if not roles:
                out["status"] = "no roles (homography or role assignment failed)"
            labels = yolo_labels(boxes, p, snap) if os.path.exists(p["pos"]) else {}

            cap = cv2.VideoCapture(p["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, snap)
            ok, frame = cap.read()
            cap.release()
            if ok:
                banner = f"{clip} | predicted snap frame {snap} | WR SET: {wr or 'none'}"
                img = os.path.join(IMG_DIR, f"{clip}.png")
                zoom = zoom_on_line(frame.copy(), boxes, roles, banner)
                if zoom is not None:
                    cv2.imwrite(os.path.join(IMG_DIR, f"{clip} - line zoom.png"), zoom)
                cv2.imwrite(img, draw(frame, boxes, roles, banner))
                out["image"] = os.path.relpath(img, EVAL_DIR)
            for i in range(len(boxes)):
                key = (clip, str(i + 1))
                box_rows.append({"folder": folder, "clip": clip, "box": i + 1,
                                 "pred_role": roles.get(i, "dropped"), "yolo_label": labels.get(i, "-"),
                                 **old_boxes.get(key, {a: "" for a in BOX_ANSWERS})})
        clip_rows.append({**out, **old_clips.get((clip,), {a: "" for a in CLIP_ANSWERS})})
        print(f"{clip}: {out['status']} | snap {out['pred_snap_frame']} | {out['pred_boxes']} boxes | {out['pred_wr_set']}")

    for path, fields, rows in ((CLIP_SHEET, CLIP_FIELDS, clip_rows), (BOX_SHEET, BOX_FIELDS, box_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {CLIP_SHEET} ({len(clip_rows)} clips) and {BOX_SHEET} ({len(box_rows)} boxes)")


if __name__ == "__main__":
    main()
