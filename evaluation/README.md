# Pipeline evaluation

Measures snap detection, player detection, team/position assignment and the
WR SET call on a fixed test set of 50 clips: 25 Wide and 25 Sideline, picked at
random (seed 370). The clips used while developing the role and WR SET rules
are excluded.

Two selected clips were later removed: **Sideline 630 and 633** turned out to
be scoreboard clips misfiled under Sideline. They stay in `clips.csv` so the
seeded selection is unchanged, but are listed in `EXCLUDED` in
`run_pipeline.py`, so they are not processed, put in the sheets or scored.
The results list them too. The test set is therefore **48 clips: 25 Wide, 23
Sideline**.

## Steps

All commands run from the project root with the project's `.venv` Python.

1. `python evaluation/run_pipeline.py`: picks the clips (`clips.csv`) and runs
   the full pipeline on them, the same commands the app's **Process** button
   uses. Processed clips are skipped, so it is safe to re-run.
2. `python evaluation/make_review_pack.py`: writes one image per clip to
   `images/` and creates the two sheets. Answers already typed into the sheets
   are kept if you run it again.
3. **Label** the sheets by hand (below).
4. `python evaluation/score.py`: prints every metric and writes `results.md`.

## The images

`images/<clip>.png` is the predicted snap frame. Every box the player detector
found is drawn with a number and the role the pipeline gave it:

| Tag | Meaning | Color |
|---|---|---|
| OL / QB / RB / TE / WR | offensive role | amber / red / magenta / purple / green |
| D | defense | gray |
| `x` | box dropped by the pipeline (judged a duplicate, fragment or extra) | thin white |

The banner shows the predicted snap frame and the WR SET.

## Filling in `clips_sheet.csv` (one row per clip)

| Column | What to enter |
|---|---|
| `true_snap_frame` | **Required.** The frame the ball is snapped, or simply `good` / `bad` (whether the predicted frame is close enough that the rest of the pipeline is unaffected). With a frame number, within ±10 frames counts as good. Open the clip in the app and scrub to the snap (the Virtual Field panel shows the frame number). A clip only counts as labeled once this is filled in. |
| `players_on_field` | Players **visible in the frame** at the snap. The field always has 22, but the camera often cuts off a wide receiver or deep safety. Start from 22, subtract anyone out of frame, and note who in `notes` (e.g. "1 WR off frame"). Don't count referees. |
| `missed_players` | Players on the field with **no box** in the image. |
| `true_wr_set` | Blank if `pred_wr_set` is right. Otherwise the correct split, e.g. `3x1`, `2x2` or `3x2`. Count WRs and TEs on each side of the ball. |
| `wr_failure_stage` | Only when the WR SET is wrong or missing: the **first** stage that went wrong. One of `snap`, `detection`, `team`, `role`, `homography`, `other`. If it is wrong only because a receiver is out of frame, use `other` and say so in `notes`. |
| `notes` | Anything unusual (motion at the snap, blurry, wrong camera). |

If the predicted snap frame is badly wrong (the players aren't set), still fill
in `true_snap_frame`, leave the rest of the row blank, and set
`wr_failure_stage` to `snap`.

## Filling in `boxes_sheet.csv` (one row per numbered box)

`pred_role` is the pipeline's answer; `yolo_label` is what the position model
alone said, kept as a baseline. Fill in `true_role` **only where the prediction
is wrong**. A blank means the prediction is correct.

| `true_role` | Use for |
|---|---|
| `OL`, `QB`, `RB`, `TE`, `WR`, `D` | the box's correct team/role |
| `REF` | a referee |
| `OTHER` | anyone not on the field (sideline, crowd) |
| `DUP` | a second box on a player who already has one |

For a box tagged `x` (dropped), a blank means dropping it was right. If it is
actually a player who should have been kept, enter their role.

## What gets measured

- **Snap:** share of usable predicted snap frames; with frame numbers, also absolute error in frames and ms and the share within ±3 and ±10 frames.
- **Player detection:** recall (players found ÷ on field), precision (real players ÷ boxes), and false boxes per clip.
- **Team and position:** team accuracy, and offensive role accuracy compared with the position model alone, with confusion tables.
- **WR SET:** exact split and name accuracy, scored twice (WRs + TEs as the app writes it, and WRs only), confusion tables, and the stage where each failure started (from `wr_failure_stage`, or derived from the box labels when it is blank).

Each metric is reported for all clips and separately for Wide and Sideline.
