"""
Assign offensive position labels from field geometry at the snap.

The YOLO position model is decent at telling offense from defense but unreliable
at QB / RB / WR / TE, and pixel-space rules break down because the wide camera is
oblique (yard lines run diagonally across the frame). Instead, every player is
looked up in the homography output (field yards: x along the field, y across it)
and roles come from where they line up relative to the line of scrimmage:

    oline         always 5: the tightest group of 5 neighbours on the line
                  around the center
    qb            directly behind the center (pistol/shotgun/under center)
    running_back  other backfield players inside the tackle box
    tight_end     on/just off the line, next to a tackle
    wide_receiver every other offensive player
"""

import statistics

NON_OFFENSE = ('defense', 'ref', 'yard_marker')

MATCH_PX = 40                # max bbox-center distance to pair a detection with a homography point
TEAM_FLIP_YD = 1.5           # this far past the LOS on the wrong side => wrong team label
TEAM_FLIP_SURE_YD = 4.0      # beyond this, flip even if the defense already has 11
ON_LINE_YD = 1.5             # within this of the LOS counts as on the line
ON_LINE_RELAXED_YD = 2.5     # widened when fewer than 5 players are found on the line
OLINE_COUNT = 5
OLINE_MAX_LATERAL_YD = 4.5   # tackles line up within this of the center
QB_MAX_LATERAL_YD = 3.0
QB_FALLBACK_LATERAL_YD = 5.0 # used when nobody is within QB_MAX_LATERAL_YD (sideline views blur lateral distance)
QB_MAX_DEPTH_YD = 9.0
RB_MIN_DEPTH_YD = 1.5
RB_MAX_LATERAL_YD = 5.0      # tackle box half-width
TE_BEYOND_TACKLE_YD = 3.0
MIN_TACKLE_OFFSET_YD = 3.0   # tackle's minimum lateral distance from the center
MAX_RB = 2
TEAM_SIZE = 11
MAX_ELIGIBLE = TEAM_SIZE - OLINE_COUNT - 1  # backs + receivers: everyone but the line and the QB
EXTRA_BACK_MIN_DEPTH_YD = 0.75              # depth behind the line for an extra player to count as a back
DUPLICATE_OVERLAP = 0.3      # same-team boxes overlapping this much (IoU) are likely one player
FRAGMENT_HEIGHT_RATIO = 0.6  # a box this much shorter than the team median is part of a player


def _match_field_points(detections, field_points):
    """Return {id(det): (x, y)} for detections paired with a homography point."""
    matched = {}
    limit_sq = MATCH_PX ** 2
    for det in detections:
        bbox = det.get('bbox') or {}
        cx, cy = bbox.get('center_x'), bbox.get('center_y')
        if cx is None or cy is None:
            continue
        best, best_d = None, limit_sq
        for fp in field_points:
            ob = fp.get('original_bbox') or {}
            ox, oy = ob.get('center_x'), ob.get('center_y')
            if ox is None or oy is None:
                continue
            d = (cx - ox) ** 2 + (cy - oy) ** 2
            if d < best_d:
                best, best_d = fp, d
        if best is not None:
            npos = best.get('normalized_position') or {}
            if npos.get('x') is not None and npos.get('y') is not None:
                matched[id(det)] = (float(npos['x']), float(npos['y']))
    return matched


def _box_iou(a, b):
    a, b = a.get('bbox') or {}, b.get('bbox') or {}
    try:
        ix = max(0.0, min(a['x2'], b['x2']) - max(a['x1'], b['x1']))
        iy = max(0.0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
        inter = ix * iy
        union = (a['x2'] - a['x1']) * (a['y2'] - a['y1']) + (b['x2'] - b['x1']) * (b['y2'] - b['y1']) - inter
    except KeyError:
        return 0.0
    return inter / union if union > 0 else 0.0


def _box_area(d):
    b = d.get('bbox') or {}
    return b.get('width', 0) * b.get('height', 0)


def _trim_team(team, distance_to_ball):
    """Cut a team down to TEAM_SIZE; return the detections removed.

    Duplicate boxes on one player (common in the line pile) go first: the smaller
    box of the most-overlapping pair. Next, fragment boxes much shorter than the
    rest of the team (part of a player in the pile). Only then drop whoever is
    farthest from the ball (refs, sideline players) - never first, since on
    offense that is the widest receiver.
    """
    removed = []
    while len(team) > TEAM_SIZE:
        pairs = [(_box_iou(a, b), a, b) for i, a in enumerate(team) for b in team[i + 1:]]
        overlap, a, b = max(pairs, key=lambda p: p[0])
        heights = [(d.get('bbox') or {}).get('height', 0) for d in team]
        median_h = statistics.median(heights) if heights else 0
        shortest = min(team, key=lambda d: (d.get('bbox') or {}).get('height', 0))
        if overlap >= DUPLICATE_OVERLAP:
            extra = a if _box_area(a) < _box_area(b) else b
        elif median_h and (shortest.get('bbox') or {}).get('height', 0) < FRAGMENT_HEIGHT_RATIO * median_h:
            extra = shortest
        else:
            extra = max(team, key=distance_to_ball)
        team.remove(extra)
        removed.append(extra)
    return removed


def label_unmatched_by_line(detections, field_points):
    """Give detections with no class (players the position model missed) a team.

    Uses the labeled players to find which way the offense faces and where the
    line of scrimmage is; an unlabeled player on the offense's side of it is
    offense (role fixed later by assign_roles_by_field_geometry), otherwise
    defense. Unlabeled detections that can't be placed are removed in place.
    """
    pos = _match_field_points(detections, field_points)

    def cls(d):
        return (d.get('class') or '').lower()

    labeled = [d for d in detections if d.get('class') and id(d) in pos]
    off_x = [pos[id(d)][0] for d in labeled if cls(d) not in NON_OFFENSE]
    def_x = [pos[id(d)][0] for d in labeled if cls(d) == 'defense']
    if off_x and def_x:
        direction = 1.0 if statistics.median(off_x) < statistics.median(def_x) else -1.0
        oline_t = [direction * pos[id(d)][0] for d in labeled if cls(d) == 'oline']
        # Line of scrimmage: the linemen if labeled, else the front of the offense
        los = statistics.median(oline_t) if oline_t else max(direction * x for x in off_x)
        for d in detections:
            if not d.get('class') and id(d) in pos:
                d['class'] = 'wide_receiver' if direction * pos[id(d)][0] < los + 0.5 else 'defense'
    detections[:] = [d for d in detections if d.get('class')]


def assign_roles_by_field_geometry(detections, field_points):
    """Relabel offensive detections in place using field coordinates.

    detections:   snap-frame detections with 'class' and pixel 'bbox'
    field_points: homography entries for the same frame, each with
                  'original_bbox' (pixels) and 'normalized_position' (yards)

    Each team is capped at 11; extra detections (duplicates, refs) are removed
    from *detections* in place.

    Returns True if roles were assigned, False if there wasn't enough data
    (caller should fall back to another method).
    """
    pos = _match_field_points(detections, field_points)
    players = [d for d in detections
               if id(d) in pos and (d.get('class') or '').lower() not in ('ref', 'yard_marker')]
    offense = [d for d in players if (d.get('class') or '').lower() not in NON_OFFENSE]
    defense = [d for d in players if (d.get('class') or '').lower() == 'defense']
    if len(offense) < 6 or not defense:
        return False

    # Orient so the offense attacks toward +t.
    off_med = statistics.median(pos[id(d)][0] for d in offense)
    def_med = statistics.median(pos[id(d)][0] for d in defense)
    direction = 1.0 if off_med < def_med else -1.0
    t = {id(d): direction * pos[id(d)][0] for d in players}
    lat = {id(d): pos[id(d)][1] for d in players}

    # Line of scrimmage: the YOLO 'oline' players if it found some, otherwise
    # the front of the offense (7 must be on the line, so use the 7th deepest).
    oline_yolo = [d for d in offense if (d.get('class') or '').lower() == 'oline']
    if len(oline_yolo) >= 2:
        los = statistics.median(t[id(d)] for d in oline_yolo)
        center_y = statistics.median(lat[id(d)] for d in oline_yolo)
    else:
        front = sorted((t[id(d)] for d in offense), reverse=True)
        los = front[min(6, len(front) - 1)]
        center_y = statistics.median(lat[id(d)] for d in offense)

    # Nobody can be on the wrong side of the ball at the snap: fix team labels.
    # Offense past the line is flipped deepest-first, and only while one side is
    # off from 11 (homography noise near the far sideline can push real receivers a
    # yard or two past the line) unless they are clearly too far past it.
    for d in sorted(offense, key=lambda d: t[id(d)], reverse=True):
        beyond = t[id(d)] - los
        if beyond > TEAM_FLIP_SURE_YD or (beyond > TEAM_FLIP_YD and (len(defense) < 11 or len(offense) > 11)):
            offense.remove(d)
            defense.append(d)
            d['class'] = 'defense'
    for d in list(defense):
        if t[id(d)] < los - TEAM_FLIP_YD:
            defense.remove(d)
            offense.append(d)
    # More than 11 defenders means offensive linemen in the pile were mislabeled;
    # hand the extras closest to the offense's side back, if they're at the line.
    if len(defense) > 11:
        for d in sorted(defense, key=lambda d: t[id(d)])[:len(defense) - 11]:
            if t[id(d)] < los + 1.0 and len(offense) < TEAM_SIZE:
                defense.remove(d)
                offense.append(d)

    # At most 11 per side.
    def distance_to_ball(d):
        return ((t[id(d)] - los) ** 2 + (lat[id(d)] - center_y) ** 2) ** 0.5
    for d in _trim_team(offense, distance_to_ball) + _trim_team(defense, distance_to_ball):
        detections.remove(d)
    if len(offense) < 6:
        return False

    # Interior line: there are always 5 linemen. Take the 5 neighbouring players on
    # the line whose spread is tightest and closest to the center; widen the line
    # band if the snap frame doesn't have 5 players on it.
    for band in (ON_LINE_YD, ON_LINE_RELAXED_YD):
        on_line = sorted((d for d in offense if abs(t[id(d)] - los) <= band),
                         key=lambda d: lat[id(d)])
        if len(on_line) >= OLINE_COUNT:
            break
    if len(on_line) > OLINE_COUNT:
        def window_cost(i):
            window = on_line[i:i + OLINE_COUNT]
            spread = lat[id(window[-1])] - lat[id(window[0])]
            offset = abs(statistics.median(lat[id(d)] for d in window) - center_y)
            return spread + offset
        start = min(range(len(on_line) - OLINE_COUNT + 1), key=window_cost)
        linemen = on_line[start:start + OLINE_COUNT]
    else:
        linemen = on_line
    # A wide receiver split out on the line is never an interior lineman: drop
    # anyone too far sideways from the middle of the group.
    if linemen:
        mid_y = statistics.median(lat[id(d)] for d in linemen)
        linemen = [d for d in linemen if abs(lat[id(d)] - mid_y) <= OLINE_MAX_LATERAL_YD]
    if len(linemen) >= 3:
        center_y = statistics.median(lat[id(d)] for d in linemen)
        los = statistics.median(t[id(d)] for d in linemen)
    # Fragments in the pile can make the line look narrower than it is; real
    # tackles are ~3 yd from the center, so don't let the TE zone shrink below that.
    tackle_offset = max(max((abs(lat[id(d)] - center_y) for d in linemen), default=0.0),
                        MIN_TACKLE_OFFSET_YD)

    remaining = [d for d in offense if d not in linemen]

    def depth(d):
        return los - t[id(d)]

    def lateral(d):
        return abs(lat[id(d)] - center_y)

    # QB: behind the center. Lateral offset matters most (shotgun with an offset RB),
    # depth breaks ties (pistol / I-form, where the QB is the shallower one).
    qb_candidates = [d for d in remaining
                     if 0.3 <= depth(d) <= QB_MAX_DEPTH_YD and lateral(d) <= QB_MAX_LATERAL_YD]
    if not qb_candidates:
        qb_candidates = [d for d in remaining
                         if 0.3 <= depth(d) <= QB_MAX_DEPTH_YD and lateral(d) <= QB_FALLBACK_LATERAL_YD]
    qb = min(qb_candidates, key=lambda d: lateral(d) + 0.3 * depth(d), default=None)
    if qb is not None:
        remaining.remove(qb)

    backs = sorted((d for d in remaining
                    if depth(d) >= RB_MIN_DEPTH_YD and lateral(d) <= RB_MAX_LATERAL_YD),
                   key=lateral)[:MAX_RB]
    remaining = [d for d in remaining if d not in backs]

    # Rules of football: 5 linemen + the QB leave at most 5 backs and receivers.
    # Extras usually come from a lineman the line search missed, or a running back
    # pushed out of the box by a compressed (sideline) view. Fill the missing line
    # spots from players on the line nearest the center, then make the deepest
    # player a back, and drop whatever is still left over.
    while len(backs) + len(remaining) > MAX_ELIGIBLE and remaining:
        on_line = [d for d in remaining if depth(d) < RB_MIN_DEPTH_YD]
        behind = [d for d in remaining if depth(d) >= EXTRA_BACK_MIN_DEPTH_YD]
        if len(linemen) < OLINE_COUNT and on_line:
            d = min(on_line, key=lateral)
            linemen.append(d)
        elif len(backs) < MAX_RB and behind:
            d = min(behind, key=lateral)
            backs.append(d)
        else:
            d = min(remaining, key=lateral)
            detections.remove(d)
        remaining.remove(d)

    tight_ends = [d for d in remaining
                  if depth(d) < RB_MIN_DEPTH_YD
                  and lateral(d) <= tackle_offset + TE_BEYOND_TACKLE_YD]

    for d in linemen:
        d['class'] = 'oline'
    if qb is not None:
        qb['class'] = 'qb'
    for d in backs:
        d['class'] = 'running_back'
    for d in remaining:
        d['class'] = 'tight_end' if d in tight_ends else 'wide_receiver'
    return True
