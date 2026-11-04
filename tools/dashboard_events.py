"""Chart event markers for the dashboard, derived from a run's ``history.jsonl`` records (pure functions, no I/O in ``derive``).

``derive(records)`` returns ``{"events": [...], "comps": [...], "n": int}``:

* ``events``: ``{"u": update, "t": type, "x": text, "v": {numbers}, "lk": [likely cause texts]?, "top": [[component, delta]]?}``
  sorted by update.  Types (``TYPES``): ``ms`` first episode to reach a milestone, ``rec`` fastest-to-milestone record
  (needs the trainer that logs ``records``), ``path`` new best path progress, ``map`` new map, ``wo`` white-outs,
  ``abias`` battle A-bias reached 0, ``boost`` entropy boost on/off, ``intr`` intrinsic-weight decay step, ``ecoef`` entropy
  coefficient reached its floor, ``badge``, ``lvl`` new highest level, ``dex`` new species seen/owned, ``rck`` route
  checkpoints (first save / first reset share), ``spike`` an outlier update of reward, entropy or value loss.
* ``comps``: ``[update, {component: net reward}]`` per update, for the "why did reward change" breakdown.

Every field is optional: an old run without a field simply produces no event of that type.  A spike's ``lk`` lists the
other events within +-1 update (a correlation, not proof), and a reward spike's ``top`` the components that changed most
against the mean of the previous 8 updates.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

MILESTONE_NAMES = ["Route 29", "Cherrygrove", "Route 30", "Route 31", "Violet gate", "Violet City", "Violet Gym", "Falkner beaten"]
TYPES = ("ms", "rec", "path", "map", "wo", "abias", "boost", "intr", "ecoef", "badge", "lvl", "dex", "rck", "spike")
# Cause priority when a spike coincides with an event (first = most likely to explain a jump).
CAUSE_ORDER = ("wo", "ms", "rec", "path", "map", "badge", "lvl", "dex", "abias", "boost", "intr", "ecoef", "rck")
SPIKE_METRICS = (("mean_reward", "reward"), ("entropy", "entropy"), ("value_loss", "value loss"))
PATH_STEP = 3            # emit a "best path progress" event only for gains of at least this many tiles
INTR_STEPS = (0.75, 0.5, 0.25, 0.1)
SPIKE_SIGMA = 3.0
SPIKE_WINDOW = 30
SPIKE_MIN_HISTORY = 10


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else None


def _spikes(vals: list[float | None]) -> list[tuple[int, float, float, float]]:
    """(index, value, rolling median, z) of outliers: > SPIKE_SIGMA robust sigmas from the median of the previous window."""
    out = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        prev = [x for x in vals[max(0, i - SPIKE_WINDOW):i] if x is not None]
        if len(prev) < SPIKE_MIN_HISTORY:
            continue
        med = statistics.median(prev)
        mad = statistics.median([abs(x - med) for x in prev])
        sigma = max(1.4826 * mad, 0.05 * abs(med), 1e-6)   # floor: a flat series must not flag tiny wiggles
        z = (v - med) / sigma
        if abs(z) > SPIKE_SIGMA:
            out.append((i, v, med, z))
    # at most one spike per ~12 updates: keep the strongest
    cap = max(8, len(vals) // 12)
    if len(out) > cap:
        out = sorted(sorted(out, key=lambda s: -abs(s[3]))[:cap])
    return out


def derive(records: list[dict]) -> dict:
    recs = [r for r in records if isinstance(r, dict) and r.get("update") is not None]
    ev: list[dict] = []

    def add(u: int, t: str, text: str, **v: Any) -> None:
        ev.append({"u": int(u), "t": t, "x": text, "v": {k: x for k, x in v.items() if x is not None}})

    mets = [r.get("metrics") if isinstance(r.get("metrics"), dict) else {} for r in recs]
    funs = [r.get("fun") if isinstance(r.get("fun"), dict) else {} for r in recs]
    ups = [int(r["update"]) for r in recs]

    # --- fastest-to-milestone records logged by the trainer (new data only)
    for r in recs:
        for e in r.get("records") or []:
            try:
                name = e.get("name") or MILESTONE_NAMES[int(e["m"])]
                scope = "from start" if e.get("scope") == "start" else "any start"
                text = (f"First time reaching {name}: {e['steps']} steps ({scope})" if e.get("first") or e.get("prev") is None
                        else f"New record: {name} in {e['steps']} steps (was {e['prev']}, {e['steps'] - e['prev']:+d}; {scope})")
                add(r["update"], "rec", text, steps=e.get("steps"), prev=e.get("prev"), env=e.get("env"))
            except (KeyError, TypeError, ValueError, IndexError):
                continue

    prev_reach: list | None = None
    path_emit: float | None = None
    abias_seen_pos = False
    boost_prev: float | None = None
    intr_done: set[float] = set()
    ecoef_series = [_num(m.get("entropy_coef")) for m in mets]
    rck_size_done = rck_share_done = False
    maps_prev = badges_prev = lvl_prev = None
    dex_prev: tuple | None = None
    names_prev: set[str] = set()
    wo_prev: int | None = None
    wo_hist: list[int] = []

    for i, (r, m, f) in enumerate(zip(recs, mets, funs)):
        u = ups[i]
        # (1) milestone first reached by anyone: ms_reach_cum 0/None -> > 0
        reach = m.get("ms_reach_cum")
        if isinstance(reach, list):
            if prev_reach is not None and i > 0:
                for k, val in enumerate(reach):
                    was = prev_reach[k] if k < len(prev_reach) else None
                    if (val or 0) > 0 and not (was or 0) > 0:
                        add(u, "ms", f"First episode to reach {MILESTONE_NAMES[k] if k < len(MILESTONE_NAMES) else 'milestone ' + str(k)}",
                            share=_num(val))
            prev_reach = reach
        # (2) best path progress
        pb = _num(m.get("path_best_max_cum"))
        if pb is not None:
            if path_emit is None:
                path_emit = pb
            elif pb >= path_emit + PATH_STEP:
                add(u, "path", f"New best path progress: {pb:g}/319 tiles", tiles=pb, prev=path_emit)
                path_emit = pb
        # (3) new maps
        mp = _num(m.get("maps_discovered"))
        names = [str(x[0]) for x in (f.get("first") or []) if isinstance(x, (list, tuple)) and x]
        if mp is not None and maps_prev is not None and mp > maps_prev:
            fresh = [n for n in names if n not in names_prev]
            label = ", ".join(fresh[:3]) + (f" +{len(fresh) - 3}" if len(fresh) > 3 else "") if fresh else f"maps discovered {maps_prev:g} -> {mp:g}"
            add(u, "map", f"New map: {label}" if fresh else label, maps=mp)
        if mp is not None:
            maps_prev = mp
        names_prev |= set(names)
        # (4) white-outs
        n_wo = None
        if _num(m.get("whiteout_episodes")) is not None:
            n_wo = int(m["whiteout_episodes"])
        elif _num(f.get("wo")) is not None:
            n_wo = int(f["wo"]) - wo_prev if wo_prev is not None else None
        if _num(f.get("wo")) is not None:
            wo_prev = int(f["wo"])
        pen = (r.get("components") or {}).get("neg", {}).get("whiteout") if isinstance(r.get("components"), dict) else None
        if n_wo is None and _num(pen) is not None and pen < 0:
            add(u, "wo", f"White-out penalty {pen:g}", penalty=pen)
        elif n_wo and n_wo > 0:
            typical = statistics.median(wo_hist[-10:]) if len(wo_hist) >= 3 else 0
            wave = n_wo >= max(3, 2 * typical)   # well above the recent norm: a "wave", the only white-outs named as a cause
            add(u, "wo", (f"White-out wave: {n_wo} white-outs (typical {typical:g})" if wave else f"{n_wo} white-out{'s' if n_wo != 1 else ''}"),
                n=n_wo, penalty=_num(pen), wave=wave or None)
        if n_wo is not None:
            wo_hist.append(n_wo)
        # (5) schedule events
        ab = _num(m.get("battle_a_bias"))
        if ab is not None:
            if ab > 1e-9:
                abias_seen_pos = True
            elif abias_seen_pos:
                add(u, "abias", "Battle A-button bias decayed to 0")
                abias_seen_pos = False
        bo = _num(m.get("entropy_boost"))
        if bo is not None:
            if boost_prev is not None and bo != boost_prev:
                add(u, "boost", "Entropy boost switched " + ("ON (anti-collapse)" if bo else "OFF"), entropy=_num(m.get("entropy")))
            boost_prev = bo
        iw = _num(m.get("intrinsic_weight"))
        if iw is not None and i > 0:
            for thr in INTR_STEPS:
                if iw <= thr and thr not in intr_done:
                    intr_done.add(thr)
                    add(u, "intr", f"Exploration bonus weight decayed to {iw:.2f} (below {thr:g})", weight=iw)
        # (6) badge / level / species
        bd = _num(m.get("badges"))
        if bd is not None and badges_prev is not None and bd > badges_prev:
            add(u, "badge", f"Badge #{int(bd)} earned", badges=bd)
        if bd is not None:
            badges_prev = bd
        lv = _num(f.get("lvl"))
        if lv is not None and lvl_prev is not None and lv > lvl_prev:
            add(u, "lvl", f"New highest Pokemon level: {int(lv)}", level=lv)
        if lv is not None:
            lvl_prev = lv
        dx = (_num(f.get("own")), _num(f.get("seen")))
        if dx[0] is not None and dex_prev is not None:
            if dx[0] > (dex_prev[0] or 0):
                add(u, "dex", f"New species caught (owned {int(dx[0])}, seen {int(dx[1] or 0)})", own=dx[0], seen=dx[1])
            elif (dx[1] or 0) > (dex_prev[1] or 0):
                add(u, "dex", f"New species seen (owned {int(dx[0])}, seen {int(dx[1] or 0)})", own=dx[0], seen=dx[1])
        if dx[0] is not None:
            dex_prev = dx
        # (7) route checkpoints
        rs = _num(m.get("route_ckpt_size"))
        if rs and rs > 0 and not rck_size_done and i > 0:
            rck_size_done = True
            add(u, "rck", "First route checkpoint saved", size=rs)
        sh = _num(m.get("route_ckpt_reset_share"))
        if sh and sh > 0 and not rck_share_done and i > 0:
            rck_share_done = True
            add(u, "rck", f"Episodes first started from route checkpoints ({sh * 100:.0f}% of resets)", share=sh)

    # entropy coefficient floor: lowest value, reached and then flat for >= 10 updates
    ec = [(i, v) for i, v in enumerate(ecoef_series) if v is not None]
    if len(ec) >= 12:
        lo = min(v for _, v in ec)
        if lo < ec[0][1] * 0.99:
            idx = next((i for i, v in ec if all(w <= lo * 1.001 for _, w in ec if _ >= i)), None)
            if idx is not None and sum(1 for i, _ in ec if i >= idx) >= 10:
                add(ups[idx], "ecoef", f"Entropy coefficient reached its floor ({lo:g})", coef=lo)

    # (8) spikes with a likely cause
    comps: list[list] = []
    for r in recs:
        c = r.get("components") if isinstance(r.get("components"), dict) else {}
        net: dict[str, float] = {}
        for side in ("pos", "neg"):
            for k, v in (c.get(side) or {}).items():
                if _num(v) is not None:
                    net[k] = round(net.get(k, 0.0) + float(v), 2)
        comps.append([int(r["update"]), net])
    base = list(ev)
    by_u: dict[int, list[dict]] = {}
    for e in base:
        by_u.setdefault(e["u"], []).append(e)
    for key, label in SPIKE_METRICS:
        for i, v, med, z in _spikes([_num(m.get(key)) for m in mets]):
            u = ups[i]
            near = [e for d in (-1, 0, 1) for e in by_u.get(u + d, []) if e["t"] != "wo" or e["v"].get("wave")]
            near.sort(key=lambda e: CAUSE_ORDER.index(e["t"]) if e["t"] in CAUSE_ORDER else 99)
            item = {"u": u, "t": "spike", "x": f"{label.capitalize()} {'spike' if z > 0 else 'drop'}: {v:.3g} vs typical {med:.3g}",
                    "v": {"metric": key, "value": round(v, 4), "median": round(med, 4), "z": round(z, 1)}}
            if near:
                item["lk"] = [e["x"] for e in near[:2]]
            if key == "mean_reward" and i >= 1:
                cur = comps[i][1]
                prior = [c[1] for c in comps[max(0, i - 8):i]]
                if prior:
                    names = set(cur) | {k for p in prior for k in p}
                    delta = {k: cur.get(k, 0.0) - sum(p.get(k, 0.0) for p in prior) / len(prior) for k in names}
                    ordered = sorted(delta.items(), key=lambda kv: kv[1])
                    top = [[k, round(d, 2)] for k, d in (ordered[:1] + ordered[-1:]) if abs(d) >= 0.05]
                    if top:
                        item["top"] = top
            ev.append(item)
    ev.sort(key=lambda e: (e["u"], TYPES.index(e["t"]) if e["t"] in TYPES else 99))
    return {"events": ev, "comps": comps, "n": len(recs)}


_cache: dict[str, tuple[tuple[int, int], dict]] = {}


def run_events(run_dir: Path) -> dict:
    """``derive`` over ``<run_dir>/history.jsonl`` (cached while the file is unchanged; a partial last line is skipped)."""
    hist = Path(run_dir) / "history.jsonl"
    try:
        st = hist.stat()
    except OSError:
        return {"events": [], "comps": [], "n": 0}
    sig = (st.st_size, int(st.st_mtime_ns))
    hit = _cache.get(str(hist))
    if hit and hit[0] == sig:
        return hit[1]
    recs = []
    with open(hist, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r.pop("heat", None)
            r.pop("envs", None)
            r.pop("episodes", None)
            recs.append(r)
    out = derive(recs)
    _cache[str(hist)] = (sig, out)
    return out
