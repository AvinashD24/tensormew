"""Milestone success-rate bookkeeping: how often do episodes reach each story milestone?

A pure diagnostic (no effect on training).  The mean return is a poor "is it learning?" signal when many reward
terms move at once; the fraction of episodes that get past Route 29, Cherrygrove, ..., Violet Gym, Falkner is not.

Definitions (``MILESTONES`` order, milestone ``i`` has story stage ``i + 1``, see ``src.story_path.current_stage``):

* an episode is ELIGIBLE for milestone ``i`` when it started before it (``start_stage <= i``): an archive reset that
  starts in Violet City cannot "reach" Route 29;
* it REACHED milestone ``i`` when its furthest stage was ``>= i + 1``;
* ``reach[i] = reached / eligible`` over the finished episodes of the window (None when nothing was eligible);
* ``envs_at[i]`` = share of ALL envs whose running episode is currently at or above stage ``i + 1``;
* ``best_path`` = best path progress (tiles of the 309-tile story path, ``src.path_progress``) of each finished episode.

Staggered first episodes that were cut short (``partial``) are skipped, like the route-progress signal does.  No numpy.
"""

from __future__ import annotations

from src.story_path import MILESTONES, MILESTONE_COUNT

MILESTONE_NAMES = tuple(m.name for m in MILESTONES)
_PRETTY = {"route29": "Route 29", "cherrygrove": "Cherrygrove", "route30": "Route 30", "route31": "Route 31",
           "violet_gate": "Violet gate", "violet_city": "Violet City", "violet_gym": "Violet Gym", "beat_falkner": "Falkner beaten"}
MILESTONE_LABELS = tuple(_PRETTY.get(n, n.replace("_", " ").capitalize()) for n in MILESTONE_NAMES)   # for humans / the dashboard


def _rates(reached: list[int], eligible: list[int]) -> list[float | None]:
    return [round(r / e, 3) if e > 0 else None for r, e in zip(reached, eligible)]


class MilestoneStats:
    def __init__(self, n: int = MILESTONE_COUNT):
        self.n = int(n)
        self._reset_window()
        self.cum_reached = [0] * self.n
        self.cum_eligible = [0] * self.n
        self.cum_eps = 0
        self.cum_best_max = 0.0

    def _reset_window(self) -> None:
        self.win_reached = [0] * self.n
        self.win_eligible = [0] * self.n
        self.win_eps = 0
        self.win_best: list[float] = []

    def add_episode(self, start_stage, furthest, best_path=None, partial: bool = False) -> bool:
        """Record one finished episode; returns False when it was skipped (partial or missing data)."""
        if partial or start_stage is None or furthest is None:
            return False
        start_stage, furthest = int(start_stage), int(furthest)
        for i in range(self.n):
            if start_stage <= i:
                self.win_eligible[i] += 1
                self.cum_eligible[i] += 1
                if furthest >= i + 1:
                    self.win_reached[i] += 1
                    self.cum_reached[i] += 1
        self.win_eps += 1
        self.cum_eps += 1
        if best_path is not None:
            self.win_best.append(float(best_path))
            self.cum_best_max = max(self.cum_best_max, float(best_path))
        return True

    def end_update(self, env_furthest=()) -> dict:
        """Compact payload for the metrics / history / state.json of one update; starts a new window."""
        fur = [int(f) for f in env_furthest if f is not None]
        at = [round(sum(1 for f in fur if f >= i + 1) / len(fur), 3) if fur else None for i in range(self.n)]
        out = {
            "ms_reach": _rates(self.win_reached, self.win_eligible),
            "ms_elig": list(self.win_eligible),
            "ms_reach_cum": _rates(self.cum_reached, self.cum_eligible),
            "ms_envs_at": at,
            "ms_eps": self.win_eps,
            "ms_eps_cum": self.cum_eps,
            "path_best_mean": round(sum(self.win_best) / len(self.win_best), 2) if self.win_best else None,
            "path_best_max": round(max(self.win_best), 1) if self.win_best else None,
            "path_best_max_cum": round(self.cum_best_max, 1),
        }
        self._reset_window()
        return out


BASE_START_STATE = "new_bark_outside"   # the only start state whose step counts are comparable "from the start"
RECORD_LOG_MAX = 50


def is_base_start(state_name) -> bool:
    """True for the plain New Bark start state (not a swarm / battle / route-checkpoint archive reset)."""
    s = str(state_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    if s.endswith(".state"):
        s = s[:-len(".state")]
    return s == BASE_START_STATE


class MilestoneRecords:
    """Fastest-to-each-milestone records, in episode steps (env steps since the start of an episode).

    * ``start``: episodes from the BASE start state only (``new_bark_outside``).  Archive resets (swarm / battle / route
      checkpoint) begin closer to the goal and would give fake records, and a staggered partial first episode starts with a
      step offset, so neither counts.  These are the headline records.
    * ``any``: the fewest steps since the start of ANY full-length episode, whatever its start (labelled "any start").
    A record is ``{steps, update, env, prev, delta}`` (``prev``/``delta`` None for a first arrival).  ``log`` keeps the last
    ``RECORD_LOG_MAX`` events ``{u, m, name, steps, prev, env, scope, first}`` of both scopes.
    """

    def __init__(self, n: int = MILESTONE_COUNT, log_max: int = RECORD_LOG_MAX):
        self.n = int(n)
        self.start: list[dict | None] = [None] * self.n
        self.any: list[dict | None] = [None] * self.n
        self.log: list[dict] = []
        self.log_max = int(log_max)
        self._new: list[dict] = []   # events since the last take_new()

    def _try(self, table: list, scope: str, m: int, steps: int, update: int, env: int) -> dict | None:
        cur = table[m]
        if cur is not None and steps >= cur["steps"]:
            return None
        prev = None if cur is None else cur["steps"]
        table[m] = {"steps": steps, "update": int(update), "env": int(env), "prev": prev,
                    "delta": None if prev is None else steps - prev}
        ev = {"u": int(update), "m": m, "name": MILESTONE_LABELS[m], "steps": steps, "prev": prev, "env": int(env),
              "scope": scope, "first": prev is None}
        self.log.append(ev)
        del self.log[:-self.log_max]
        self._new.append(ev)
        return ev

    def observe(self, env: int, m: int, steps: int, update: int, from_start: bool, partial: bool = False) -> list[dict]:
        """One first arrival in an episode: milestone index ``m`` after ``steps`` steps.  Returns the new record events."""
        if partial or steps is None or not (0 <= int(m) < self.n) or int(steps) <= 0:
            return []
        steps = int(steps)
        out = []
        if from_start:
            ev = self._try(self.start, "start", int(m), steps, update, env)
            if ev:
                out.append(ev)
        ev = self._try(self.any, "any", int(m), steps, update, env)
        if ev:
            out.append(ev)
        return out

    def take_new(self) -> list[dict]:
        """Events since the previous call (they go into the history record of the update)."""
        out, self._new = self._new, []
        return out

    def snapshot(self) -> dict:
        """Compact payload for state.json."""
        return {"start": [dict(r) if r else None for r in self.start], "any": [dict(r) if r else None for r in self.any],
                "log": list(self.log), "names": list(MILESTONE_LABELS), "base_state": BASE_START_STATE}

    def load_events(self, events) -> None:
        """Rebuild the records from logged events (history.jsonl ``records`` lists) when a run resumes."""
        for ev in sorted((e for e in events if isinstance(e, dict)), key=lambda e: e.get("u", 0)):
            try:
                m, steps = int(ev["m"]), int(ev["steps"])
            except (KeyError, TypeError, ValueError):
                continue
            scope = "start" if ev.get("scope") == "start" else "any"
            if 0 <= m < self.n:
                self._try(self.start if scope == "start" else self.any, scope, m, steps, int(ev.get("u", 0)), int(ev.get("env", -1)))
        self._new = []

    @classmethod
    def from_history(cls, history_jsonl, n: int = MILESTONE_COUNT) -> "MilestoneRecords":
        """Reconstruct from an existing ``history.jsonl`` (line scan for ``records``; unreadable file = empty records)."""
        import json
        rec = cls(n)
        events: list = []
        try:
            with open(history_jsonl, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"records"' not in line:
                        continue
                    try:
                        events.extend(json.loads(line).get("records") or [])
                    except ValueError:
                        continue
        except OSError:
            pass
        rec.load_events(events)
        return rec


# --------------------------------------------------------------------------------------------- learned vs head start
SOURCES = ("base", "route", "swarm", "battle")   # reset sources, in dashboard order ("other" = any other named start state)
SOURCE_LABELS = {"base": "Base start", "route": "Route checkpoint", "swarm": "Swarm archive", "battle": "Battle archive",
                 "other": "Other start state"}
_SOURCE_PREFIX = (("swarm:", "swarm"), ("battle:", "battle"), ("route:", "route"))


def source_of(state_name) -> str:
    """Reset source of an episode from its start-state name: ``swarm:..`` / ``battle:..`` / ``route:..`` archive states,
    the plain New Bark start (``base``), or ``other`` for any other named start state."""
    s = str(state_name or "")
    for prefix, src in _SOURCE_PREFIX:
        if s.startswith(prefix):
            return src
    return "base" if is_base_start(s) else "other"


class SourceSplitStats:
    """Per reset source: how many episodes, mean path-progress GAIN and milestone reach rates (long_v7 ``source_split_metrics``).

    The mean return and the overall reach rate mix episodes that start at New Bark with episodes that start from an archived
    state closer to the goal (a head start).  Split by source, the ``base`` row is the honest learned-from-scratch measure:

    * ``gain`` = best path progress minus the progress the episode STARTED at (>= 0), mean per source;
    * ``reach[i]`` counts only milestones NOT already passed at the start (``start_stage <= i``), like ``MilestoneStats``.

    Payload keys (each a dict source -> ``{"n", "gain", "reach", "elig"}``): ``src_split`` (this update's window) and
    ``src_split_cum`` (the whole run).  Staggered partial first episodes are skipped.  No numpy.
    """

    def __init__(self, n: int = MILESTONE_COUNT):
        self.n = int(n)
        self.cum: dict[str, dict] = {}
        self._reset_window()

    def _reset_window(self) -> None:
        self.win: dict[str, dict] = {}

    def _slot(self, table: dict, src: str) -> dict:
        if src not in table:
            table[src] = {"n": 0, "gain_sum": 0.0, "gain_n": 0, "reached": [0] * self.n, "eligible": [0] * self.n}
        return table[src]

    def add_episode(self, source: str, start_stage, furthest, best_path=None, start_path=None, partial: bool = False) -> bool:
        if partial or start_stage is None or furthest is None:
            return False
        start_stage, furthest = int(start_stage), int(furthest)
        gain = None
        if best_path is not None and start_path is not None:
            gain = max(0.0, float(best_path) - float(start_path))
        for table in (self.win, self.cum):
            slot = self._slot(table, source)
            slot["n"] += 1
            if gain is not None:
                slot["gain_sum"] += gain
                slot["gain_n"] += 1
            for i in range(self.n):
                if start_stage <= i:
                    slot["eligible"][i] += 1
                    if furthest >= i + 1:
                        slot["reached"][i] += 1
        return True

    @staticmethod
    def _payload(table: dict) -> dict:
        out = {}
        for src, s in table.items():
            out[src] = {"n": s["n"], "gain": round(s["gain_sum"] / s["gain_n"], 2) if s["gain_n"] else None,
                        "reach": _rates(s["reached"], s["eligible"]), "elig": list(s["eligible"])}
        return out

    def end_update(self) -> dict:
        out = {"src_split": self._payload(self.win), "src_split_cum": self._payload(self.cum)}
        self._reset_window()
        return out
