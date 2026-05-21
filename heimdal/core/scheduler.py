"""Scheduler.

Enforces Work > Dream > Mirror priority and the mode feature flags
(docs/builder_pack/04_runtime and WORK_DREAM_MIRROR_MODES). Work Mode always
preempts background modes; Mirror Mode is disabled by default; Dream Mode is
feature-flagged.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

WORK = "work"
DREAM = "dream"
MIRROR = "mirror"

_PRIORITY = {WORK: 0, DREAM: 1, MIRROR: 2}


@dataclass(order=True)
class _Entry:
    priority: int
    seq: int
    mode: str
    payload: object


class Scheduler:
    def __init__(self, config):
        scheduler_cfg = config.scheduler
        self.work_preempts_background = scheduler_cfg.get("work_preempts_background", True)
        self.dream_enabled = bool(scheduler_cfg.get("dream_enabled", False))
        self.mirror_enabled = bool(scheduler_cfg.get("mirror_enabled", False))
        self._heap: list[_Entry] = []
        self._counter = itertools.count()

    def can_run(self, mode: str) -> tuple[bool, str]:
        """Whether a mode may run right now, with a human-readable reason."""
        if mode == WORK:
            return True, "Work Mode always runs."
        if mode == DREAM:
            if not self.dream_enabled:
                return False, "Dream Mode is feature-flagged off."
            if self.work_preempts_background and self._has_pending(WORK):
                return False, "Work Mode is pending and preempts Dream."
            return True, "Dream Mode may run while idle."
        if mode == MIRROR:
            if not self.mirror_enabled:
                return False, "Mirror Mode is disabled by default."
            if self.work_preempts_background and self._has_pending(WORK):
                return False, "Work Mode is pending and preempts Mirror."
            return True, "Mirror Mode may run while idle."
        return False, f"Unknown mode '{mode}'."

    def submit(self, mode: str, payload: object) -> None:
        heapq.heappush(
            self._heap,
            _Entry(_PRIORITY.get(mode, 9), next(self._counter), mode, payload),
        )

    def next_runnable(self):
        """Pop the highest-priority entry whose mode is currently allowed."""
        deferred: list[_Entry] = []
        chosen = None
        while self._heap:
            entry = heapq.heappop(self._heap)
            allowed, _reason = self.can_run(entry.mode)
            if allowed:
                chosen = entry
                break
            deferred.append(entry)
        for entry in deferred:
            heapq.heappush(self._heap, entry)
        return None if chosen is None else (chosen.mode, chosen.payload)

    def _has_pending(self, mode: str) -> bool:
        return any(entry.mode == mode for entry in self._heap)

    def mode_status(self) -> dict:
        return {
            "work": "always_enabled",
            "dream": "enabled" if self.dream_enabled else "disabled",
            "mirror": "enabled" if self.mirror_enabled else "disabled",
            "work_preempts_background": self.work_preempts_background,
        }
