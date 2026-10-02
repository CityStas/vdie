"""Gesture Grammar — gestures as a small interaction language.

A gesture vocabulary of N poses gives you N commands if you treat each pose as a
button.  Add an explicit *pause token* and the same vocabulary gives you
sequences: ``INDEX_UP · pause · SWIPE_LEFT`` is a different command from
``SWIPE_LEFT`` alone.  The pause is not a gesture — it is punctuation, and it is
free, because a human hand naturally comes to rest between deliberate actions.

This module implements exactly that: a token stream with pause punctuation and a
pattern matcher over the tail of the stream.  It is the cheapest way to multiply
the expressive power of an existing vocabulary without inventing new poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import EngineConfig

PAUSE_TOKEN = "pause"


@dataclass
class GrammarMatch:
    name: str
    pattern: list[str]
    timestamp: float


@dataclass
class GestureGrammar:
    cfg: EngineConfig
    _tokens: list[tuple[str, float]] = field(default_factory=list)
    _last_gesture: str = "NONE"
    _last_change: float = 0.0
    _last_match: float = -1e9
    _history: list[GrammarMatch] = field(default_factory=list)

    def reset(self) -> None:
        self._tokens.clear()
        self._last_gesture = "NONE"
        self._last_change = 0.0
        self._last_match = -1e9
        self._history.clear()

    # ------------------------------------------------------------------ #
    @property
    def tokens(self) -> list[str]:
        return [t for t, _ in self._tokens]

    def matches(self) -> list[GrammarMatch]:
        return list(self._history)

    # ------------------------------------------------------------------ #
    def push(self, gesture: str, timestamp: float) -> list[GrammarMatch]:
        """Feed the confirmed gesture for this frame; return newly matched rules."""
        g = self.cfg.grammar
        if not g.enabled or gesture in ("NONE", ""):
            return []

        if gesture == self._last_gesture:
            return []

        gap = timestamp - self._last_change
        if self._last_gesture != "NONE" and gap * 1000.0 >= g.separator_ms:
            self._tokens.append((PAUSE_TOKEN, timestamp))
        self._tokens.append((gesture, timestamp))
        self._last_gesture = gesture
        self._last_change = timestamp

        # Prune tokens outside the sequence window.
        cutoff = timestamp - g.sequence_timeout_ms / 1000.0
        self._tokens = [t for t in self._tokens if t[1] >= cutoff]

        return self._match(timestamp)

    # ------------------------------------------------------------------ #
    def _match(self, timestamp: float) -> list[GrammarMatch]:
        g = self.cfg.grammar
        names = [t for t, _ in self._tokens]
        found: list[GrammarMatch] = []
        if (timestamp - self._last_match) * 1000.0 < g.sequence_timeout_ms * 0.5:
            return found

        for name, pattern in g.rules.items():
            if len(pattern) > len(names):
                continue
            if names[-len(pattern) :] == list(pattern):
                match = GrammarMatch(name=name, pattern=list(pattern), timestamp=timestamp)
                found.append(match)
                self._history.append(match)
                self._tokens.clear()
                self._last_gesture = "NONE"
                self._last_match = timestamp
                break
        return found

    # ------------------------------------------------------------------ #
    def partial(self) -> str:
        """Longest prefix of any rule currently matched — for the overlay."""
        names = [t for t, _ in self._tokens]
        best = ""
        for pattern in self.cfg.grammar.rules.values():
            for n in range(len(pattern), 0, -1):
                if len(names) >= n and names[-n:] == list(pattern)[:n]:
                    candidate = " · ".join(pattern[:n])
                    if len(candidate) > len(best):
                        best = candidate
                    break
        return best
