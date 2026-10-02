"""Summarise a diagnostic JSONL file for bug reports.

Usage::
    python -m tools.log_summary logs/gesture_diagnostic_123.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    args = ap.parse_args()
    states=Counter(); modes=Counter(); actions=Counter(); hands=Counter(); stage_values: dict[str,list[float]]={}
    frames=0; visible=0; recovery=0; cooldown=0; bimanual=0
    p=Path(args.path)
    with p.open("r",encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            row=json.loads(line); frames += 1
            states[str(row.get("state",""))]+=1
            modes[str(row.get("cursor_mode",""))]+=1
            if row.get("has_hand"): visible+=1
            if row.get("state")=="RECOVERY" or row.get("cursor_mode")=="RECOVERY": recovery+=1
            if row.get("state")=="COOLDOWN": cooldown+=1
            hc=int(row.get("hand_count",0) or 0); hands[str(hc)]+=1
            b=row.get("bimanual") or {};
            if b.get("enabled"): bimanual += 1
            for a in row.get("actions",[]) or []: actions[str(a.get("kind",""))]+=1
            for k,v in (row.get("stage_ms") or {}).items(): stage_values.setdefault(k,[]).append(float(v))
    print(f"frames: {frames}")
    print(f"hand_visible: {visible}/{frames} ({100*visible/max(frames,1):.1f}%)")
    print(f"hand_count: {dict(hands)}")
    print(f"states: {dict(states)}")
    print(f"cursor_modes: {dict(modes)}")
    print(f"actions: {dict(actions)}")
    print(f"bimanual_frames: {bimanual}/{frames} ({100*bimanual/max(frames,1):.1f}%)")
    print(f"recovery_frames: {recovery}; cooldown_frames: {cooldown}")
    if stage_values:
        print("stages_ms_p95:")
        for k,v in sorted(stage_values.items()): print(f"  {k}: {np.percentile(v,95):.3f}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
