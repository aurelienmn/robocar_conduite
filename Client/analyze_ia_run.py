from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume un dossier logs/ia_runs en sortie copiable.")
    parser.add_argument("run_dir", type=Path, help="Ex: logs/ia_runs/20260629_173151_live")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--around", type=int, nargs="*", default=[], help="Frames autour desquelles afficher le detail.")
    parser.add_argument("--window", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path):
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "frame":
                events.append(event)
    return events


def category(reason: str) -> str:
    if reason.startswith("centerline"):
        return "centerline"
    if reason.startswith("target_ray"):
        return "target_ray"
    if reason.startswith("boundary_guard"):
        return reason
    return reason


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return "min={:.3f} avg={:.3f} max={:.3f}".format(min(values), mean(values), max(values))


def compact_frame(event):
    diag = event.get("diagnostics", {})
    return {
        "i": event.get("frame_idx"),
        "t": round(event.get("elapsed_s", 0.0), 2),
        "reason": event.get("reason"),
        "steer": round(event.get("steering", 0.0), 3),
        "thr": round(event.get("throttle", 0.0), 3),
        "mask": round(event.get("mask_fraction", 0.0), 4),
        "tc": round(event.get("track_center", 0.0), 3),
        "th": round(event.get("track_heading", 0.0), 3),
        "conf": round(event.get("track_confidence", 0.0), 3),
        "front": diag.get("front_min"),
        "lmin": diag.get("left_min"),
        "rmin": diag.get("right_min"),
        "target": diag.get("target_idx"),
        "guard": diag.get("guard_side"),
        "emerg": diag.get("emergency_side"),
        "pre": None if diag.get("pre_smooth_steering") is None else round(diag["pre_smooth_steering"], 3),
        "smooth": None if diag.get("smoothing") is None else round(diag["smoothing"], 3),
    }


def print_examples(title: str, examples, max_examples: int) -> None:
    print("\n{}".format(title))
    if not examples:
        print("  aucun")
        return
    for event in examples[:max_examples]:
        print("  {}".format(json.dumps(compact_frame(event), sort_keys=True)))
    if len(examples) > max_examples:
        print("  ... {} autres".format(len(examples) - max_examples))


def frames_around(events, frame_idx, window):
    return [
        event for event in events
        if abs(int(event.get("frame_idx", -10**9)) - frame_idx) <= window
    ]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    events_path = run_dir / "events.jsonl"
    summary = load_json(run_dir / "summary.json") or {}
    config = load_json(run_dir / "config.json") or {}

    if not events_path.exists():
        raise SystemExit("events.jsonl introuvable dans {}".format(run_dir))

    events = load_events(events_path)
    if not events:
        raise SystemExit("Aucune frame dans {}".format(events_path))

    reasons = Counter(event.get("reason", "") for event in events)
    categories = Counter(category(event.get("reason", "")) for event in events)
    diagnostics = [event.get("diagnostics", {}) for event in events]

    line_lost = [event for event in events if event.get("reason") == "line_lost"]
    emergencies = [event for event in events if event.get("reason") == "emergency_avoid"]
    guards = [event for event in events if event.get("reason", "").startswith("boundary_guard")]
    low_track = [
        event for event in events
        if 0.0 < event.get("track_confidence", 0.0) < 0.30
    ]
    close_front_not_emergency = [
        event for event, diag in zip(events, diagnostics)
        if diag.get("front_min") is not None
        and diag.get("front_min") < 90
        and event.get("reason") != "emergency_avoid"
    ]
    high_throttle_close = [
        event for event, diag in zip(events, diagnostics)
        if diag.get("front_min") is not None
        and diag.get("front_min") < 110
        and event.get("throttle", 0.0) > 0.045
    ]
    sign_flips = []
    for prev, cur in zip(events, events[1:]):
        prev_steer = float(prev.get("steering", 0.0))
        cur_steer = float(cur.get("steering", 0.0))
        if abs(prev_steer - cur_steer) > 0.65 and abs(prev_steer) > 0.25 and abs(cur_steer) > 0.25:
            sign_flips.append(cur)

    print("RUN {}".format(run_dir))
    print("frames={} duration_s={:.1f} stop_reason={}".format(
        len(events),
        float(summary.get("duration_s", 0.0)),
        summary.get("stop_reason", "unknown"),
    ))
    perception = config.get("perception", {})
    controller = config.get("controller", {})
    print("config: n_rays={} fov={} base_thr={} min_thr={} max_thr={}".format(
        perception.get("n_rays"),
        perception.get("fov"),
        controller.get("base_throttle"),
        controller.get("min_throttle"),
        controller.get("max_throttle"),
    ))
    print("fps: {}".format(stats([event.get("fps") for event in events])))
    print("throttle: {}".format(stats([event.get("throttle") for event in events])))
    print("steering: {}".format(stats([event.get("steering") for event in events])))
    print("abs_steering_avg={:.3f}".format(mean(abs(float(event.get("steering", 0.0))) for event in events)))
    print("mask_fraction: {}".format(stats([event.get("mask_fraction") for event in events])))
    print("track_confidence: {}".format(stats([event.get("track_confidence") for event in events])))
    print("front_min: {}".format(stats([diag.get("front_min") for diag in diagnostics])))
    print("left_min: {}".format(stats([diag.get("left_min") for diag in diagnostics])))
    print("right_min: {}".format(stats([diag.get("right_min") for diag in diagnostics])))

    print("\nReason categories:")
    for name, count in categories.most_common():
        print("  {:>22}: {:4d} ({:5.1f}%)".format(name, count, 100.0 * count / len(events)))

    print("\nTop exact reasons:")
    for name, count in reasons.most_common(12):
        print("  {:>30}: {:4d}".format(name, count))

    print_examples("Frames line_lost", line_lost, args.max_examples)
    print_examples("Frames emergency_avoid", emergencies, args.max_examples)
    print_examples("Frames boundary_guard", guards, args.max_examples)
    print_examples("Track confidence faible mais non nul", low_track, args.max_examples)
    print_examples("Front proche sans emergency", close_front_not_emergency, args.max_examples)
    print_examples("Throttle haut avec front proche", high_throttle_close, args.max_examples)
    print_examples("Gros changements de steering", sign_flips, args.max_examples)

    for frame_idx in args.around:
        print_examples(
            "Detail autour frame {}".format(frame_idx),
            frames_around(events, frame_idx, args.window),
            max_examples=args.window * 2 + 1,
        )


if __name__ == "__main__":
    main()
