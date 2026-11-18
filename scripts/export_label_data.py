"""Export rollout frames and annotator labels for the human labeling tool (tools/label_tool.html).

    python scripts/export_label_data.py RESULTS_DIR OUT_DIR [--tasks ...] [--sample-fail 10 --sample-unreliable 10 --sample-reliable 5]

Writes OUT_DIR/frames/<task>/<episode>/<chunk>.jpg (stored frames, original resolution) and OUT_DIR/data.js
(`window.LABEL_DATA = {...}`, readable from file:// without a server). A fixed random subset is flagged
`sample` for a first agreement check; the annotator's answers are included but hidden by default in the tool.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

SUBTASKS = {  # from the RoboCasa task docstrings
    "CoffeeSetupMug": ["Pick the mug from the counter", "Place it under the coffee machine dispenser"],
    "GetToastedBread": ["Pull the toaster lever down and wait for the toasting to finish",
                        "Once the lever pops up, pull the bread out",
                        "Place the bread on the plate on the dining counter"],
    "SteamInMicrowave": ["Pick the vegetable from the sink and place it in the bowl",
                         "Pick the bowl and place it in the microwave",
                         "Close the microwave door", "Press the start button"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tasks", nargs="+", default=list(SUBTASKS))
    ap.add_argument("--sample-fail", type=int, default=10)
    ap.add_argument("--sample-unreliable", type=int, default=10)
    ap.add_argument("--sample-reliable", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    episodes = []
    for task in args.tasks:
        labels = json.loads((args.results / task / "labels.json").read_text())
        exec_steps = json.loads((args.results / task / "buffer_meta.json").read_text())["exec_steps"]
        stems = sorted(labels)
        fails = [s for s in stems if not labels[s]["success"]]
        unrel = [s for s in stems if labels[s]["success"] and labels[s]["reliable_success"] is False]
        rel = [s for s in stems if labels[s]["success"] and labels[s]["reliable_success"] is True]
        sample = set(rng.sample(fails, min(args.sample_fail, len(fails))) + rng.sample(unrel, min(args.sample_unreliable, len(unrel)))
                     + rng.sample(rel, min(args.sample_reliable, len(rel))))
        for stem in stems:
            lab = labels[stem]
            ep = np.load(args.results / task / "buffer" / f"{stem}.npz")
            frame_dir = args.out / "frames" / task / stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            idx = [int(i) for i in ep["thumb_idx"]]
            for img, i in zip(ep["thumbs"], idx):
                path = frame_dir / f"{i}.jpg"
                if not path.exists():
                    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            episodes.append(dict(
                id=f"{task}/{stem}", task=task, episode=stem, instruction=str(ep["instruction"]), subtasks=SUBTASKS.get(task, []),
                success=bool(lab["success"]), chunks=int(lab["chunks"]), steps=int(ep["steps"]), exec_steps=exec_steps,
                frames=idx, sample=stem in sample,
                annotator=dict(model=lab.get("annotator_model"), source=lab.get("source"),
                               failure_onset_chunk=lab.get("failure_onset_chunk"), reliable_success=lab.get("reliable_success"),
                               confidence=lab.get("confidence"), failure_type=lab.get("failure_type"), notes=lab.get("notes"))))
        print(f"{task}: {len(stems)} episodes, sample {len(sample)}", flush=True)
    (args.out / "data.js").write_text("window.LABEL_DATA = " + json.dumps(dict(episodes=episodes)) + ";\n")
    print(f"wrote {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
