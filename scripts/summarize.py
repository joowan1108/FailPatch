"""Collect the results into tables (markdown on stdout + summary.csv + gpu_summary.csv).

Per task:  success rate of the base policy and of FailPatch on the same 50 scenes.
Per model: SFT time (s) and its parts, the GPU, GPU hours, utilisation and peak memory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(r) -> str:
    return "-" if r is None else f"{100 * r['success_rate']:.1f} ({r['successes']}/{r['num_episodes']})"


def num(v, fmt="{:.0f}") -> str:
    return "-" if v is None else fmt.format(v)


def tsv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                out[parts[0]] = out.get(parts[0], 0.0) + float(parts[1])
    return out


def gpu_stats(path: Path):
    util, mem = [], []
    if path.exists():
        for row in csv.reader(path.open()):
            try:
                util.append(float(row[1])); mem.append(float(row[2]))
            except (IndexError, ValueError):
                continue
    return (sum(util) / len(util), max(mem)) if util else (None, None)


def table(header, rows) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    gpu_file = root / "gpu.txt"
    gpu = gpu_file.read_text().strip().splitlines()[0].split(",")[0].strip() if gpu_file.exists() else "?"
    models = sorted(p for p in root.iterdir() if p.is_dir() and (p / "reb").exists())

    header = ["model", "task", "base SR %", "FailPatch SR %", "patched chunks %", "buffer success/fail",
              "annotator", "collect (h)", "base+FailPatch eval (h)"]
    rows = []
    for model_dir in models:
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != "reb"):
            base, fp = load(task_dir / "eval_base.json"), load(task_dir / "eval_failpatch.json")
            meta, labels = load(task_dir / "buffer_meta.json") or {}, load(task_dir / "labels.json") or {}
            times = tsv(task_dir / "stage_times.tsv")
            backend = next(iter(labels.values()), {}).get("backend", "-")
            rows.append([
                model_dir.name, task_dir.name, pct(base), pct(fp),
                num(None if fp is None else 100 * fp["mean_patched_fraction"], "{:.1f}"),
                f"{meta.get('n_success', '-')}/{meta.get('n_fail', '-')}", backend,
                num(times.get("collect", 0) / 3600 if "collect" in times else None, "{:.2f}"),
                num((times.get("base_eval", 0) + times.get("failpatch_eval", 0)) / 3600 if times else None, "{:.2f}"),
            ])
    table(header, rows)

    sft_header = ["model", "GPU", "SFT time (s)", "feature extraction (s)", "optimisation (s)", "trainable params",
                  "GPU hours whole pipeline (h)", "GPU hours of SFT (h)", "mean GPU util (%)", "peak GPU memory (MiB)"]
    sft_rows = []
    for model_dir in models:
        train = load(model_dir / "reb" / "train_summary.json") or {}
        wall = sum(tsv(model_dir / "model_wall_time.tsv").values()) / 3600
        util, peak = gpu_stats(model_dir / "gpu_usage.csv")
        sft = train.get("sft_time_s")
        sft_rows.append([
            model_dir.name, f"1x {train.get('gpu', gpu)}", num(sft), num(train.get("feature_extraction_s")),
            num(train.get("optimization_s")),
            "-" if not train else f"{train['trainable_params'] / 1e6:.1f}M ({100 * train['trainable_fraction']:.2f}%)",
            f"{wall:.2f}", num(None if sft is None else sft / 3600, "{:.2f}"), num(util), num(peak),
        ])
    print()
    table(sft_header, sft_rows)
    for row in sft_rows:
        if row[2] != "-":
            print(f"\n{row[0]}: SFT took {row[2]} s = {row[7]} h on {row[1]}; whole pipeline {row[6]} h on the same GPU")

    with (root / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with (root / "gpu_summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([sft_header] + sft_rows)
    print(f"\nwrote {root / 'summary.csv'} and {root / 'gpu_summary.csv'}")


if __name__ == "__main__":
    main()
