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
    models = sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("reb_*")))

    header = ["model", "task", "variant", "base SR %", "FailPatch SR %", "patched chunks %", "buffer success/fail",
              "annotator", "collect (h)"]
    rows = []
    for model_dir in models:
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("reb_") and p.name != "features"):
            base = load(task_dir / "eval_base.json")
            meta, labels = load(task_dir / "buffer_meta.json") or {}, load(task_dir / "labels.json") or {}
            times = tsv(task_dir / "stage_times.tsv")
            first = next(iter(labels.values()), {})
            backend = first.get("annotator_model") or first.get("backend", "-")
            evals = sorted(task_dir.glob("eval_failpatch_*.json")) or [None]
            for path in evals:
                fp = load(path) if path else None
                rows.append([
                    model_dir.name, task_dir.name, path.stem.replace("eval_failpatch_", "") if path else "-",
                    pct(base), pct(fp), num(None if fp is None else 100 * fp["mean_patched_fraction"], "{:.1f}"),
                    f"{meta.get('n_success', '-')}/{meta.get('n_fail', '-')}", backend,
                    num(times.get("collect", 0) / 3600 if "collect" in times else None, "{:.2f}"),
                ])
    table(header, rows)

    sft_header = ["model", "variant", "GPU", "SFT time (s)", "feature extraction (s)", "optimisation (s)",
                  "trainable params", "GPU hours of SFT (h)", "peak GPU memory in SFT (MiB)"]
    sft_rows = []
    for model_dir in models:
        for reb_dir in sorted(model_dir.glob("reb_*")):
            train = load(reb_dir / "train_summary.json")
            if not train:
                continue
            sft = train["sft_time_s"]
            sft_rows.append([
                model_dir.name, reb_dir.name[4:], f"1x {train.get('gpu', gpu)}", num(sft),
                num(train["feature_extraction_s"]), num(train["optimization_s"]),
                f"{train['trainable_params'] / 1e6:.1f}M ({100 * train['trainable_fraction']:.2f}%)",
                num(sft / 3600, "{:.2f}"), num(train.get("peak_gpu_memory_mib")),
            ])
    print()
    table(sft_header, sft_rows)
    for row in sft_rows:
        print(f"\n{row[0]} ({row[1]}): SFT took {row[3]} s = {row[7]} h on {row[2]}")
    print("\nNote: a variant trained after another one reuses its cached demonstration features, so its "
          "'feature extraction' time is only the loading time; add the first variant's value for a from-scratch SFT time.")

    gpu_header = ["model", "GPU", "GPU hours whole pipeline (h)", "mean GPU util (%)", "peak GPU memory (MiB)"]
    gpu_rows = []
    for model_dir in models:
        wall = sum(tsv(model_dir / "model_wall_time.tsv").values()) / 3600
        util, peak = gpu_stats(model_dir / "gpu_usage.csv")
        gpu_rows.append([model_dir.name, f"1x {gpu}", f"{wall:.2f}", num(util), num(peak)])
    print()
    table(gpu_header, gpu_rows)

    with (root / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with (root / "gpu_summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows([sft_header] + sft_rows + [[]] + [gpu_header] + gpu_rows)
    print(f"\nwrote {root / 'summary.csv'} and {root / 'gpu_summary.csv'}")


if __name__ == "__main__":
    main()
