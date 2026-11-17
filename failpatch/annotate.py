"""Offline annotation of the deployment buffer (FailPatch supplement S2).

For every failed rollout: the failure-onset chunk. For every successful rollout: whether it is a
reliable success. Writes <out-dir>/labels.json.

Backends (--annotator):
  openai     GPT via the OpenAI API (the paper uses GPT-5.5); needs OPENAI_API_KEY
  anthropic  Claude via the Anthropic API; needs ANTHROPIC_API_KEY. Model and reasoning effort:
             ANTHROPIC_MODEL (default claude-opus-5-5), ANTHROPIC_EFFORT (default medium)
  heuristic  no VLM: every successful rollout is reliable and the last `--heuristic-fraction`
             of a failed rollout is failure-associated
  auto       openai if OPENAI_API_KEY is set, else anthropic if ANTHROPIC_API_KEY is set,
             else heuristic
The prompts are the ones printed in the paper; only the sentence describing the contact sheet is
adapted, because our rollouts are longer than 24 chunks: frames are stored at native resolution
for up to 96 evenly spaced chunks and sent as one or two 8x6 sheets of 256 px tiles (<= 2048x1536,
inside the 2576 px limit, so Claude sees them without downscaling).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SYSTEM_PROMPT = """You are an expert robotic manipulation analyst. Your job is to analyze temporally ordered RGB observations from a robot episode.
For a failed rollout, identify the earliest action chunk at which a visible and actionable mistake or wrong commitment begins, marking the failure onset before the episode becomes irreversible.
For a successful rollout, determine whether it is a reliable success. A rollout is reliable only if it contains no visually apparent substantive intermediate deviation, such as a failed grasp, collision, incorrect placement, wrong object selection, prolonged stall, or retry.
Respond only with valid JSON matching the requested schema. If uncertain, return the best estimate and set confidence accordingly."""

USER_PROMPT = """Task instruction: {task_instruction}
Rollout outcome: {rollout_outcome}
You are given {sheets} of RGB frames from a single deployment rollout. Each tile is labeled with its chunk index t in execution order. The rollout has {num_chunks} action chunks (indices 0..{last_chunk}); the tiles show {coverage} (left agent-view camera).
Definitions:
failure_onset_chunk: for a failed rollout, the earliest chunk where a visible mistake or wrong commitment appears that will likely cause final task failure if not corrected.
reliable_success: for a successful rollout, whether the trajectory contains no visually apparent substantive intermediate deviation that would make it unreliable as a success prototype (true = reliable, false = unreliable).
Rules:
If the rollout outcome is failure, mark the earliest visible wrong action or commitment, not the moment when failure is already irreversible.
If the rollout outcome is success, set failure_onset_chunk and failure_type to null, and determine reliable_success based on the complete trajectory.
A successful rollout with a failed grasp, visible collision, incorrect placement, wrong object selection, prolonged stall, or retry should be marked as reliable_success=false, even if the task is eventually completed.
Short-lived motion jitter that does not disrupt task progress is not sufficient to mark a successful rollout as unreliable.
If multiple errors occur in a failed rollout, return the first visible error.
If RGB observations are insufficient, return null for the corresponding uncertain field and explain the reason in notes.
Chunk indices are 0-based and align with policy action chunks ({exec_steps} environment steps per chunk). Report failure_onset_chunk as a chunk index t (it may lie between two shown tiles).
Output JSON schema:
{{
"failure_onset_chunk": <int or null>,
"confidence": <float 0-1>,
"failure_type": <string or null>,
"notes": <string>,
"reliable_success": <bool or null>
}}
For a failed rollout, set reliable_success to null.
For a successful rollout, set failure_onset_chunk and failure_type to null.
Return JSON only."""


def contact_sheets(thumbs: np.ndarray, idx: np.ndarray, max_tiles: int = 96, per_sheet: int = 48,
                   cols: int = 8, tile: int = 256) -> list[bytes]:
    """All stored frames (up to `max_tiles`, evenly spaced) on as few sheets of `per_sheet` tiles
    as possible, split evenly. Frames are only resized when they are not `tile` pixels already."""
    keep = np.unique(np.linspace(0, len(idx) - 1, min(max_tiles, len(idx))).round().astype(int))
    sheets = []
    for part in np.array_split(keep, math.ceil(len(keep) / per_sheet)):
        rows = math.ceil(len(part) / cols)
        sheet = np.zeros((rows * tile, min(cols, len(part)) * tile, 3), np.uint8)
        for n, j in enumerate(part):
            img = thumbs[j] if thumbs[j].shape[:2] == (tile, tile) else \
                cv2.resize(thumbs[j], (tile, tile), interpolation=cv2.INTER_CUBIC)
            img = img[:, :, ::-1].copy()  # RGB -> BGR
            cv2.rectangle(img, (0, 0), (92, 26), (0, 0, 0), -1)
            cv2.putText(img, f"t={int(idx[j])}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            r, c = divmod(n, cols)
            sheet[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = img
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sheets.append(buf.tobytes())
    return sheets


def describe_sheets(n_sheets: int, n_tiles: int, num_chunks: int) -> tuple[str, str]:
    sheets = ("a temporally ordered contact sheet" if n_sheets == 1 else
              f"{n_sheets} temporally ordered contact sheets (each continues the previous one)")
    coverage = ("every chunk" if n_tiles >= num_chunks else
                f"an evenly spaced subset of {n_tiles} of them (about one every {num_chunks / n_tiles:.1f} chunks)")
    return sheets, coverage


def _post(url: str, headers: dict, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except Exception as error:  # rate limits / transient errors
            if attempt == 4:
                raise
            print(f"[annotate] request failed ({error}); retrying", flush=True)
            time.sleep(5 * (attempt + 1))


def ask_openai(images: list[bytes], prompt: str, model: str) -> str:
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
               for im in images]
    out = _post("https://api.openai.com/v1/chat/completions",
                {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                {"model": model, "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content + [{"type": "text", "text": prompt}]}]})
    return out["choices"][0]["message"]["content"]


def ask_anthropic(images: list[bytes], prompt: str, model: str, effort: str) -> str:
    """Claude Messages API. Raw HTTPS instead of the `anthropic` SDK because the simulator
    environment of the Docker image does not ship the SDK and this is its only API call.
    Claude Opus 5.5 always thinks (adaptive); depth is set with output_config.effort, and
    max_tokens has to leave room for the thinking plus the short JSON answer."""
    out = _post("https://api.anthropic.com/v1/messages",
                {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": 16000, "system": SYSTEM_PROMPT,
                 "output_config": {"effort": effort},
                 "messages": [{"role": "user", "content": [
                     {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "data": base64.b64encode(im).decode()}} for im in images
                 ] + [{"type": "text", "text": prompt}]}]})
    if out.get("stop_reason") == "refusal":
        raise RuntimeError(f"request declined ({(out.get('stop_details') or {}).get('category')})")
    return "".join(block["text"] for block in out["content"] if block.get("type") == "text")


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="task directory containing buffer/")
    parser.add_argument("--annotator", default="auto", choices=("auto", "openai", "anthropic", "heuristic"))
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--anthropic-model", default=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"))
    parser.add_argument("--anthropic-effort", default=os.environ.get("ANTHROPIC_EFFORT", "medium"),
                        choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--heuristic-fraction", type=float, default=0.5)
    args = parser.parse_args()

    backend = args.annotator
    if backend == "auto":
        backend = ("openai" if os.environ.get("OPENAI_API_KEY") else
                   "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic")
    out = Path(args.out_dir)
    meta = json.loads((out / "buffer_meta.json").read_text())
    labels_path = out / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    labels = {k: v for k, v in labels.items() if v.get("backend") == backend}  # resume same backend only
    tic = time.time()
    for path in sorted((out / "buffer").glob("ep_*.npz")):
        if path.stem in labels:
            continue
        ep = np.load(path, allow_pickle=False)
        success, chunks = bool(ep["success"]), int(ep["hidden"].shape[0])
        fallback_onset = int(math.floor(chunks * (1 - args.heuristic_fraction)))
        model_name = {"openai": args.openai_model, "anthropic": f"{args.anthropic_model} (effort {args.anthropic_effort})"}.get(backend)
        entry = dict(success=success, chunks=chunks, backend=backend, source=backend, annotator_model=model_name,
                     frames_shown=min(96, len(ep["thumb_idx"])), frame_px=int(ep["thumbs"].shape[1]),
                     failure_onset_chunk=None if success else fallback_onset,
                     reliable_success=True if success else None)
        if backend != "heuristic":
            sheet = contact_sheets(ep["thumbs"], ep["thumb_idx"])
            n_tiles = min(96, len(ep["thumb_idx"]))
            sheets_text, coverage = describe_sheets(len(sheet), n_tiles, chunks)
            prompt = USER_PROMPT.format(task_instruction=str(ep["instruction"]),
                                        rollout_outcome="success" if success else "failure",
                                        sheets=sheets_text, coverage=coverage,
                                        num_chunks=chunks, last_chunk=chunks - 1, exec_steps=meta["exec_steps"])
            try:
                text = (ask_openai(sheet, prompt, args.openai_model) if backend == "openai"
                        else ask_anthropic(sheet, prompt, args.anthropic_model, args.anthropic_effort))
                answer = parse_json(text)
                entry.update(confidence=answer.get("confidence"), failure_type=answer.get("failure_type"),
                             notes=answer.get("notes"))
                if success and isinstance(answer.get("reliable_success"), bool):
                    entry["reliable_success"] = answer["reliable_success"]
                elif not success and isinstance(answer.get("failure_onset_chunk"), int):
                    entry["failure_onset_chunk"] = min(max(answer["failure_onset_chunk"], 0), chunks - 1)
                else:
                    entry["source"] = "heuristic-fallback (VLM returned null)"
            except Exception as error:
                entry["source"] = f"heuristic-fallback ({type(error).__name__}: {error})"[:200]
        labels[path.stem] = entry
        labels_path.write_text(json.dumps(labels, indent=1))
    fails = [v for v in labels.values() if not v["success"]]
    wins = [v for v in labels.values() if v["success"]]
    fallbacks = sum(v["source"].startswith("heuristic-fallback") for v in labels.values())
    print(f"[annotate] backend={backend}: {len(wins)} successes ({sum(v['reliable_success'] for v in wins)} reliable), "
          f"{len(fails)} failures (mean onset at {np.mean([v['failure_onset_chunk'] / v['chunks'] for v in fails]) if fails else float('nan'):.2f} "
          f"of the episode), {fallbacks} heuristic fallbacks, {time.time() - tic:.0f}s")


if __name__ == "__main__":
    main()
