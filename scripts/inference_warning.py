#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_warning.py

runtime inference for ADAS warnings, with warning types.

Outputs BOTH:
1) results (per-type): results[TYPE] = {triggered, warning_direction, reasoning, safety_reason}
2) warnings[] aggregated from triggered results (capped by --max-warnings)

Debug:
- Writes JSONL lines including raw_model_output + parsed_model_json.
- Optionally dumps per-window raw and parsed files to --debug-dir (default: <out>.debug)
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import transformers

from tqdm import tqdm

DEFAULT_SAMPLE_FPS = 5
DEFAULT_WINDOW_SEC = 2.0
DEFAULT_STEP_SEC = 0.5
DEFAULT_MAX_SIDE = 768
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"

_MARKER_BEGIN_RE = re.compile(r"BEGIN_WARNING_PROMPT_([A-Za-z0-9_]+)\b")
_MARKER_END_RE = re.compile(r"END_WARNING_PROMPT_([A-Za-z0-9_]+)\b")


def parse_warning_types_from_prompt_text(prompt_text: str) -> List[str]:
    begins = _MARKER_BEGIN_RE.findall(prompt_text)
    ends = _MARKER_END_RE.findall(prompt_text)

    seen = set()
    ordered: List[str] = []
    for t in begins:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    ends_set = set(ends)
    if ends_set:
        return [t for t in ordered if t in ends_set] or ordered
    return ordered


def load_warning_prompt(warning_prompt_path: Path) -> Tuple[str, List[str]]:
    txt = warning_prompt_path.read_text(encoding="utf-8")
    types = parse_warning_types_from_prompt_text(txt)
    if not types:
        raise RuntimeError(
            f"No warning types found in {warning_prompt_path}. "
            "Expected markers like BEGIN_WARNING_PROMPT_<TYPE>."
        )
    return txt, types


def setting_prompt_template(
    allowed_types: List[str],
    max_warnings: int,
    sample_fps: int,
    window_sec: float,
) -> str:
    types_str = ", ".join(allowed_types)
    return (
        "Runtime Setting (ADAS Warning Inference):\n"
        "\n"
        "Camera setup:\n"
        "- Single forward-facing camera mounted inside the windshield.\n"
        "- Camera optical axis aligned with the vehicle longitudinal centerline.\n"
        "\n"
        "Temporal constraint:\n"
        f"- Frames are sampled at {sample_fps} frames per second.\n"
        f"- Each inference window covers the past {window_sec:.1f} seconds.\n"
        "- Frames are strictly ordered from oldest to newest.\n"
        "- Do NOT assume any future information beyond the last frame.\n"
        "\n"
        "Spatial focus (near-field ROI):\n"
        "- Focus your attention and make warning judgments within 5 meters ahead in ego path.\n"
        "- Absolute depth is not directly available; approximate 5m using visual cues:\n"
        "  * lower image region near road surface / ego-lane corridor,\n"
        "  * large apparent object scale / rapid scale change,\n"
        "  * strong proximity cues (occupying significant pixels in near-field area).\n"
        "- De-prioritize far-field objects near the horizon unless they show strong imminent hazard cues.\n"
        "\n"
        
        "Ego-motion context (visual-only inference):\n"
        "- You MAY infer the qualitative ego vehicle motion state from consecutive frames to improve overall judgment.\n"
        "- Examples of ego-motion cues include:\n"
        "  * global optical flow patterns,\n"
        "  * consistent forward scene expansion,\n"
        "  * lateral scene drift,\n"
        "  * lane boundary movement relative to image center,\n"
        "  * horizon or perspective shift.\n"
        "- Use ego-motion inference only as supporting context to interpret object or lane behavior.\n"
        "- Do NOT assume any external sensors (speed, IMU, steering angle) or numerical velocities.\n"
        "- Ego-motion must be inferred strictly from observable visual evidence within the provided frames.\n"
        "\n"

        "Decision task (per-type independent evaluation):\n"
        "- Evaluate EACH allowed warning type independently (do not let one type suppress another).\n"
        "- Decide triggered vs not triggered using ONLY visible evidence in the provided frames.\n"
        "- Do NOT speculate beyond observable visual cues.\n"
        "\n"
        "Allowed warning types:\n"
        f"- {types_str}\n"
        f"- Maximum simultaneous triggered warnings (aggregated): {max_warnings}\n"
        "\n"
        "Output requirements (STRICT):\n"
        "- Output EXACTLY one JSON object.\n"
        "- No text outside JSON.\n"
        "- Do NOT output additional keys beyond the schema below.\n"
        "- ALL string fields MUST be ENGLISH ONLY; do NOT output any non-English characters.\n"
        "- Keep all string fields SHORT (1 sentence, <= 25 words).\n"
        "- JSON schema:\n"
        "  {\n"
        "    \"ts_ms\": integer,\n"
        "    \"time_window_ms\": [integer, integer],\n"
        "    \"results\": {\n"
        "      \"<WARNING_TYPE>\": {\n"
        "        \"triggered\": boolean,\n"
        "        \"warning_direction\": \"LEFT\" | \"RIGHT\" | \"CENTER\" | \"UNKNOWN\",\n"
        "        \"reasoning\": string,\n"
        "        \"safety_reason\": string\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "\n"
        "Rules:\n"
        "- \"results\" MUST contain exactly one entry for each allowed warning type, using the type name as the key.\n"
        "- If a type is NOT triggered: set triggered=false, warning_direction=\"UNKNOWN\", reasoning=\"\", and a NON-EMPTY safety_reason explaining why it is safe.\n"
        "- If a type IS triggered: set triggered=true, provide a NON-EMPTY reasoning; safety_reason MUST be empty (\"\").\n"
        "- Do NOT invent new warning types.\n"
        "- Do NOT use ego artifacts (hood, reflections, windshield glare) as evidence.\n"
        "- reasoning/safety_reason should reference observable motion/position/temporal consistency and near-field ROI cues when relevant.\n"
    )


def sample_frames_from_video_rgb(video_path: str, sample_fps: int, max_side: int) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(orig_fps / max(sample_fps, 1))))

    frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    ok, bgr = cap.read()
    while ok:
        if (idx % step) == 0:
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)

            h, w = bgr.shape[:2]
            if min(w, h) > max_side:
                if w < h:
                    new_w = max_side
                    new_h = int(h * (max_side / w))
                else:
                    new_h = max_side
                    new_w = int(w * (max_side / h))
                bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((ts_ms, rgb))
        idx += 1
        ok, bgr = cap.read()

    cap.release()
    return frames


def iter_windows_two_pointer(frames: List[Tuple[int, np.ndarray]], window_ms: int, step_ms: int):
    if not frames:
        return

    ts_list = [t for t, _ in frames]
    start_ts = ts_list[0]
    end_ts = ts_list[-1]

    t0 = start_ts + window_ms
    l = 0
    r = 0
    n = len(frames)

    while t0 <= end_ts:
        while r < n and frames[r][0] <= t0:
            r += 1
        win_start = t0 - window_ms
        while l < r and frames[l][0] <= win_start:
            l += 1
        if l < r:
            yield t0, frames[l:r]
        t0 += step_ms


def assemble_model_input_text(
    setting_prompt: str,
    warning_prompt_text: str,
    window_frames: List[Tuple[int, np.ndarray]],
) -> str:
    lines: List[str] = []
    lines.append(setting_prompt.strip())

    lines.append("\n---\nCompiled Warning Rules (Domain Logic):\n")
    lines.append(warning_prompt_text.strip())

    lines.append("\n---\nTemporal Context (oldest -> newest):\n")
    for i, (ts, _rgb) in enumerate(window_frames):
        lines.append(f"Frame {i}: timestamp = {ts} ms")

    if window_frames:
        t_last = window_frames[-1][0]
        lines.append(f"Decision timestamp ts_ms MUST equal last frame timestamp: {t_last} ms")

    lines.append("\n---\nTask:\n")
    lines.append(
        "For EACH allowed warning type, decide triggered vs not triggered at the decision timestamp, "
        "using ONLY the compiled warning rules and visual evidence in the past-only frames. "
        "Return exactly one JSON object following the schema."
    )
    return "\n".join(lines)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t).strip()

    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return {}
    cand = m.group(0).strip()
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return {}


def coerce_to_v4_schema(obj: Dict[str, Any], allowed_types: List[str]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"results": {}}
    res = obj.get("results")
    if not isinstance(res, dict):
        return {"results": {}}

    allowed_set = set(allowed_types)
    filtered: Dict[str, Any] = {}
    for k, v in res.items():
        if k in allowed_set and isinstance(v, dict):
            filtered[k] = v
    return {"results": filtered}


def _normalize_direction(d: Any) -> str:
    s = str(d or "").strip().upper()
    if s in {"LEFT", "RIGHT", "CENTER", "UNKNOWN"}:
        return s
    s2 = str(d or "").strip().lower()
    if s2 == "left":
        return "LEFT"
    if s2 == "right":
        return "RIGHT"
    if s2 in {"front", "center", "straight"}:
        return "CENTER"
    return "UNKNOWN"


def validate_and_normalize_model_output(
    coerced: Dict[str, Any],
    parsed_model_json: Dict[str, Any],
    raw_model_output: str,
    *,
    t0_ms: int,
    window_ms: int,
    allowed_types: List[str],
    max_warnings: int,
) -> Dict[str, Any]:
    results_out: Dict[str, Dict[str, Any]] = {}
    res_in = coerced.get("results") if isinstance(coerced, dict) else {}
    if not isinstance(res_in, dict):
        res_in = {}

    for wt in allowed_types:
        entry = res_in.get(wt, {})
        triggered = bool(entry.get("triggered")) if isinstance(entry, dict) else False
        direction = _normalize_direction(entry.get("warning_direction")) if isinstance(entry, dict) else "UNKNOWN"
        reasoning = str(entry.get("reasoning", "")).strip() if isinstance(entry, dict) else ""
        safety_reason = str(entry.get("safety_reason", "")).strip() if isinstance(entry, dict) else ""

        if not triggered:
            if not safety_reason:
                safety_reason = (
                    "No sufficient visual evidence within the past-only window (focused on near-field ego path ~5m) "
                    "to trigger this warning type."
                )
            direction = "UNKNOWN"
            reasoning = ""
        else:
            if not reasoning:
                reasoning = "Triggered based on compiled rules and consistent visual evidence across frames."

        results_out[wt] = {
            "triggered": bool(triggered),
            "warning_direction": direction,
            "reasoning": reasoning,
            "safety_reason": safety_reason,
        }

    warnings_agg: List[Dict[str, Any]] = []
    for wt in allowed_types:
        r = results_out[wt]
        if r["triggered"]:
            warnings_agg.append(
                {
                    "warning_type": wt,
                    "warning_direction": r["warning_direction"],
                    "reasoning": r["reasoning"],
                }
            )
            if len(warnings_agg) >= max_warnings:
                break

    return {
        "ts_ms": int(t0_ms),
        "time_window_ms": [int(t0_ms - window_ms), int(t0_ms)],
        "results": results_out,
        "warnings": warnings_agg,
        "raw_model_output": raw_model_output,
        "parsed_model_json": parsed_model_json,
    }


def save_window_debug(debug_dir: Path, t0_ms: int, raw_text: str, parsed_json: Dict[str, Any]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = f"t0_{t0_ms:010d}"
    (debug_dir / f"{base}.raw.txt").write_text(raw_text or "", encoding="utf-8")
    (debug_dir / f"{base}.parsed.json").write_text(
        json.dumps(parsed_json or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass
class ModelBundle:
    model: transformers.PreTrainedModel
    processor: Any


def load_model(
    model_name: str,
    *,
    dtype: str = "float16",
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
) -> ModelBundle:
    transformers.set_seed(0)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype.lower(), torch.float16)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    return ModelBundle(model=model, processor=processor)


def run_model_on_window(
    bundle: ModelBundle,
    prompt_text: str,
    window_frames_rgb: List[np.ndarray],
    *,
    fps: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    conv = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant for autonomous driving safety warning inference."}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "video", "video": window_frames_rgb},
        ]},
    ]

    inputs = bundle.processor.apply_chat_template(
        conv,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    ).to(bundle.model.device)

    gen = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    gen_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen, strict=False)]
    return bundle.processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main():
    p = argparse.ArgumentParser(description="Inference Warning v4 (local, per-type + aggregated warnings) with debug dumps")
    p.add_argument("--front", required=True, help="Path to front-view mp4")
    p.add_argument("--warning-prompt", required=True, help="Path to compiled warning prompt (text) with markers")
    p.add_argument("--out", default="results_v4_dual.jsonl", help="Output jsonl file path")

    p.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="Past window seconds")
    p.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC, help="Sliding step seconds")
    p.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS, help="Frame sampling fps")
    p.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Resize short edge to this value")
    p.add_argument("--max-warnings", type=int, default=2, help="Max simultaneous warnings in aggregated list")

    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name/path")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype")
    p.add_argument("--device-map", default="auto", help="Transformers device_map")
    p.add_argument("--attn-impl", default="sdpa", help="Attention impl")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Generation length cap")

    p.add_argument("--debug-dir", default="", help="Directory to dump per-window raw/parsed outputs (default: <out>.debug)")
    p.add_argument("--no-dump-windows", action="store_true", help="Disable per-window raw/parsed dump files")
    p.add_argument("--dry-run", action="store_true", help="Do not run model; write SAFE outputs")
    args = p.parse_args()

    video_path = Path(args.front)
    warning_prompt_path = Path(args.warning_prompt)
    out_path = Path(args.out)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not warning_prompt_path.exists():
        raise SystemExit(f"Warning prompt not found: {warning_prompt_path}")

    warning_prompt_text, allowed_types = load_warning_prompt(warning_prompt_path)
    print(f"Detected warning types from warning_prompt: {allowed_types}")

    setting_prompt = setting_prompt_template(
        allowed_types=allowed_types,
        max_warnings=args.max_warnings,
        sample_fps=args.sample_fps,
        window_sec=args.window_sec,
    )

    print("Sampling frames from video...")
    frames = sample_frames_from_video_rgb(str(video_path), args.sample_fps, args.max_side)
    if not frames:
        raise SystemExit("No frames sampled; abort.")

    window_ms = int(args.window_sec * 1000)
    step_ms = int(args.step_sec * 1000)

    bundle: Optional[ModelBundle] = None
    if not args.dry_run:
        print(f"Loading model: {args.model}")
        bundle = load_model(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_impl,
        )

    debug_dir = Path(args.debug_dir) if args.debug_dir else Path(str(out_path) + ".debug")
    if args.no_dump_windows:
        print("Per-window debug dump: DISABLED")
    else:
        print(f"Per-window debug dump dir: {debug_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        print("Running inference windows...")
        window_count = len(list(iter_windows_two_pointer(frames, window_ms, step_ms)))
        pbar = tqdm(total=window_count)
        for (t0, win) in iter_windows_two_pointer(frames, window_ms, step_ms):
            prompt_text = assemble_model_input_text(setting_prompt, warning_prompt_text, win)
            if args.dry_run:
                raw_text = ""
                parsed_obj: Dict[str, Any] = {}
                coerced = {"results": {}}
            else:
                assert bundle is not None
                win_rgb = [rgb for (_ts, rgb) in win]
                raw_text = run_model_on_window(
                    bundle,
                    prompt_text,
                    win_rgb,
                    fps=max(1, args.sample_fps),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                parsed_obj = extract_first_json_object(raw_text)
                coerced = coerce_to_v4_schema(parsed_obj, allowed_types)

                if not args.no_dump_windows:
                    save_window_debug(debug_dir, t0, raw_text, parsed_obj)

            normalized = validate_and_normalize_model_output(
                coerced=coerced,
                parsed_model_json=parsed_obj,
                raw_model_output=raw_text,
                t0_ms=t0,
                window_ms=window_ms,
                allowed_types=allowed_types,
                max_warnings=args.max_warnings,
            )

            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            f.flush()

            pbar.update(1)

    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
