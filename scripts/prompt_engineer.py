#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prompt_engineer.py

Dynamic ADAS warning rule compiler with strict structural constraints
to avoid over-conservative AND-style rules.

Key improvements:
- Hard gates limited to 1~2 prerequisite-only checks
- 3~5 OR trigger patterns required
- Each pattern: exactly 1 primary cue + optional 1 secondary cue
- Must include at least one low-threshold early warning pattern
- No class-specific hardcoding
- Post-compile structural validation
"""

import argparse
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 1600


def utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_warning_types_from_spec(spec: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    if isinstance(spec.get("warning_types"), dict):
        keys = list(spec["warning_types"].keys())
    elif isinstance(spec.get("warning_classes"), dict):
        keys = list(spec["warning_classes"].keys())

    out = []
    for k in keys:
        if not isinstance(k, str):
            continue
        kk = k.strip()
        if not kk:
            continue
        if kk.upper() in {"SAFE", "NONE", "NO_WARNING", "NOALERT"}:
            continue
        out.append(kk)
    return out


def build_compiler_prompt(spec: Dict[str, Any], warning_types: List[str]) -> str:
    spec_json = json.dumps(spec, ensure_ascii=False, indent=2)

    marker_lines = []
    for wt in warning_types:
        marker_lines.append(f"BEGIN_WARNING_PROMPT_{wt}")
        marker_lines.append(f"END_WARNING_PROMPT_{wt}")
    marker_str = "\n".join(f"- {m}" for m in marker_lines)

    section_template = (
        "Use EXACTLY this section structure inside each marker block.\n"
        "Section headers must match exactly (case-sensitive):\n\n"

        "1) [Hard gates]\n"
        "   - List ONLY 1~2 prerequisite checks.\n"
        "   - Hard gates must be about observability or validity conditions ONLY.\n"
        "   - Do NOT place hazard evidence (closing, crossing, cut-in, drift) in hard gates.\n"
        "   - If any hard gate fails, suppress this warning.\n\n"

        "2) [Trigger patterns (OR)]\n"
        "   - Provide 3~5 trigger patterns.\n"
        "   - Any ONE pattern is sufficient to trigger.\n"
        "   - Each pattern must contain exactly ONE primary trigger cue.\n"
        "   - Each pattern may include at most ONE secondary corroboration cue.\n"
        "   - Primary cue must describe the core hazard signal.\n"
        "   - Secondary cue may describe region constraint or persistence.\n"
        "   - At least ONE pattern must be a low-threshold early-warning pattern.\n"
        "   - You may use OR inside a bullet (e.g., A OR B) to avoid over-constraining.\n\n"

        "3) [Evidence checks]\n"
        "   - Supporting cues that increase confidence.\n"
        "   - Do NOT duplicate trigger pattern bullets.\n\n"

        "4) [Temporal consistency]\n"
        "   - Define persistence across frames.\n"
        "   - Allow short-circuit only for extreme hazards.\n\n"

        "5) [Exclusions]\n"
        "   - Specific, testable suppression conditions.\n"
        "   - Must not eliminate all realistic triggers.\n\n"

        "6) [Tie-breakers]\n"
        "   - Practical rules when evidence is mixed.\n"
        "   - Avoid writing rules that almost never trigger.\n"
    )

    global_bias = (
        "Global behavioral requirements:\n"
        "- Do NOT default to SAFE due to mild ambiguity.\n"
        "- If visual evidence is suggestive and consistent across frames,\n"
        "  prefer defining a trigger pattern rather than suppressing entirely.\n"
        "- Avoid AND-ing multiple independent hazard cues in the same pattern.\n"
        "- Widen trigger patterns instead of moving hazard cues into hard gates.\n"
    )

    return (
        "You are an ADAS warning rule designer.\n"
        "Compile the given spec into operational decision rules.\n\n"
        "Output ONLY marker-wrapped rule blocks.\n"
        "Required markers (must match exactly):\n"
        f"{marker_str}\n\n"
        "Hard constraints:\n"
        "- Do NOT include camera mounting info or runtime I/O schema.\n"
        "- Prefer observable cues over abstract wording.\n\n"
        f"{global_bias}\n"
        f"{section_template}\n"
        "Spec (JSON):\n"
        f"{spec_json}\n\n"
        "Now output ONLY the compiled warning prompt text."
    )


@dataclass
class ModelBundle:
    model: transformers.PreTrainedModel
    processor: Any


def load_model(model_name: str, dtype: str, device_map: str, attn_impl: str) -> ModelBundle:
    transformers.set_seed(0)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype.lower(), torch.bfloat16)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    return ModelBundle(model=model, processor=processor)


def generate_text(bundle: ModelBundle, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    conv = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an ADAS warning rule designer."}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = bundle.processor.apply_chat_template(
        conv,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.model.device)

    gen = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    gen_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
    text = bundle.processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text


def extract_block(text: str, begin: str, end: str) -> Optional[str]:
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), flags=re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return (begin + m.group(1) + end).strip()


def validate_structure(compiled: str) -> List[str]:
    issues = []
    sections_required = [
        "[Hard gates]",
        "[Trigger patterns (OR)]",
        "[Evidence checks]",
        "[Temporal consistency]",
        "[Exclusions]",
        "[Tie-breakers]",
    ]

    for sec in sections_required:
        if sec not in compiled:
            issues.append(f"Missing section: {sec}")

    # crude checks
    if compiled.count("[Hard gates]") > 0:
        hard_section = re.search(r"\[Hard gates\](.*?)(\n\[|\Z)", compiled, re.DOTALL)
        if hard_section:
            bullets = re.findall(r"- ", hard_section.group(1))
            if len(bullets) > 2:
                issues.append("Hard gates > 2 bullets (too strict)")

    if compiled.count("[Trigger patterns (OR)]") > 0:
        trig_section = re.search(r"\[Trigger patterns \(OR\)\](.*?)(\n\[|\Z)", compiled, re.DOTALL)
        if trig_section:
            patterns = re.findall(r"- ", trig_section.group(1))
            if len(patterns) < 3:
                issues.append("Less than 3 trigger patterns")

    return issues


def normalize_compiled_output(raw_text: str, warning_types: List[str]) -> Tuple[str, List[str]]:
    blocks = []
    missing = []

    for wt in warning_types:
        b = f"BEGIN_WARNING_PROMPT_{wt}"
        e = f"END_WARNING_PROMPT_{wt}"
        blk = extract_block(raw_text, b, e)
        if blk is None:
            missing.extend([b, e])
        else:
            blocks.append(blk)

    if blocks:
        return "\n\n".join(blocks).strip(), sorted(set(missing))
    return raw_text.strip(), sorted(set(missing))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default=DEFAULT_DTYPE)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    warning_types = get_warning_types_from_spec(spec)

    compiler_prompt = build_compiler_prompt(spec, warning_types)

    print(f"[{utc_iso()}] Loading model...")
    bundle = load_model(args.model, args.dtype, args.device_map, args.attn_impl)

    raw = generate_text(
        bundle,
        compiler_prompt,
        args.max_new_tokens,
        args.do_sample,
        args.temperature,
    )

    compiled, missing = normalize_compiled_output(raw, warning_types)

    issues = validate_structure(compiled)
    if issues:
        print("STRUCTURE WARNINGS:")
        for i in issues:
            print(" -", i)

    Path(args.out).write_text(compiled + "\n", encoding="utf-8")
    print(f"[{utc_iso()}] Wrote: {args.out}")


if __name__ == "__main__":
    main()