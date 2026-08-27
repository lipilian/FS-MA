#!/home/liu4000/miniconda3/envs/fs/bin/python
"""Benchmark FoundationStereo PyTorch inference with CUDA-event timing."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_ROOT = REPO_ROOT / "FoundationStereo"
WEIGHT_DIR = REPO_ROOT / "weight" / "23-51-11"
CHECKPOINT_PATH = WEIGHT_DIR / "model_best_bp2.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--disable-xformers", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def main() -> None:
    args = parse_args()
    if args.disable_xformers:
        os.environ["XFORMERS_DISABLED"] = "1"
    else:
        os.environ.pop("XFORMERS_DISABLED", None)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))

    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda:0")
    config = OmegaConf.load(WEIGHT_DIR / "cfg.yaml")
    config.vit_size = config.get("vit_size", "vitl")
    config.height = args.height
    config.width = args.width
    config.valid_iters = args.iters
    config.low_memory = 0
    model = FoundationStereo(config)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    first_attention = model.feature.dino.depth_anything.pretrained.blocks[0].attn
    xformers_available = first_attention.forward.__func__.__globals__.get("XFORMERS_AVAILABLE", False)
    attention_implementation = type(first_attention).__name__
    model.to(device).eval().requires_grad_(False)

    left = torch.rand(1, 3, args.height, args.width, device=device) * 255.0
    right = torch.rand(1, 3, args.height, args.width, device=device) * 255.0
    forward_kwargs = {"iters": args.iters, "test_mode": True, "low_memory": False}

    print(f"Model: FoundationStereo {config.vit_size}")
    print(f"DINO attention: {attention_implementation}, xFormers enabled: {xformers_available}")
    print(f"Input: left/right float32 [1, 3, {args.height}, {args.width}]")

    with torch.inference_mode():
        for _ in range(args.warmup):
            model.forward(left, right, **forward_kwargs)
        torch.cuda.synchronize(device)

        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: list[float] = []
        output = None
        for _ in range(args.runs):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = model.forward(left, right, **forward_kwargs)
            finished.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(started.elapsed_time(finished))

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    print(f"Output: {tuple(output.shape)}, {output.dtype}")
    print(f"Latency: mean={sum(latencies_ms) / len(latencies_ms):.2f} ms")
    print(f"Latency: median={percentile(latencies_ms, 0.50):.2f} ms, p90={percentile(latencies_ms, 0.90):.2f} ms")
    print(
        "GPU memory: allocated peak {:.1f} MiB (+{:.1f} MiB), reserved peak {:.1f} MiB (before {:.1f} MiB)".format(
            peak_allocated / 2**20,
            (peak_allocated - allocated_before) / 2**20,
            peak_reserved / 2**20,
            reserved_before / 2**20,
        )
    )


if __name__ == "__main__":
    main()
