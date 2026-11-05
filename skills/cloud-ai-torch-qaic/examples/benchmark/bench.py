# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
torch_qaic inference benchmark template.

Usage:
    python bench.py                     # BERT-base, seq_len=128, batch=1
    python bench.py --model resnet50    # ResNet50, batch=1
    python bench.py --model bert --seq-len 256 --batch 4
    python bench.py --warmup 5 --iters 20

Measures wall-clock latency (host-side, includes device sync) and reports
the QAIC/CPU op split so you can see how much work is landing on-device.
"""

import argparse, time
import torch_qaic, torch
from torch_qaic import stats


def load_model(name, device):
    if name == "bert":
        from transformers import BertModel, BertConfig
        return BertModel(BertConfig()).eval().to(device), "transformer"
    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        return resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device), "cnn"
    raise ValueError(f"Unknown model: {name}. Add it to load_model().")


def make_inputs(model_type, model, batch, seq_len, device):
    if model_type == "transformer":
        from transformers import BertConfig
        cfg = model.config if hasattr(model, "config") else BertConfig()
        return {
            "input_ids":      torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device),
            "attention_mask": torch.ones(batch, seq_len, dtype=torch.long, device=device),
        }
    if model_type == "cnn":
        return {"x": torch.randn(batch, 3, 224, 224, device=device)}
    raise ValueError(f"Unknown model type: {model_type}")


def run(model, inputs, n):
    with torch.no_grad():
        for _ in range(n):
            if isinstance(inputs, dict) and "x" in inputs:
                model(inputs["x"])
            else:
                model(**inputs)
            torch.qaic.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="bert",   choices=["bert", "resnet50"])
    parser.add_argument("--device",  default="qaic:0")
    parser.add_argument("--batch",   type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128,  help="transformer only")
    parser.add_argument("--warmup",  type=int, default=3)
    parser.add_argument("--iters",   type=int, default=10)
    args = parser.parse_args()

    device = args.device
    print(f"device: {device}  ({torch.qaic.get_device_info(0).name})")

    model, model_type = load_model(args.model, device)
    inputs = make_inputs(model_type, model, args.batch, args.seq_len, device)

    print(f"warming up ({args.warmup} iters)...")
    run(model, inputs, args.warmup)

    stats.reset_recent_stats()
    t0 = time.perf_counter()
    run(model, inputs, args.iters)
    elapsed = time.perf_counter() - t0

    latency_ms  = elapsed / args.iters * 1000
    throughput  = args.iters * args.batch / elapsed

    op = stats.get_global_op_stats()
    total_compute = op["Compute"] + op["ImplCPUFallback"] + op["UnimplCPUFallback"]
    qaic_pct = op["Compute"] / total_compute * 100 if total_compute else 0

    print(f"\n--- results ({args.model}, batch={args.batch}) ---")
    print(f"latency:    {latency_ms:.2f} ms")
    print(f"throughput: {throughput:.2f} samples/sec")
    print(f"\ncompute ops: {total_compute}")
    print(f"  on QAIC:           {op['Compute']}  ({qaic_pct:.0f}%)")
    print(f"  CPU ImplFallback:  {op['ImplCPUFallback']}  (kernel exists, dtype/layout mismatch)")
    print(f"  CPU UnimplFallback:{op['UnimplCPUFallback']}  (no kernel)")

    fallbacks = stats.get_global_fallback_op_schemas()
    if fallbacks:
        print("\nfallbacks per forward pass:")
        for schema, count in fallbacks.items():
            short = schema.split("(")[0]
            per_fwd = count // args.iters
            if per_fwd > 0:
                print(f"  {per_fwd:3d}x  {short}")


if __name__ == "__main__":
    main()
