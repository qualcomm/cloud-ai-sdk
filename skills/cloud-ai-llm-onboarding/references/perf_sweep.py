# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Measure causal-LM fixed-context and optional CCL QPC performance.

Run this script with the interpreter that imports the selected QEfficient
package. Supply the model, compile configuration, context points, and one
representative prompt/generation pair. The JSON output contains TTFT and decode
tokens per second for every measured point. Use --append to add new points.
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepConfig:
    """Causal-LM compile and workload inputs."""

    model: str
    num_cores: int
    device_ids: list[int]
    prefill_seq_len: int
    ctx_lens: list[int]
    prompt_tokens: int
    generation_len: int
    workload: str
    batch_size: int
    mxfp6_matmul: bool | None
    mxint8_kv_cache: bool | None
    ccl_prefill_ctx_lens: list[int] | None
    ccl_decode_ctx_lens: list[int] | None

    @property
    def num_devices(self) -> int:
        return len(self.device_ids)

    @property
    def ccl_enabled(self) -> bool:
        return self.ccl_prefill_ctx_lens is not None

    @property
    def ccl_max_ctx(self) -> int:
        if self.ccl_decode_ctx_lens is None:
            raise ValueError("CCL context lengths are not configured")
        return max(self.ccl_decode_ctx_lens)

    @property
    def ccl_compile_kwargs(self) -> dict[str, list[int]]:
        if self.ccl_prefill_ctx_lens is None or self.ccl_decode_ctx_lens is None:
            raise ValueError("CCL context lengths are not configured")
        return {
            "comp_ctx_lengths_prefill": self.ccl_prefill_ctx_lens,
            "comp_ctx_lengths_decode": self.ccl_decode_ctx_lens,
        }


def parse_args() -> argparse.Namespace:
    """Parse required model, compiler, and workload inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--num-cores", type=int, required=True)
    parser.add_argument("--device-ids", type=int, nargs="+", required=True)
    parser.add_argument("--prefill-seq-len", type=int, required=True)
    parser.add_argument("--ctx-lens", type=int, nargs="+", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generation-len", type=int, required=True)
    parser.add_argument("--workload", required=True, help="Label stored in JSON")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--mxfp6-matmul",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set MXFP6 MatMul compression when the target artifact uses it.",
    )
    parser.add_argument(
        "--mxint8-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set MXINT8 KV-cache precision when the target artifact uses it.",
    )
    parser.add_argument(
        "--ccl",
        action="store_true",
        help="Also compile and measure one multi-specialization CCL QPC.",
    )
    parser.add_argument("--ccl-prefill-ctx-lens", type=int, nargs="+")
    parser.add_argument("--ccl-decode-ctx-lens", type=int, nargs="+")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out", default="perf_sweep_results.json")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> SweepConfig:
    """Validate inputs that argparse cannot express directly."""

    if args.num_cores <= 0 or args.prefill_seq_len <= 0:
        raise ValueError("--num-cores and --prefill-seq-len must be positive")
    if any(value < 0 for value in args.device_ids):
        raise ValueError("--device-ids values must be non-negative")
    if any(value <= 0 for value in args.ctx_lens):
        raise ValueError("--ctx-lens values must be positive")
    if args.prompt_tokens < 0 or args.generation_len <= 0:
        raise ValueError("--prompt-tokens must be non-negative and --generation-len positive")
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("--iterations and --batch-size must be positive")
    if args.ccl:
        if not args.ccl_prefill_ctx_lens or not args.ccl_decode_ctx_lens:
            raise ValueError(
                "--ccl requires --ccl-prefill-ctx-lens and --ccl-decode-ctx-lens"
            )
        for name, values in (
            ("--ccl-prefill-ctx-lens", args.ccl_prefill_ctx_lens),
            ("--ccl-decode-ctx-lens", args.ccl_decode_ctx_lens),
        ):
            if len(values) > 5 or any(value <= 0 or value % 1024 for value in values):
                raise ValueError(f"{name} needs one to five positive multiples of 1024")
    elif args.ccl_prefill_ctx_lens or args.ccl_decode_ctx_lens:
        raise ValueError("CCL context lists require --ccl")

    return SweepConfig(
        model=args.model,
        num_cores=args.num_cores,
        device_ids=args.device_ids,
        prefill_seq_len=args.prefill_seq_len,
        ctx_lens=args.ctx_lens,
        prompt_tokens=args.prompt_tokens,
        generation_len=args.generation_len,
        workload=args.workload,
        batch_size=args.batch_size,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        ccl_prefill_ctx_lens=args.ccl_prefill_ctx_lens,
        ccl_decode_ctx_lens=args.ccl_decode_ctx_lens,
    )


def build_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Build text with approximately the requested tokenizer length."""

    seed = (
        "Machine learning workloads have different context and generation "
        "requirements. Measure the intended workload before deployment. "
    )
    repetitions = max(200, (target_tokens // 20) + 100)
    token_ids = tokenizer(
        seed * repetitions, add_special_tokens=False
    )["input_ids"]
    if len(token_ids) < target_tokens:
        raise RuntimeError(f"seed too short: got {len(token_ids)} < {target_tokens}")
    return tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)


def effective_ccl_ctx(config: SweepConfig) -> int | None:
    """Return the smallest CCL decode bucket that holds the workload."""

    required = config.prompt_tokens + config.generation_len
    for bucket in sorted(config.ccl_decode_ctx_lens or []):
        if bucket >= required:
            return bucket
    return None


def compile_model(
    config: SweepConfig, causal_lm_class: Any, *, ccl: bool, ctx_len: int | None
) -> tuple[Any, float, float]:
    """Load and compile one fixed-context or CCL causal-LM QPC."""

    load_started = time.perf_counter()
    if ccl:
        model = causal_lm_class.from_pretrained(
            config.model, qaic_config={"ccl_enabled": True}
        )
    else:
        model = causal_lm_class.from_pretrained(config.model)
    load_sec = round(time.perf_counter() - load_started, 2)

    compile_kwargs: dict[str, Any] = {
        "num_cores": config.num_cores,
        "num_devices": config.num_devices,
        "batch_size": config.batch_size,
        "prefill_seq_len": config.prefill_seq_len,
    }
    if config.mxfp6_matmul is not None:
        compile_kwargs["mxfp6_matmul"] = config.mxfp6_matmul
    if config.mxint8_kv_cache is not None:
        compile_kwargs["mxint8_kv_cache"] = config.mxint8_kv_cache
    if ccl:
        compile_kwargs.update(config.ccl_compile_kwargs)
    elif ctx_len is not None:
        compile_kwargs["ctx_len"] = ctx_len
    else:
        raise ValueError("a fixed-context compile needs ctx_len")

    compile_started = time.perf_counter()
    model.compile(**compile_kwargs)
    compile_sec = round(time.perf_counter() - compile_started, 2)
    return model, load_sec, compile_sec


def measure(
    config: SweepConfig,
    ctx_len: int,
    tokenizer: Any,
    model: Any,
    *,
    qpc_kind: str,
    iterations: int,
    load_sec: float,
    compile_sec: float,
) -> dict[str, Any]:
    """Run one QPC and return its structured performance record."""

    qpc_max_ctx = config.ccl_max_ctx if qpc_kind == "ccl" else ctx_len
    record: dict[str, Any] = {
        "ctx_len": ctx_len,
        "qpc_kind": qpc_kind,
        "qpc_max_ctx": qpc_max_ctx,
        "workload": config.workload,
        "prefill_seq_len": config.prefill_seq_len,
        "prompt_tokens": config.prompt_tokens,
        "generation_len": config.generation_len,
        "num_cores": config.num_cores,
        "num_devices": config.num_devices,
        "device_ids": config.device_ids,
        "load_sec": load_sec,
        "compile_sec": compile_sec,
        "iterations": iterations,
    }
    if config.prompt_tokens + config.generation_len > ctx_len:
        record.update(
            status="skipped",
            reason=(
                f"prompt({config.prompt_tokens}) + generation({config.generation_len}) "
                f"exceeds ctx_len {ctx_len}"
            ),
        )
        return record

    prompt = build_prompt(tokenizer, config.prompt_tokens)
    record["actual_prompt_tokens"] = len(
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )
    run_started = time.perf_counter()
    output = model.generate(
        prompts=[prompt],
        tokenizer=tokenizer,
        device_id=config.device_ids,
        generation_len=config.generation_len,
        iteration=iterations,
    )
    record["wallclock_sec"] = round(time.perf_counter() - run_started, 2)

    metrics = output.perf_metrics
    record.update(
        ttft_sec=round(float(metrics.prefill_time), 4),
        decode_tok_per_sec=round(float(metrics.decode_perf), 2),
        total_tok_per_sec=round(float(metrics.total_perf), 2),
        total_e2e_sec=round(float(metrics.total_time), 2),
        status="ok",
    )
    return record


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identify a result by context, workload, and request shape."""

    return (
        record.get("ctx_len"),
        record.get("qpc_kind"),
        record.get("workload"),
        record.get("prompt_tokens"),
        record.get("generation_len"),
    )


def load_results(
    config: SweepConfig, args: argparse.Namespace
) -> tuple[dict[str, Any], set[tuple[Any, ...]]]:
    """Create or extend the JSON result document."""

    output = Path(args.out)
    if args.append and output.exists():
        results = json.loads(output.read_text())
        if results.get("model") not in {None, config.model}:
            raise ValueError("--append output belongs to a different model")
        results.setdefault("sweep", [])
        results["last_appended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return results, {record_key(record) for record in results["sweep"]}

    results = {
        "model": config.model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(config),
        "iterations": args.iterations,
        "sweep": [],
    }
    return results, set()


def write_results(results: dict[str, Any], output: Path) -> None:
    """Persist partial results so a failed later point can be resumed."""

    output.write_text(json.dumps(results, indent=2) + "\n")


def append_record(
    results: dict[str, Any],
    existing: set[tuple[Any, ...]],
    record: dict[str, Any],
    output: Path,
) -> None:
    """Store a record and prevent duplicate work on --append."""

    results["sweep"].append(record)
    existing.add(record_key(record))
    write_results(results, output)


def failure_record(
    config: SweepConfig, ctx_len: int, qpc_kind: str, error: Exception
) -> dict[str, Any]:
    """Represent an exception without losing the remaining sweep."""

    return {
        "ctx_len": ctx_len,
        "qpc_kind": qpc_kind,
        "qpc_max_ctx": config.ccl_max_ctx if qpc_kind == "ccl" else ctx_len,
        "workload": config.workload,
        "prompt_tokens": config.prompt_tokens,
        "generation_len": config.generation_len,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> int:
    """Execute the configured causal-LM sweep."""

    args = parse_args()
    try:
        config = validate_args(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    from QEfficient import QEFFAutoModelForCausalLM
    from transformers import AutoTokenizer

    print(
        f"[sweep] model={config.model} ctx_lens={config.ctx_lens} "
        f"workload={config.workload} ccl={config.ccl_enabled} "
        f"device_ids={config.device_ids} append={args.append}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    results, existing = load_results(config, args)
    output = Path(args.out)

    for ctx_len in config.ctx_lens:
        key = (
            ctx_len,
            "fixed",
            config.workload,
            config.prompt_tokens,
            config.generation_len,
        )
        if key in existing:
            print(f"[sweep] fixed ctx_len={ctx_len}: already measured", flush=True)
            continue
        if config.prompt_tokens + config.generation_len > ctx_len:
            record = measure(
                config,
                ctx_len,
                tokenizer,
                model=None,
                qpc_kind="fixed",
                iterations=args.iterations,
                load_sec=0.0,
                compile_sec=0.0,
            )
            append_record(results, existing, record, output)
            print(f"[sweep] fixed ctx_len={ctx_len}: {record['status']}", flush=True)
            continue
        try:
            model, load_sec, compile_sec = compile_model(
                config, QEFFAutoModelForCausalLM, ccl=False, ctx_len=ctx_len
            )
            record = measure(
                config,
                ctx_len,
                tokenizer,
                model,
                qpc_kind="fixed",
                iterations=args.iterations,
                load_sec=load_sec,
                compile_sec=compile_sec,
            )
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            record = failure_record(config, ctx_len, "fixed", error)
        append_record(results, existing, record, output)
        print(f"[sweep] fixed ctx_len={ctx_len}: {record['status']}", flush=True)

    if config.ccl_enabled:
        ccl_ctx = effective_ccl_ctx(config)
        if ccl_ctx is None:
            print("[sweep] CCL skipped: workload exceeds every decode bucket", flush=True)
        else:
            key = (
                ccl_ctx,
                "ccl",
                config.workload,
                config.prompt_tokens,
                config.generation_len,
            )
            if key not in existing:
                try:
                    model, load_sec, compile_sec = compile_model(
                        config, QEFFAutoModelForCausalLM, ccl=True, ctx_len=None
                    )
                    record = measure(
                        config,
                        ccl_ctx,
                        tokenizer,
                        model,
                        qpc_kind="ccl",
                        iterations=args.iterations,
                        load_sec=load_sec,
                        compile_sec=compile_sec,
                    )
                except Exception as error:  # noqa: BLE001
                    traceback.print_exc()
                    record = failure_record(config, ccl_ctx, "ccl", error)
                append_record(results, existing, record, output)
                print(f"[sweep] CCL ctx_len={ccl_ctx}: {record['status']}", flush=True)

    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_results(results, output)
    print(f"[sweep] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
