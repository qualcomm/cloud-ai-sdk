####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""qaic_models.base.compile — qaic-compile flag-building and invocation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def build_compile_cmd(
    model: str,
    cores: int,
    aic_binary_dir,
    *,
    mos: Optional[int] = None,
    ols: Optional[int] = None,
    multicast_weights: bool = False,
    depth_first: bool = False,
    quantize: bool = False,
    node_precision_info=None,
    extra_args=(),
) -> list[str]:
    """Build the qaic-compile argv list. Pure function — no subprocess, no filesystem access.

    `extra_args` carries recipe-specific compiler flags that are not yet shared
    by a second model (for example MDP partitioning or network specialization).
    """
    cmd = [
        "/opt/qti-aic/exec/qaic-compile",
        f"-m={model}",
        "-aic-hw",
        "-aic-hw-version=ai100",
        f"-aic-num-cores={cores}",
        f"-aic-binary-dir={aic_binary_dir}",
    ]
    if mos:
        cmd.append(f"-mos={mos}")
    if ols:
        cmd.append(f"-ols={ols}")
    if multicast_weights:
        cmd.append("-multicast-weights")
    if depth_first:
        cmd.append("-aic-enable-depth-first")
    if quantize:
        cmd.extend([
            "-convert-to-quantize",
            "-quantization-schema-activations=symmetric_with_uint8",
            "-quantization-schema-constants=symmetric_with_uint8",
            "-quantization-precision=Int8",
            "-use-random-input-data=gaussian",
        ])
    else:
        cmd.append("-convert-to-fp16")
    if node_precision_info:
        cmd.append(f"-node-precision-info={node_precision_info}")
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def _prepare_compile(
    model: str,
    cores: int,
    aic_binary_dir,
    *,
    cwd,
    mos: Optional[int] = None,
    ols: Optional[int] = None,
    multicast_weights: bool = False,
    depth_first: bool = False,
    quantize: bool = False,
    node_precision_info=None,
    extra_args=(),
) -> list[str]:
    """Remove any pre-existing target dir (qaic-compile fails if it already
    exists) and build its argv, ready for execution by the caller."""
    aic_binary_dir = Path(aic_binary_dir)
    if aic_binary_dir.exists():
        subprocess.run(["rm", "-rf", str(aic_binary_dir)], check=True)
    cmd = build_compile_cmd(
        model, cores, aic_binary_dir,
        mos=mos, ols=ols, multicast_weights=multicast_weights,
        depth_first=depth_first, quantize=quantize,
        node_precision_info=node_precision_info,
        extra_args=extra_args,
    )
    print("+", " ".join(cmd), flush=True)
    return cmd


def qaic_compile(
    model: str,
    cores: int,
    aic_binary_dir,
    *,
    cwd,
    mos: Optional[int] = None,
    ols: Optional[int] = None,
    multicast_weights: bool = False,
    depth_first: bool = False,
    quantize: bool = False,
    node_precision_info=None,
    extra_args=(),
) -> None:
    """Compile model with qaic-compile into aic_binary_dir, removing any
    pre-existing directory first (qaic-compile fails if the target already
    exists)."""
    cmd = _prepare_compile(
        model, cores, aic_binary_dir,
        cwd=cwd, mos=mos, ols=ols, multicast_weights=multicast_weights,
        depth_first=depth_first, quantize=quantize,
        node_precision_info=node_precision_info,
        extra_args=extra_args,
    )
    subprocess.run(cmd, cwd=str(cwd), check=True)


def qaic_compile_parallel(compiles) -> None:
    """Compile independent QPCs concurrently and always.

    Each entry of `compiles` is a dict of the keyword arguments
    ``qaic_compile`` takes. Targets must be distinct QPC dirs;
    ``qaic-compile`` writes only into its own ``-aic-binary-dir``, so
    concurrent invocations sharing a cwd do not collide. Their stdout and
    stderr interleave in the run log; every compile is waited for even
    after one fails, and a nonzero exit raises ``CalledProcessError``.
    """
    launched = []
    for spec in compiles:
        cmd = _prepare_compile(**spec)
        launched.append((spec["cwd"], subprocess.Popen(cmd, cwd=str(spec["cwd"]))))
    first_failure = None
    for cwd, proc in launched:
        returncode = proc.wait()
        if returncode != 0 and first_failure is None:
            first_failure = (returncode, proc.args)
    if first_failure is not None:
        raise subprocess.CalledProcessError(*first_failure)
