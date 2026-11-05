#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#
# base_pytorch_transform.py — Base class framework for pre-export PyTorch transforms
#
# Mirrors the QEfficient.base.pytorch_transforms pattern from the efficient-transformers repo.
# Apply transforms to the loaded nn.Module BEFORE calling torch.onnx.export().
#
# Framework classes:
#
#   ModuleMappingTransform     — replace module types in-place (swaps __class__, no re-init).
#                                Use for custom RMSNorm, custom attention, custom pooler, etc.
#
#   MethodPatchTransform       — replace one named method on a target class at the class level.
#                                Use for forward() overrides that eliminate unsupported ops.
#                                Call undo() after torch.onnx.export() to restore the original.
#
# Ready-to-use concrete transforms:
#
#   GridSampleTransform        — patches F.grid_sample globally with a bilinear subgraph that
#                                traces to supported ONNX ops (Floor, Cast, Gather/GatherElements,
#                                Clip, Mul, Add). Handles padding_mode='zeros'/'border' and
#                                both align_corners=True/False. Falls back to the original for
#                                modes other than 'bilinear'. Call undo() after export.
#
#   MaxPool3dToMaxPool2dTransform — replaces nn.MaxPool3d(kernel[0]=1) with MaxPool2d + channel-
#                                stride slice. Cloud AI 100 supports 1D/2D pooling only; models
#                                using MaxPool3d (e.g. LPRNet) fail to compile without this.
#
# CLI:
#   python base_pytorch_transform.py --list   # show all available transforms
#
# Library usage (export_<model>_onnx.py):
#
#   from base_pytorch_transform import GridSampleTransform, MethodPatchTransform, PytorchTransform
#
#   # Apply generic transform — no model object needed:
#   GridSampleTransform.apply(None)
#
#   # Model-specific method patch:
#   class FixDecoder(MethodPatchTransform):
#       _target_class = MyDecoderClass
#       _method_name  = "forward"
#
#       @staticmethod
#       def replacement(self, *args, **kwargs):
#           # ... patched implementation that avoids NonZero / GridSample ...
#           pass
#
#   FixDecoder.apply(model)
#   GridSampleTransform.apply(model)
#   torch.onnx.export(model, ...)
#   FixDecoder.undo()
#   GridSampleTransform.undo()
#
# Tip: run `python base_onnx_transform.py --list` to see complementary post-export transforms.
#      Pre-export (this file) and post-export (base_onnx_transform.py) transforms are independent
#      and both should be applied — pre-export first, then export, then post-export.

import sys
import logging
import types
from typing import Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PytorchTransform:
    """Base class for pre-export PyTorch transforms. Do not instantiate — use `apply`."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` class method.")

    @classmethod
    def apply(cls, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        """Apply the transform. Returns (model, changed).

        model may be None for global transforms (e.g. GridSampleTransform) that patch
        a module-level function rather than walking the model graph.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    @classmethod
    def undo(cls) -> None:
        """Restore the original state (if applicable). No-op by default."""
        pass


# ---------------------------------------------------------------------------
# ModuleMappingTransform
# ---------------------------------------------------------------------------

class ModuleMappingTransform(PytorchTransform):
    """Replace nn.Module subclass types in-place by swapping __class__.

    Define _module_mapping as a dict mapping {OriginalClass: ReplacementClass}.
    The replacement class must be layout-compatible (same __init__ signature or
    accepts the original module's __dict__).  __class__ is swapped in-place so
    parameters and buffers are preserved without re-initialisation.

    Example:
        class UseCustomNorm(ModuleMappingTransform):
            _module_mapping = {LlamaRMSNorm: CustomRMSNormAIC}
    """

    _module_mapping: Dict[Type[nn.Module], Type[nn.Module]] = {}

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        if model is None:
            raise ValueError("ModuleMappingTransform requires a model argument")
        changed = False
        for module in model.modules():
            replacement = cls._module_mapping.get(type(module))
            if replacement is not None:
                module.__class__ = replacement
                if hasattr(module, "__qeff_init__"):
                    module.__qeff_init__()
                changed = True
        if changed:
            log.info("%s: replaced %d module(s)",
                     cls.__name__,
                     sum(1 for m in model.modules() if type(m) in cls._module_mapping.values()))
        return model, changed


# ---------------------------------------------------------------------------
# ModuleMutatorTransform
# ---------------------------------------------------------------------------

class ModuleMutatorTransform(PytorchTransform):
    """Replace nn.Module instances by constructing a brand-new module via mutate().

    Unlike ModuleMappingTransform (swaps __class__ in-place, all parameters
    preserved as-is), ModuleMutatorTransform calls mutate() to build a completely
    new module — use this when the replacement requires a different __init__ or
    when weights must be repacked (e.g. fused QKV → split Q/K/V linears,
    static position-bias buffer pre-computation, fused gate+up → separate).

    Define:
      _match_class — the nn.Module type to replace
      mutate()     — classmethod(original_module, parent_module) → nn.Module

    The walk is recursive (DFS). When the model itself is an instance of
    _match_class, mutate() is called with parent_module=None and the returned
    module replaces the model root.

    Example:
        class SplitQKVTransform(ModuleMutatorTransform):
            _match_class = WindowAttentionFused

            @classmethod
            def mutate(cls, original: nn.Module,
                       parent: Optional[nn.Module]) -> nn.Module:
                new = WindowAttentionSplit(
                    dim=original.dim, num_heads=original.num_heads,
                )
                d = original.dim
                new.q_proj.weight.data.copy_(original.qkv.weight[:d])
                new.k_proj.weight.data.copy_(original.qkv.weight[d:2*d])
                new.v_proj.weight.data.copy_(original.qkv.weight[2*d:])
                return new
    """

    _match_class: Optional[Type[nn.Module]] = None

    @classmethod
    def apply(cls, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        if cls._match_class is None:
            raise ValueError(f"{cls.__name__}: _match_class must be set")
        if model is None:
            raise ValueError("ModuleMutatorTransform requires a model argument")
        changed = False
        for name, child in list(model.named_children()):
            if isinstance(child, cls._match_class):
                setattr(model, name, cls.mutate(child, model))
                changed = True
            else:
                _, child_changed = cls.apply(child)
                changed = changed or child_changed
        if isinstance(model, cls._match_class):
            model = cls.mutate(model, None)
            changed = True
        if changed:
            log.info("%s: replaced module(s) matching %s",
                     cls.__name__, cls._match_class.__name__)
        return model, changed

    @classmethod
    def mutate(cls, original_module: nn.Module,
               parent_module: Optional[nn.Module]) -> nn.Module:
        raise NotImplementedError(
            f"{cls.__name__}: define a mutate(original_module, parent_module) classmethod"
        )


# ---------------------------------------------------------------------------
# MethodPatchTransform
# ---------------------------------------------------------------------------

class MethodPatchTransform(PytorchTransform):
    """Replace a named method on a target class at the class level.

    Subclass and define:
      _target_class  — the nn.Module (or any class) whose method to patch
      _method_name   — name of the method to replace (e.g. "forward")
      replacement    — static method with the same signature as the original

    The patch is applied globally (all instances) and is reversed by undo().
    Always call undo() after torch.onnx.export() to avoid contaminating
    subsequent uses of the target class.

    Example:
        class FixAttentionForward(MethodPatchTransform):
            _target_class = MyAttentionClass
            _method_name  = "forward"

            @staticmethod
            def replacement(self, hidden_states, attention_mask=None, **kwargs):
                # ... replacement that avoids NonZero / dynamic indexing ...
                pass
    """

    _target_class: Optional[type] = None
    _method_name:  Optional[str]  = None
    _originals:    Dict[Tuple[type, str], object] = {}   # class-level registry

    @classmethod
    def apply(cls, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        if cls._target_class is None or cls._method_name is None:
            raise ValueError(f"{cls.__name__}: _target_class and _method_name must be set")
        key = (cls._target_class, cls._method_name)
        if key in cls._originals:
            log.warning("%s: already applied — undo() first if re-applying", cls.__name__)
            return model, False
        original = getattr(cls._target_class, cls._method_name, None)
        if original is None:
            log.warning("%s: method %s.%s not found — skipping",
                        cls.__name__, cls._target_class.__name__, cls._method_name)
            return model, False
        cls._originals[key] = original
        setattr(cls._target_class, cls._method_name, cls.replacement)
        log.info("%s: patched %s.%s", cls.__name__, cls._target_class.__name__, cls._method_name)
        return model, True

    @classmethod
    def undo(cls) -> None:
        """Restore the original method."""
        if cls._target_class is None or cls._method_name is None:
            return
        key = (cls._target_class, cls._method_name)
        if key not in cls._originals:
            return
        setattr(cls._target_class, cls._method_name, cls._originals.pop(key))
        log.info("%s: restored %s.%s", cls.__name__, cls._target_class.__name__, cls._method_name)

    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError("Subclasses must define a replacement() static method")


# ---------------------------------------------------------------------------
# ExternalModuleMapperTransform
# ---------------------------------------------------------------------------

class ExternalModuleMapperTransform(PytorchTransform):
    """Patch named methods on matching module instances at the instance level.

    Supports matching by class type or class name string — the string form
    requires no import and is useful for HuggingFace models where the exact
    class lives in a deeply nested private submodule.

    Patches are applied per-instance via types.MethodType: each matching module
    gets its own bound replacement function, so the class itself is never
    modified and no undo() is required.

    Compared to MethodPatchTransform:
      MethodPatchTransform        — class-level (all instances); requires undo()
      ExternalModuleMapperTransform — instance-level (each matched instance); no undo()

    Use ExternalModuleMapperTransform when:
      - The class cannot be cleanly imported (buried in private submodules), or
      - The same class is instantiated multiple times and only specific
        instances (matched by type or name) need patching.

    Define one or both class variables:
      _match_class_replace_method  — {ClassType: {method_name: fn}}
        Type matching takes priority over string matching.
      _match_string_replace_method — {class_name_str: {method_name: fn}}
        Matches module.__class__.__name__ (unqualified, not the full dotted path).

    Example — patch by class name string (no import required):
        def _fixed_forward(self, x, mask=None):
            # ... replacement that avoids NonZero / GridSample ...
            pass

        class FixWindowAttention(ExternalModuleMapperTransform):
            _match_string_replace_method = {
                "WindowAttention": {"forward": _fixed_forward},
            }

        FixWindowAttention.apply(model)     # no undo() call needed
        torch.onnx.export(model, ...)

    Example — patch by type (requires importable class):
        from transformers.models.foo.modeling_foo import FooAttention

        class FixFooAttention(ExternalModuleMapperTransform):
            _match_class_replace_method = {
                FooAttention: {"forward": _fixed_foo_attention_forward},
            }
    """

    _match_class_replace_method:  Dict[Type[nn.Module], Dict[str, Callable]] = {}
    _match_string_replace_method: Dict[str, Dict[str, Callable]] = {}

    @classmethod
    def apply(cls, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        if model is None:
            raise ValueError("ExternalModuleMapperTransform requires a model argument")
        changed = False
        for module in model.modules():
            method_map = (
                cls._match_class_replace_method.get(type(module))
                or cls._match_string_replace_method.get(module.__class__.__name__)
            )
            if method_map:
                for method_name, replacement_fn in method_map.items():
                    setattr(module, method_name,
                            types.MethodType(replacement_fn, module))
                    changed = True
                    log.info("%s: patched %s.%s (instance-level)",
                             cls.__name__, module.__class__.__name__, method_name)
        return model, changed


# ---------------------------------------------------------------------------
# GridSampleTransform
# ---------------------------------------------------------------------------

_orig_grid_sample = F.grid_sample


def _grid_sample_onnx_safe(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """F.grid_sample replacement using ONNX-traceable ops.

    Supported:   mode='bilinear', padding_mode='zeros'/'border', any align_corners.
    Unsupported: mode='nearest'/'bicubic' — falls back to the original F.grid_sample.

    ONNX ops generated: Floor, Cast(INT64), Sub, Mul, Add, Clip,
                        GatherElements, Reshape, Expand.
    All are in the qaic-compile supported op list (opset 11+).

    Args:
        input:  (N, C, H, W)
        grid:   (N, Hout, Wout, 2)  — normalised coords in [-1, 1]
    Returns:
        (N, C, Hout, Wout)
    """
    if mode != "bilinear":
        return _orig_grid_sample(input, grid, mode=mode,
                                 padding_mode=padding_mode, align_corners=align_corners)

    N, C, H, W = input.shape
    Hout, Wout = grid.shape[1], grid.shape[2]

    gx = grid[..., 0]   # (N, Hout, Wout)
    gy = grid[..., 1]   # (N, Hout, Wout)

    # Denormalise: convert [-1, 1] coordinates to pixel space.
    if align_corners:
        x = (gx + 1.0) * ((W - 1) * 0.5)
        y = (gy + 1.0) * ((H - 1) * 0.5)
    else:
        x = (gx + 1.0) * (W * 0.5) - 0.5
        y = (gy + 1.0) * (H * 0.5) - 0.5

    # Integer corners for bilinear stencil.
    x0f = torch.floor(x)
    y0f = torch.floor(y)
    x1f = x0f + 1.0
    y1f = y0f + 1.0

    # Bilinear weights: distance from each corner.
    wx1 = (x - x0f).unsqueeze(1)                   # (N, 1, Hout, Wout)
    wy1 = (y - y0f).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def _gather(xi_f: torch.Tensor, yi_f: torch.Tensor) -> torch.Tensor:
        """Return input sampled at integer pixel (xi, yi) with the configured padding."""
        if padding_mode == "zeros":
            # Mask out-of-bounds samples to zero.
            valid = (
                (xi_f >= 0.0) & (xi_f < float(W)) &
                (yi_f >= 0.0) & (yi_f < float(H))
            ).float().unsqueeze(1)                  # (N, 1, Hout, Wout)
            xi_i = xi_f.clamp(0, W - 1).long()
            yi_i = yi_f.clamp(0, H - 1).long()
        else:  # 'border'
            valid = torch.ones(N, 1, Hout, Wout,
                               dtype=input.dtype, device=input.device)
            xi_i = xi_f.clamp(0, W - 1).long()
            yi_i = yi_f.clamp(0, H - 1).long()

        # Flat index into the H*W spatial dimension.
        flat_idx = (yi_i * W + xi_i)               # (N, Hout, Wout)
        flat_idx = (
            flat_idx.unsqueeze(1)
                    .expand(N, C, Hout, Wout)
                    .reshape(N, C, Hout * Wout)
        )
        vals = (
            torch.gather(input.reshape(N, C, H * W), 2, flat_idx)
                 .reshape(N, C, Hout, Wout)
        )
        return vals * valid

    p00 = _gather(x0f, y0f)
    p10 = _gather(x1f, y0f)
    p01 = _gather(x0f, y1f)
    p11 = _gather(x1f, y1f)

    return p00 * wy0 * wx0 + p10 * wy0 * wx1 + p01 * wy1 * wx0 + p11 * wy1 * wx1


class GridSampleTransform(PytorchTransform):
    """Patch F.grid_sample globally with a bilinear ONNX-safe replacement.

    Must be called BEFORE torch.onnx.export(). Call undo() after export to
    restore the original F.grid_sample for downstream code.

    Applies globally — the model argument is unused (pass None or the model).
    GridSample appears at the F.* call site, not as an nn.Module.

    Example:
        GridSampleTransform.apply(None)
        torch.onnx.export(model, ...)
        GridSampleTransform.undo()
    """

    _patched = False

    @classmethod
    def apply(cls, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        if cls._patched:
            log.warning("GridSampleTransform: already applied — undo() first if re-applying")
            return model, False
        import torch.nn.functional as _F_mod
        _F_mod.grid_sample = _grid_sample_onnx_safe
        # Also patch via the `F` reference used in this module (for completeness).
        globals()["F"] = _F_mod
        cls._patched = True
        log.info("GridSampleTransform: patched F.grid_sample with bilinear ONNX-safe replacement")
        return model, True

    @classmethod
    def undo(cls) -> None:
        if not cls._patched:
            return
        import torch.nn.functional as _F_mod
        _F_mod.grid_sample = _orig_grid_sample
        cls._patched = False
        log.info("GridSampleTransform: restored original F.grid_sample")


# ---------------------------------------------------------------------------
# MaxPool3dToMaxPool2dTransform
# ---------------------------------------------------------------------------

class MaxPool3dToMaxPool2dTransform(ModuleMutatorTransform):
    """Replace nn.MaxPool3d(kernel[0]=1) with MaxPool2d + channel-stride slice.

    Cloud AI 100 supports 1D/2D pooling only — MaxPool3d fails to compile.
    Applies to any model using MaxPool3d on 4D input [N,C,H,W] as a
    channel-subsampling + spatial-pooling op (kernel depth = 1).

    Decomposition (mathematically exact, max diff = 0.0 on random inputs):
      MaxPool3d(k=(1,kh,kw), s=(sc,sh,sw), p=(0,ph,pw))
      →  MaxPool2d(kernel=(kh,kw), stride=(sh,sw), padding=(ph,pw))
         followed by  x[:, ::sc, :, :]  (channel stride-slice)

    Raises RuntimeError for kernel[0] > 1 (non-trivial depth kernel — needs
    manual analysis).  No-ops when no MaxPool3d modules are present.
    """

    _match_class = nn.MaxPool3d

    @classmethod
    def mutate(cls, original: nn.Module, parent: Optional[nn.Module]) -> nn.Module:
        def _to3(v):
            return v if isinstance(v, (list, tuple)) else (v, v, v)

        k = _to3(original.kernel_size)
        s = _to3(original.stride) if original.stride is not None else k
        p = _to3(original.padding)

        if k[0] != 1:
            raise RuntimeError(
                f"MaxPool3dToMaxPool2dTransform: kernel_size[0]={k[0]} != 1 — "
                "non-trivial depth kernel requires manual analysis"
            )

        class _MaxPool2dWithChannelStride(nn.Module):
            def __init__(self, pool: nn.MaxPool2d, stride_ch: int) -> None:
                super().__init__()
                self.pool2d = pool
                self.stride_ch = stride_ch

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = self.pool2d(x)
                return x[:, :: self.stride_ch, :, :] if self.stride_ch > 1 else x

        pool2d = nn.MaxPool2d(
            kernel_size=(k[1], k[2]),
            stride=(s[1], s[2]),
            padding=(p[1], p[2]),
        )
        return _MaxPool2dWithChannelStride(pool2d, s[0])


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

class PytorchTransformPipeline:
    """Apply a sequence of PytorchTransform subclasses in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def apply(self, model: Optional[nn.Module]) -> Tuple[Optional[nn.Module], bool]:
        any_changed = False
        for t in self.transforms:
            model, changed = t.apply(model)
            any_changed = any_changed or changed
        return model, any_changed

    def undo(self) -> None:
        for t in reversed(self.transforms):
            t.undo()


def run_transforms(model: Optional[nn.Module], transforms: list) -> Tuple[Optional[nn.Module], bool]:
    """Apply a list of PytorchTransform subclasses and return (model, any_changed)."""
    return PytorchTransformPipeline(transforms).apply(model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        import inspect
        import sys as _sys
        _abstract_bases = (
            ModuleMappingTransform,
            ModuleMutatorTransform,
            MethodPatchTransform,
            ExternalModuleMapperTransform,
        )
        subclasses = [
            cls for _, cls in inspect.getmembers(_sys.modules[__name__], inspect.isclass)
            if issubclass(cls, PytorchTransform) and cls is not PytorchTransform
            and cls not in _abstract_bases
        ]
        print("Available concrete transforms in base_pytorch_transform.py:")
        for cls in subclasses:
            doc = (cls.__doc__ or "").strip().splitlines()[0]
            print(f"  {cls.__name__:<30} {doc}")
        print()
        print("Abstract base classes (subclass these for model-specific patches):")
        for cls in _abstract_bases:
            doc = (cls.__doc__ or "").strip().splitlines()[0]
            print(f"  {cls.__name__:<30} {doc}")
        sys.exit(0)

    print(f"Usage: {sys.argv[0]} --list", file=sys.stderr)
    sys.exit(1)
