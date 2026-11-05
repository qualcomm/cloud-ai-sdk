# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


class PytorchTransform:
    @classmethod
    def apply(cls, model: Optional[Any]) -> Tuple[Optional[Any], bool]:
        raise NotImplementedError

    @classmethod
    def undo(cls) -> None:
        pass


class MethodPatchTransform(PytorchTransform):
    _target_class: Optional[type] = None
    _method_name: Optional[str] = None
    _originals: dict[tuple[type, str], object] = {}

    @classmethod
    def apply(cls, model: Optional[Any]) -> Tuple[Optional[Any], bool]:
        if cls._target_class is None or cls._method_name is None:
            raise ValueError(f"{cls.__name__}: _target_class and _method_name must be set")
        key = (cls._target_class, cls._method_name)
        if key in cls._originals:
            return model, False
        original = getattr(cls._target_class, cls._method_name)
        cls._originals[key] = original
        setattr(cls._target_class, cls._method_name, cls.replacement)
        return model, True

    @classmethod
    def undo(cls) -> None:
        if cls._target_class is None or cls._method_name is None:
            return
        key = (cls._target_class, cls._method_name)
        if key in cls._originals:
            setattr(cls._target_class, cls._method_name, cls._originals.pop(key))

    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError


class FunctionPatchTransform(PytorchTransform):
    _target_module = None
    _function_name: Optional[str] = None
    _replacement: Optional[Callable] = None
    _originals: dict[tuple[object, str], object] = {}

    @classmethod
    def apply(cls, model: Optional[Any]) -> Tuple[Optional[Any], bool]:
        if cls._target_module is None or cls._function_name is None or cls._replacement is None:
            raise ValueError(f"{cls.__name__}: target module, function name, and replacement must be set")
        key = (cls._target_module, cls._function_name)
        if key in cls._originals:
            return model, False
        cls._originals[key] = getattr(cls._target_module, cls._function_name)
        setattr(cls._target_module, cls._function_name, cls._replacement)
        return model, True

    @classmethod
    def undo(cls) -> None:
        if cls._target_module is None or cls._function_name is None:
            return
        key = (cls._target_module, cls._function_name)
        if key in cls._originals:
            setattr(cls._target_module, cls._function_name, cls._originals.pop(key))
