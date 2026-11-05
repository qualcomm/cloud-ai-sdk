# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Structured raw-output contracts for Ultralytics model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class FieldSelector:
    name: str
    index: int | None = None
    bounds: tuple[int, int] | None = None

    @property
    def is_index(self) -> bool:
        return self.index is not None

    @property
    def is_slice(self) -> bool:
        return self.bounds is not None


@dataclass(frozen=True)
class Ranking:
    field: str
    reduction: str


@dataclass(frozen=True)
class RawRecordContract:
    output_name: str
    batch_axis: int
    record_axis: int
    field_axis: int
    fields: tuple[FieldSelector, ...]
    ranking: Ranking

    def _field(self, name: str) -> FieldSelector:
        for field in self.fields:
            if field.name == name:
                return field
        raise ValueError(f"raw output {self.output_name!r} has no field {name!r}")

    def field_index(self, name: str) -> int:
        field = self._field(name)
        if not field.is_index:
            raise ValueError(f"raw output field {name!r} is a slice, not an index")
        return field.index  # type: ignore[return-value]

    def field_slice(self, name: str) -> slice:
        field = self._field(name)
        if not field.is_slice:
            raise ValueError(f"raw output field {name!r} is an index, not a slice")
        start, stop = field.bounds  # type: ignore[misc]
        return slice(start, stop)

    @property
    def field_width(self) -> int:
        """Return the minimum canonical field width implied by the contract."""
        end = 0
        for field in self.fields:
            if field.is_index:
                end = max(end, field.index + 1)  # type: ignore[operator]
            else:
                end = max(end, field.bounds[1])  # type: ignore[index]
        return end

    def records_and_ranks(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a batch-one tensor to records and derive its raw rank values."""
        array = np.asarray(tensor)
        required_rank = max(self.batch_axis, self.record_axis, self.field_axis) + 1
        if array.ndim < required_rank:
            raise ValueError(
                f"raw output {self.output_name!r} requires rank at least {required_rank}, got {array.ndim}"
            )
        if array.shape[self.batch_axis] != 1:
            raise ValueError(
                f"raw output {self.output_name!r} requires batch size 1 on axis {self.batch_axis}, "
                f"got {array.shape[self.batch_axis]}"
            )

        squeezed = np.take(array, 0, axis=self.batch_axis)
        record_axis = self.record_axis - (self.record_axis > self.batch_axis)
        field_axis = self.field_axis - (self.field_axis > self.batch_axis)
        if squeezed.ndim != 2:
            raise ValueError(
                f"raw output {self.output_name!r} must have exactly two non-batch axes, "
                f"got rank {squeezed.ndim} after removing the batch axis"
            )
        records = np.moveaxis(squeezed, (record_axis, field_axis), (0, 1))
        field_width = records.shape[1]
        self._validate_runtime_selectors(field_width)

        ranking_field = self._field(self.ranking.field)
        if ranking_field.is_index:
            ranks = records[:, ranking_field.index]  # type: ignore[index]
        else:
            start, stop = ranking_field.bounds  # type: ignore[misc]
            ranks = records[:, start:stop].max(axis=1)
        return records, np.asarray(ranks)

    def _validate_runtime_selectors(self, field_width: int) -> None:
        for field in self.fields:
            if field.is_index:
                if field.index < 0 or field.index >= field_width:  # type: ignore[operator]
                    raise ValueError(
                        f"raw output field {field.name!r} index {field.index} is outside field width {field_width}"
                    )
            else:
                start, stop = field.bounds  # type: ignore[misc]
                if start < 0 or start >= stop or stop > field_width:
                    raise ValueError(
                        f"raw output field {field.name!r} slice [{start}, {stop}) is outside "
                        f"field width {field_width}"
                    )

    def top_records(self, tensor: np.ndarray, *, limit: int = 20) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError(f"record limit must be a non-negative integer, got {limit!r}")
        records, ranks = self.records_and_ranks(tensor)
        indices = np.argsort(-ranks, kind="stable")[:limit]
        return [
            {
                "record_index": int(index),
                "rank": _json_safe(ranks[index]),
                "values": _json_safe(records[index]),
            }
            for index in indices
        ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def render_triton_output(
    name: str,
    tensor: np.ndarray,
    contract: RawRecordContract | None,
    *,
    max_records: int = 20,
) -> dict[str, Any]:
    """Render bounded, JSON-safe evidence for one named Triton output."""
    if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 0:
        raise ValueError(f"max_records must be a non-negative integer, got {max_records!r}")
    array = np.asarray(tensor)
    rendered: dict[str, Any] = {
        "name": name,
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
    }
    if contract is not None:
        records, _ranks = contract.records_and_ranks(array)
        rendered.update(
            {
                "record_count": int(records.shape[0]),
                "ranking": {
                    "field": contract.ranking.field,
                    "reduction": contract.ranking.reduction,
                },
                "top_records": contract.top_records(array, limit=max_records),
            }
        )
    elif array.size <= max_records:
        rendered["values"] = _json_safe(array)
    else:
        rendered["value_count"] = int(array.size)
    return rendered


def load_raw_record_contract(model_yaml: str | Path, output_name: str) -> RawRecordContract:
    path = Path(model_yaml).expanduser().resolve()
    return _load_raw_record_contract(path, output_name)


@cache
def _load_raw_record_contract(path: Path, output_name: str) -> RawRecordContract:
    try:
        document = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ValueError(f"{path}: cannot read model manifest: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc

    document = _mapping(document, path, "model document")
    io = _mapping(document.get("io"), path, "io")
    outputs = io.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"{path}: io.outputs must be a list")  # noqa: TRY004
    output = next((item for item in outputs if isinstance(item, Mapping) and item.get("name") == output_name), None)
    if output is None:
        raise ValueError(f"{path}: io.outputs has no output named {output_name!r}")
    output = _mapping(output, path, f"io.outputs[{output_name!r}]")
    layout = output.get("record_layout")
    if not isinstance(layout, Mapping):
        raise ValueError(f"{path}: record_layout must be a mapping")  # noqa: TRY004

    batch_axis = _axis(layout, path, "batch_axis")
    record_axis = _axis(layout, path, "record_axis")
    field_axis = _axis(layout, path, "field_axis")
    if len({batch_axis, record_axis, field_axis}) != 3:
        raise ValueError(f"{path}: record_layout axes must be distinct")

    fields_value = layout.get("fields")
    if not isinstance(fields_value, list) or not fields_value:
        raise ValueError(f"{path}: record_layout.fields must be a non-empty list")
    fields: list[FieldSelector] = []
    names: set[str] = set()
    for index, raw_field in enumerate(fields_value):
        member = f"record_layout.fields[{index}]"
        field = _mapping(raw_field, path, member)
        name = field.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: {member}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"{path}: {member}.name duplicates field {name!r}")
        names.add(name)
        has_index = "index" in field
        has_slice = "slice" in field
        if has_index == has_slice:
            raise ValueError(f"{path}: {member} must contain exactly one of index or slice")
        if has_index:
            value = field["index"]
            if not _is_int(value) or value < 0:
                raise ValueError(f"{path}: {member}.index must be a non-negative integer")
            fields.append(FieldSelector(name=name, index=value))
        else:
            value = field["slice"]
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(_is_int(item) for item in value)
                or value[0] < 0
                or value[0] >= value[1]
            ):
                raise ValueError(f"{path}: {member}.slice must be a non-empty [start, stop) pair")
            fields.append(FieldSelector(name=name, bounds=(value[0], value[1])))

    ranking_value = layout.get("ranking")
    ranking = _mapping(ranking_value, path, "record_layout.ranking")
    ranking_field = ranking.get("field")
    reduction = ranking.get("reduction")
    if not isinstance(ranking_field, str) or ranking_field not in names:
        raise ValueError(f"{path}: record_layout.ranking.field must name a declared field")
    if reduction not in {"identity", "max"}:
        raise ValueError(f"{path}: record_layout.ranking.reduction must be identity or max")
    selected = next(field for field in fields if field.name == ranking_field)
    if reduction == "identity" and not selected.is_index:
        raise ValueError(f"{path}: record_layout.ranking identity requires an index field")
    if reduction == "max" and not selected.is_slice:
        raise ValueError(f"{path}: record_layout.ranking max requires a slice field")

    return RawRecordContract(
        output_name=output_name,
        batch_axis=batch_axis,
        record_axis=record_axis,
        field_axis=field_axis,
        fields=tuple(fields),
        ranking=Ranking(field=ranking_field, reduction=reduction),
    )


def _mapping(value: Any, path: Path, member: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: {member} must be a mapping")  # noqa: TRY004
    return value


def _axis(layout: Mapping[str, Any], path: Path, name: str) -> int:
    value = layout.get(name)
    if not _is_int(value) or value < 0:
        raise ValueError(f"{path}: record_layout.{name} must be a non-negative integer")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
