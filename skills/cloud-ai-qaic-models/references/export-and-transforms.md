# Export and Transforms

Load this reference for export, ONNX surgery, compilation, or generated-output
validation.

## Export contract

`export.py` accepts `--work-dir`, resolves it absolutely, writes every output
there, and uses explicit input/output names and artifact filenames. Immediately
inspect the exported graph's output count, names, shapes, and dtypes. Confirm
that training, debug, intermediate, and auxiliary outputs did not leak into the
deployable artifact.

Generate CPU golden outputs from the final deployable artifact, then compile
that artifact with `qaic_models/base/compile.py` helpers where possible. Keep
heavy export-only imports out of runtime paths and force CPU-only dependency
resolution in export extras.

## Transform ownership

`qaic_models/base/onnx_transforms.py` and `base/pytorch_transforms.py` own
interfaces, pipeline execution, generic passes, and broadly reusable
primitives. A recipe's `onnx_transforms.py` owns model-specific post-export
graph surgery; `pytorch_transforms.py` owns model-specific pre-export patches.
`export.py` imports these modules and declares their explicit order. Promote a
transform only after a second canonical consumer proves the reuse boundary.
Never embed graph-rewrite implementation directly in `export.py`.

## Completion checks

The final graph must match established model facts and golden behavior. Keep
compile flags, node-selection/precision specifications, and artifact layouts
documented as inputs or metadata. Do not store ONNX, QPC, raw, or other model
binaries in the recipe package.
