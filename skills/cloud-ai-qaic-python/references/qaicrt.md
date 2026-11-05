# qaicrt Python Runtime Reference

Use this reference when a Python program needs direct control of QPC loading,
buffer mappings, execution objects, queues, or lower-level resource sharing.
Use `qaic.Session` for the normal Python inference path.

**REQUIRED SUB-SKILL:** Use `cloud-ai-env-setup` to enter a compatible SDK
container and select a Ready QID with enough free NSPs.

## Before writing the program

Work inside the SDK container. Locate the ABI-matched binding and inspect the
installed example before relying on an API:

~~~bash
QAIC_ARCH=$(uname -m)
QAIC_PYTHON_LIB="/opt/qti-aic/dev/lib/$QAIC_ARCH"
QAIC_PYTHON_API="/opt/qti-aic/dev/python"
test -d "$QAIC_PYTHON_LIB"
test -d "$QAIC_PYTHON_API"
export PYTHONPATH="$QAIC_PYTHON_LIB:$QAIC_PYTHON_API:${PYTHONPATH:-}"
python -c 'import qaicrt, QAicApi_pb2; print(qaicrt.__file__)'
python /opt/qti-aic/examples/python/QAicInferenceExample.py --help
~~~

The QPC input to `Qpc` is its directory plus an optional QPC filename, not a
bare binary path. Obtain buffer names, indices, directions, and byte sizes from
`qpc.getBufferMappings()`. The `InferenceVector` path handles raw bytes; it does
not expose tensor dtypes or shapes. For a fixed-shape QPC, obtain those from the
model I/O contract. When the QPC directory contains `ios.json`, its `map-to` and
`data-type` fields associate buffer names with dtypes. Use the descriptor-driven
path below when typed bindings or runtime-selected shapes are needed. `qaicrt`
does not perform model preprocessing or postprocessing.

## Determine QPC device requirements

Validate the QPC before assigning devices:

~~~bash
/opt/qti-aic/tools/qaic-qpc validate \
  --qpc "$QPC_DIR/$QPC_FILE"
~~~

Use the report for QPC validity, hardware compatibility, NSP requirements, and
DDR requirements. `Number of NSP required` describes the network's core
requirement; it is not the number of QIDs required.

For an MDP or tensor-sliced QPC, use the partition configuration that produced
it. Count the entries in `partitions[0].devices`; that count is the number of
QIDs to pass to the program. Some compile artifacts record the same value as
`mdp_ts_num_devices`. Preserve the device order from the partition config when
assigning repeated `--qid` arguments. If the partition config is unavailable,
do not infer a multi-device count from `qaic-qpc ls`; recover it from the
conversion/compile artifacts or use the `cloud-ai-qaic-runner` MDP guidance.

## Write the program

Create a model-specific program such as `<model>_qaicrt.py` in the model's
output directory. It should:

1. Accept the QPC directory and filename, one or more selected QIDs, input
   files, and an output directory.
2. Build a `QIDList` and `Context` for the selected device.
3. Load the QPC and print its buffer mappings.
4. Create an `InferenceVector` from exact input files, or generated input for a
   runtime-only smoke test.
5. Create `Program` and `ExecObj`, attach the vector, and check `QStatus`.
6. Save each output buffer as raw bytes for model-specific decoding or checks.

Use this complete starting point:

~~~python
#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from qaicrt import (
    BufferIoTypeEnum,
    Context,
    ExecObj,
    InferenceVector,
    Program,
    QAicProgramProperties,
    QIDList,
    Qpc,
    QStatus,
    Queue,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpc-dir", type=Path, required=True)
    parser.add_argument("--qpc-file", default="programqpc.bin")
    parser.add_argument("--qid", type=int, action="append", required=True)
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--async-run", action="store_true")
    return parser.parse_args()


def check_status(status, operation):
    if status != QStatus.QS_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")


def output_path(output_dir, mapping):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", mapping.bufferName)
    if not safe_name.strip("._-"):
        safe_name = f"buffer_{mapping.index}"
    return output_dir / f"{mapping.index}_{safe_name}.bin"


def main():
    args = parse_args()
    qpc_path = args.qpc_dir / args.qpc_file
    if not qpc_path.is_file():
        raise FileNotFoundError(f"QPC file not found: {qpc_path}")

    devices = QIDList()
    for qid in args.qid:
        devices.append(qid)
    context = Context(devices)
    try:
        qpc = Qpc(str(args.qpc_dir), args.qpc_file)
    except RuntimeError as error:
        raise RuntimeError(f"unable to load QPC: {args.qpc_dir}") from error

    mappings = qpc.getBufferMappings()
    input_mappings = [
        mapping
        for mapping in mappings
        if mapping.ioType == BufferIoTypeEnum.BUFFER_IO_TYPE_INPUT
    ]
    for mapping in mappings:
        print(mapping.index, mapping.bufferName, mapping.size, mapping.ioType)

    if args.input_file:
        if len(args.input_file) != len(input_mappings):
            raise ValueError("provide one --input-file for each QPC input buffer")
        for path, mapping in zip(args.input_file, input_mappings):
            if path.stat().st_size != mapping.size:
                raise ValueError(
                    f"{path} is {path.stat().st_size} bytes; "
                    f"{mapping.bufferName} requires {mapping.size}"
                )
        inference = InferenceVector(mappings, [str(path) for path in args.input_file])
    else:
        inference = InferenceVector(mappings)

    properties = QAicProgramProperties()
    program_qid = devices[0] if len(args.qid) == 1 else None
    if len(args.qid) > 1:
        properties.devMapping = ":".join(str(qid) for qid in args.qid)
    program = Program(context, program_qid, qpc, properties)
    check_status(program.load(), "load")
    check_status(program.activate(), "activate")
    exec_obj = ExecObj(context, program)
    try:
        check_status(exec_obj.setData(inference), "set data")

        if args.async_run:
            queue = Queue(context, devices[0])
            check_status(queue.enqueue(exec_obj), "enqueue")
            check_status(exec_obj.waitForCompletion(), "completion")
        else:
            check_status(exec_obj.run(inference), "run")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for mapping in mappings:
            if mapping.ioType != BufferIoTypeEnum.BUFFER_IO_TYPE_OUTPUT:
                continue
            output = bytes(inference.getVector()[mapping.index])
            path = output_path(args.output_dir, mapping)
            path.write_bytes(output)
            print(f"wrote {path} ({len(output)} bytes)")
    finally:
        del exec_obj
        check_status(program.deactivate(), "deactivate")


if __name__ == "__main__":
    main()
~~~

The input-file order must match the input mappings returned by the QPC. Validate
byte sizes before execution. Repeat `--qid` for a multi-device QPC; every
selected QID must be Ready and have the resources required by that QPC.
Generated input is useful only for checking that the runtime path works; use
model-valid input for correctness testing.

For a multi-QID QPC, construct the context with every selected QID and set
`QAicProgramProperties.devMapping` in the same order. Do not pass the first QID
as the `Program` device ID for a multi-QID QPC: pass `None` and use the mapping.

## Use descriptor-driven buffers for named or dynamic QPCs

Use this path instead of `InferenceVector` when the program needs named NumPy
inputs, typed outputs, runtime-selected shapes, or permitted partial buffers.
The QPC I/O descriptor is the source of truth for those contracts.

Load and parse the descriptor before allocating buffers:

~~~python
import numpy as np
import QAicApi_pb2
import qaicrt


NP_DTYPES = {
    getattr(QAicApi_pb2, "BFLOAT16_TYPE", 11): np.dtype("float16"),
    QAicApi_pb2.FLOAT_TYPE: np.dtype("float32"),
    QAicApi_pb2.FLOAT_16_TYPE: np.dtype("float16"),
    QAicApi_pb2.INT8_Q_TYPE: np.dtype("int8"),
    QAicApi_pb2.UINT8_Q_TYPE: np.dtype("uint8"),
    QAicApi_pb2.INT16_Q_TYPE: np.dtype("int16"),
    QAicApi_pb2.INT32_Q_TYPE: np.dtype("int32"),
    QAicApi_pb2.INT32_I_TYPE: np.dtype("int32"),
    QAicApi_pb2.INT64_I_TYPE: np.dtype("int64"),
    QAicApi_pb2.INT8_TYPE: np.dtype("int8"),
}

iodesc = QAicApi_pb2.IoDesc()
status, serialized_iodesc = qpc.getIoDescriptor()
check_status(status, "get I/O descriptor")
iodesc.ParseFromString(bytes(serialized_iodesc))

bindings = iodesc.selected_set.bindings
binding_index = {binding.name: binding.index for binding in bindings}
allowed_shapes = [
    [(NP_DTYPES[shape.type].itemsize, list(shape.dims)) for shape in choice.shapes]
    for choice in iodesc.allowed_shapes
]
~~~

Use the binding's exact name and index. A graph may prefix a tensor name, so do
not derive an index from a display name. Before execution, select a compatible
entry from `allowed_shapes` when it is present and set dimensions for every
provided binding. If no allowed-shape set matches, stop and report the QPC's
accepted shapes rather than padding or truncating model input arbitrarily.

Allocate buffers and update their dimensions from contiguous model-specific
arrays. `QBuffer` owns a copy of the byte string in this path:

~~~python
qbuffers = [qaicrt.QBuffer(bytes(binding.size)) for binding in bindings]
buf_dims = qaicrt.BufferDimensionsVecRef(
    [(NP_DTYPES[binding.type].itemsize, list(binding.dims)) for binding in bindings]
)


def set_binding(name, array):
    array = np.ascontiguousarray(array)
    index = binding_index[name]
    qbuffers[index] = qaicrt.QBuffer(array.tobytes())
    buf_dims[index] = (array.itemsize, list(array.shape) or [1])
~~~

For an omitted binding, use a zero-length buffer and dimensions `[0]` only when
that binding's `is_partial_buf_allowed` field permits it. Do not use that form
to omit a required input or output.

Create the program with its lifecycle and device mapping explicit. Set
`dataPathTimeoutMs` only when an individual execution can take longer than the
runtime default; it is a millisecond timeout for one device response, not a
throughput control.

~~~python
devices = qaicrt.QIDList(selected_qids)
context = qaicrt.Context(devices)
properties = qaicrt.QAicProgramProperties()
properties.dataPathTimeoutMs = data_path_timeout_ms

program_qid = selected_qids[0] if len(selected_qids) == 1 else None
if len(selected_qids) > 1:
    properties.devMapping = ":".join(str(qid) for qid in selected_qids)

program = qaicrt.Program(context, program_qid, qpc, properties)
check_status(program.load(), "load")
check_status(program.activate(), "activate")
exec_obj = qaicrt.ExecObj(context, program)
try:
    check_status(exec_obj.setData(qbuffers, buf_dims), "set data")
    queue = qaicrt.Queue(context, selected_qids[0])
    check_status(queue.enqueue(exec_obj), "enqueue")
    check_status(exec_obj.waitForCompletion(), "completion")
    status, output_qbuffers = exec_obj.getData()
    check_status(status, "get data")
finally:
    del exec_obj
    check_status(program.deactivate(), "deactivate")
~~~

Decode each requested output with the descriptor's dtype and the dimensions
submitted for that binding:

~~~python
outputs = {}
for binding in bindings:
    if binding.dir != QAicApi_pb2.BUFFER_IO_TYPE_OUTPUT:
        continue
    index = binding.index
    outputs[binding.name] = np.frombuffer(
        bytes(output_qbuffers[index]), dtype=NP_DTYPES[binding.type]
    ).reshape(buf_dims[index][1])
~~~

Map these arrays to model-specific postprocessing and correctness checks. Keep
the descriptor, program, and execution object alive until output decoding is
complete.

## Use caller-owned DMA buffers

Use the DMA path when the application already owns exported DMA-BUF handles
and the QPC's DMA I/O contract is intentional. Do not substitute a host
pointer, a device address, or a file descriptor for a different resource. The
application owns allocation, mapping, synchronization, and lifetime of these
handles.

Obtain the DMA mappings separately and create one `QBuffer` at each mapping's
index. Populate and read the buffers through the allocator or sharing API that
owns the DMA-BUF handles.

~~~python
from qaicrt import InferenceVector, QBuffer

dma_mappings = qpc.getBufferMappingsDma()
dma_handles_by_index = {...}  # caller-owned DMA-BUF handle for every mapping index
dma_buffers = [None] * len(dma_mappings)
for mapping in dma_mappings:
    dmaBufHandle = dma_handles_by_index[mapping.index]
    dma_buffers[mapping.index] = QBuffer(mapping.size, dmaBufHandle)
inference = InferenceVector(dma_mappings, dma_buffers)
exec_obj.setData(inference)
~~~

`QBuffer(bufferSize, dmaBufHandle, offset=0)` wraps an existing DMA-BUF
handle. Keep every handle valid until execution completes. Use `offset` only
for a subrange defined by the owning allocator and mapping contract. DMA output
buffers may not be directly host-readable; use the owning buffer API to
synchronize and read them rather than assuming `bytes(QBuffer)` is valid. Close
each owned handle after its final completion and readback.

For pooled DMA execution, the `InferenceSet` overload accepts a vector of
vectors of `QBuffer` objects. The outer vector must contain
`set_size * num_activations` entries, and each inner vector must match the DMA
mapping order.

## Add concurrency deliberately

Use synchronous `exec_obj.run()` for a single request. Use `Queue.enqueue()` and
`waitForCompletion()` when the caller needs asynchronous submission; do not
reuse an enqueued `ExecObj` until completion.

Use `InferenceSet` for a managed pool of execution objects or multiple
activations. Its core flow is:

1. Create `InferenceSet(context, qpc, dev_id, set_size, num_activations)`.
2. Submit an `InferenceVector` with an application ID and timeout.
3. Retrieve completion with `getCompleted()` or `getCompletedId()`.
4. Release the completed handle with `putCompleted()`.

Size `set_size` and `num_activations` for the required concurrency and available
device resources. More activations do not improve single-request latency.
Use `ProgramGroup` or `InferenceSetGroup` only when multiple programs must share
resources on one device; consult the parent Python or C++ API guidance for that
lifecycle.

For a managed inference pool, the completion handle owns the completed buffers
until `putCompleted()` returns it to the pool:

~~~python
from qaicrt import InferenceSet

inference_set = InferenceSet(
    context, qpc, devices[0], 1, 1
)
request_id = 1
check_status(inference_set.submit(inference, request_id, 0), "submit")
status, handle = inference_set.getCompletedId(request_id, 0)
check_status(status, "completion")
completed_buffers = handle.getBuffers()
# Consume or copy completed_buffers before releasing the handle.
check_status(inference_set.putCompleted(handle), "release")
~~~

For resource sharing between multiple programs on one QID, configure and
enable a `ProgramGroup` before enqueueing its `ExecObj`s, then disable it after
completion:

~~~python
from qaicrt import ProgramGroup

group = ProgramGroup(context, devices[0], "model-group", None)
status, grouped_program = group.addProgram(qpc)
check_status(status, "add program")
check_status(group.enable(), "enable group")
group_exec = ExecObj(context, grouped_program)
group_exec.setData(inference)
check_status(group.enqueue(group_exec), "enqueue")
check_status(group_exec.waitForCompletion(), "completion")
check_status(group.disable(True, 0), "disable group")
~~~

Program groups are single-device resource-sharing objects. Do not use them for
MDP or tensor-sliced QPCs; those require their full QID set independently.

## Verify the program

Run the program's `--help` and one known-input case inside the SDK container.
Confirm that:

- the selected QID was reported Ready before execution;
- every mapping has the expected direction and byte size;
- `run`, `enqueue`, and `waitForCompletion` return `QStatus.QS_SUCCESS`;
- output files are present and pass the model's correctness check.

Use the installed `qaicrt` module, the bundled example, and the QPC mappings as
the source of truth when the binding or a QPC differs from this template.
