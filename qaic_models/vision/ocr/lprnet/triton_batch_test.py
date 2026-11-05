"""Real Triton dynamic-batching test for ``lprnet_qpc``.

Preprocesses a set of plate images from the LPRNet repo's ``data/test``
directory (the filename stem is the ground-truth plate, e.g. ``京PL3N67.jpg``)
and sends them **concurrently** to the running Triton ``lprnet_qpc`` model so
the server's dynamic batching coalesces them into real QPC batches (up to the
QPC's fixed batch of 8 per instance, 4 instances).

Because the test images are all *distinct* plates, this catches the failure
modes an all-identical-plate smoke test cannot:

- **row mixing / wrong slicing** — if a real row's logits were decoded from the
  wrong batch slot, the plate would not match its own filename;
- **padding corruption** — if empty slots were zero-padded (instead of
  tile-filled with real inputs), the real rows' logits shift and the decode
  gains spurious characters (the ``京PBZL3N67``-style corruption).

Two metrics are reported:

1. **Triton vs filename** — the decoded plate vs the ground-truth plate in the
   image name. High accuracy confirms the model is being served correctly.
2. **Triton vs reference** — the decoded plate vs a clean batch-1 inference of
   the *same* image on the latency QPC (no padding, no other rows). This is the
   airtight check that the batching/padding path does not corrupt any row: the
   two must agree for essentially every image.

Run against a Triton server started **after** the repo was (re)generated with
the current ``triton_model.py`` (see README "Triton serving"):

    python3 triton_batch_test.py --num 100 --concurrency 32
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.ocr.lprnet.postprocess import OUTPUT_NAME, decode_ctc
from qaic_models.vision.ocr.lprnet.preprocess import INPUT_NAME, preprocess_plate_image


def _triton_infer(client, arr: np.ndarray) -> tuple[str, float]:
    """Send one preprocessed plate [1,3,24,94] to lprnet_qpc; return (plate, conf)."""
    infer_input = InferInput(INPUT_NAME, arr.shape, "FP32")
    infer_input.set_data_from_numpy(arr)
    result = client.infer(
        model_name="lprnet_qpc",
        inputs=[infer_input],
        outputs=[
            InferRequestedOutput("plate"),
            InferRequestedOutput("confidence"),
            InferRequestedOutput("num_chars"),
        ],
    )
    plate = result.as_numpy("plate")[0].decode("utf-8")
    conf = float(result.as_numpy("confidence")[0])
    return plate, conf


def _reference_plate(session, arr: np.ndarray) -> str:
    """Clean batch-1 decode of one plate on the latency QPC (no padding)."""
    logits = session.run({INPUT_NAME: np.ascontiguousarray(arr)})[OUTPUT_NAME]
    return decode_ctc(logits)["plate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="build/lprnet/repo/data/test", help="LPRNet data/test directory")
    parser.add_argument("--num", type=int, default=100, help="number of images to test (0 = all)")
    parser.add_argument("--concurrency", type=int, default=32, help="concurrent Triton requests (drives batching)")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument(
        "--reference-qpc",
        default="build/lprnet/qpc_latency/programqpc.bin",
        help="batch-1 QPC used as the clean per-image reference (empty string to skip)",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test directory not found: {test_dir}")
    images = sorted(test_dir.glob("*.jpg"))
    if args.num:
        images = images[: args.num]
    if not images:
        raise SystemExit(f"no .jpg images found in {test_dir}")
    print(f"testing {len(images)} images from {test_dir}, concurrency {args.concurrency}")

    # Preprocess once; reuse the tensors for both the Triton and reference paths.
    preprocessed = [(p, p.stem, preprocess_plate_image(str(p))) for p in images]

    # Reference pass (sequential, batch 1): the unambiguous per-image decode.
    # Optional — the filename ground truth is the primary check; the reference
    # is a second opinion that the batching path does not corrupt any row.
    ref_plates: list[str] = []
    if args.reference_qpc:
        ref_path = Path(args.reference_qpc)
        if not ref_path.is_file():
            print(f"note: reference QPC not found at {ref_path}; skipping reference pass")
        else:
            import qaic  # noqa: PLC0415  # QAIC SDK is only available in Cloud AI runtime containers.

            print(f"computing batch-1 reference on {ref_path} ...")
            ref_session = qaic.Session(model_path=str(ref_path), num_activations=1)
            try:
                for _path, _stem, arr in preprocessed:
                    ref_plates.append(_reference_plate(ref_session, arr))
            finally:
                # Tear the reference session down so it doesn't keep an SoC busy
                # for the rest of the run (there are multiple SoCs, so it does
                # not starve the throughput instances — it just wastes one).
                ref_session.reset()

    # Triton pass (concurrent): dynamic batching coalesces the in-flight requests.
    print(f"sending {len(preprocessed)} concurrent requests to {args.url} ...")
    # tritonclient's HTTP client is gevent-backed and NOT thread-safe: one shared
    # client across N OS threads deadlocks on its shared gevent pool (the server
    # processes the requests, but the client never returns them). Give each worker
    # thread its own client to keep real concurrency without the deadlock.
    _tls = threading.local()

    def _client() -> triton_http.InferenceServerClient:
        c = getattr(_tls, "client", None)
        if c is None:
            c = triton_http.InferenceServerClient(url=args.url)
            _tls.client = c
        return c

    def _one(item):
        _path, _stem, arr = item
        return _triton_infer(_client(), arr)

    triton_results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for (_path, stem, _arr), (plate, conf) in zip(preprocessed, pool.map(_one, preprocessed)):
            triton_results.append((stem, plate, conf))

    # Report.
    correct_filename = sum(1 for stem, plate, _c in triton_results if plate == stem)
    acc_filename = 100.0 * correct_filename / len(triton_results)
    print(f"\nTriton vs filename : {correct_filename}/{len(triton_results)} = {acc_filename:.2f}%")

    if ref_plates:
        correct_ref = sum(1 for (stem, plate, _c), ref in zip(triton_results, ref_plates) if plate == ref)
        acc_ref = 100.0 * correct_ref / len(triton_results)
        print(f"Triton vs reference: {correct_ref}/{len(triton_results)} = {acc_ref:.2f}%")

    ref_by_idx = dict(enumerate(ref_plates)) if ref_plates else {}
    mismatches = [
        (stem, plate, conf, ref_by_idx.get(i, "?"))
        for i, (stem, plate, conf) in enumerate(triton_results)
        if plate != stem
    ]
    if mismatches:
        print(f"\n{len(mismatches)} filename mismatches (first 50):")
        for stem, plate, conf, ref in mismatches[:50]:
            print(f"  gt={stem!r}  triton={plate!r}  ref={ref!r}  conf={conf:.4f}")


if __name__ == "__main__":
    main()
