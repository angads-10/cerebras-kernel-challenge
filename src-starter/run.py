#!/usr/bin/env cs_python
"""Host-side runner for the Top-K k-NN wafer kernel.

Loads compile-time parameters from <name>/out.json, creates the selected test
case from reference.py, copies the sharded database and query vector onto the
wafer, launches the kernel, reads back the global top-K result, checks it
against the NumPy reference implementation, and prints PASS: <case> when the
output matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


# Try to import the canonical reference implementation first. This works in the
# normal grader setup where the parent directory is visible on sys.path. If that
# import fails, use the local fallback definitions below, which are needed in the
# docker/SIF environment where only the current directory is bind-mounted.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from reference import (  # type: ignore
        topk_reference,
        make_baseline,
        make_k_eq_1,
        make_k_large,
        make_uneven,
        make_all_equal,
        make_duplicates,
    )
except ImportError:
    # Fallback copy of reference.py.
    def topk_reference(D, q, K, squared=True):
        diff = D - q[None, :]
        d2 = np.einsum("nd,nd->n", diff, diff).astype(np.float32)
        idx_all = np.arange(D.shape[0], dtype=np.int32)
        order = np.lexsort((idx_all, d2))
        top_order = order[:K]
        indices = idx_all[top_order].astype(np.int32)
        distances = d2[top_order] if squared else np.sqrt(d2[top_order])
        return indices, distances.astype(np.float32)

    def make_baseline(seed=0):
        rng = np.random.default_rng(seed)
        N, d = 2048, 32
        D = rng.standard_normal((N, d), dtype=np.float32)
        q = rng.standard_normal(d, dtype=np.float32)
        return {"name": "baseline", "D": D, "q": q, "K": 16, "P": 4}

    def make_k_eq_1(seed=1):
        rng = np.random.default_rng(seed)
        N, d = 1024, 32
        D = rng.standard_normal((N, d), dtype=np.float32)
        q = rng.standard_normal(d, dtype=np.float32)
        return {"name": "k=1", "D": D, "q": q, "K": 1, "P": 2}

    def make_k_large(seed=2):
        rng = np.random.default_rng(seed)
        N, d = 1024, 16
        D = rng.standard_normal((N, d), dtype=np.float32)
        q = rng.standard_normal(d, dtype=np.float32)
        return {"name": "k=256", "D": D, "q": q, "K": 256, "P": 2}

    def make_uneven(seed=3):
        rng = np.random.default_rng(seed)
        N, d = 1009, 32
        D = rng.standard_normal((N, d), dtype=np.float32)
        q = rng.standard_normal(d, dtype=np.float32)
        return {"name": "uneven", "D": D, "q": q, "K": 16, "P": 4}

    def make_all_equal():
        N, d = 1024, 16
        D = np.ones((N, d), dtype=np.float32) * 0.5
        q = np.zeros(d, dtype=np.float32)
        return {"name": "all-equal", "D": D, "q": q, "K": 16, "P": 2}

    def make_duplicates(seed=5):
        rng = np.random.default_rng(seed)
        N, d = 1024, 16
        D = rng.standard_normal((N, d), dtype=np.float32)
        D[500] = D[100]
        D[800] = D[250]
        q = D[100] + np.float32(1e-3) * rng.standard_normal(d, dtype=np.float32)
        return {"name": "duplicates", "D": D, "q": q, "K": 8, "P": 2}


CASE_MAKERS = {
    "baseline":   make_baseline,
    "k_eq_1":     make_k_eq_1,
    "k_large":    make_k_large,
    "uneven":     make_uneven,
    "all_equal":  make_all_equal,
    "duplicates": make_duplicates,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compiled output directory")
    parser.add_argument("--case", required=True, choices=list(CASE_MAKERS))
    parser.add_argument("--cmaddr", help="optional IP:port for the CS system")
    args = parser.parse_args()

    # Read compile-time kernel parameters.
    with open(f"{args.name}/out.json", encoding="utf-8") as f:
        params = json.load(f)["params"]

    P           = int(params["P"])
    d_dim       = int(params["d_dim"])
    rows_per_pe = int(params["rows_per_pe"])
    K           = int(params["K"])

    # Build the requested test case and compute the reference answer.
    case = CASE_MAKERS[args.case]()
    D, q = case["D"], case["q"]
    N = D.shape[0]

    assert case["K"] == K, f"K mismatch: case={case['K']} compile={K}"
    assert case["P"] == P, f"P mismatch: case={case['P']} compile={P}"
    assert q.shape[0] == d_dim, f"d mismatch: case={q.shape[0]} compile={d_dim}"

    idx_oracle, dist_oracle = topk_reference(D, q, K)

    # Partition D across the P×P PE grid. Each PE stores rows_per_pe rows,
    # with column-major layout inside the PE.
    PE_total = P * P
    padded_rows = PE_total * rows_per_pe
    assert padded_rows >= N, f"rows_per_pe*P^2 = {padded_rows} < N = {N}"

    D_padded = np.zeros((padded_rows, d_dim), dtype=np.float32)
    D_padded[:N] = D

    # pe_id = py * P + px, with local storage shaped as [local_row, dim].
    D_per_pe = D_padded.reshape(PE_total, rows_per_pe, d_dim)

    # Convert each PE shard to column-major order: D[col * rows_per_pe + row].
    D_colmajor = np.ascontiguousarray(D_per_pe.transpose(0, 2, 1))

    # Shape for memcpy_h2d: height=P maps to py, width=P maps to px.
    D_h2d = D_colmajor.reshape(P, P, rows_per_pe * d_dim).astype(np.float32)

    # Track how many real rows each PE owns before padding.
    valid_rows = np.zeros((P, P), dtype=np.uint32)
    for py in range(P):
        for px in range(P):
            start = (py * P + px) * rows_per_pe
            valid_rows[py, px] = max(0, min(rows_per_pe, N - start))

    # Initialize runtime and resolve exported symbols.
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    sym_D          = runner.get_id("D")
    sym_q          = runner.get_id("q")
    sym_valid_rows = runner.get_id("valid_rows")
    sym_result_d   = runner.get_id("result_d")
    sym_result_i   = runner.get_id("result_i")

    runner.load()
    runner.run()

    dt = MemcpyDataType.MEMCPY_32BIT
    order = MemcpyOrder.ROW_MAJOR

    # Copy database shards to every PE.
    runner.memcpy_h2d(
        sym_D,
        D_h2d.ravel(),
        0,
        0,
        P,
        P,
        rows_per_pe * d_dim,
        streaming=False,
        data_type=dt,
        order=order,
        nonblock=False,
    )

    # Copy one valid-row count per PE.
    runner.memcpy_h2d(
        sym_valid_rows,
        valid_rows.ravel(),
        0,
        0,
        P,
        P,
        1,
        streaming=False,
        data_type=dt,
        order=order,
        nonblock=False,
    )

    # Copy q only to PE(0,0). The kernel broadcasts q over the fabric.
    runner.memcpy_h2d(
        sym_q,
        q.astype(np.float32),
        0,
        0,
        1,
        1,
        d_dim,
        streaming=False,
        data_type=dt,
        order=order,
        nonblock=False,
    )

    # Launch the device kernel.
    runner.launch("main", nonblock=False)

    # Fetch the final top-K result from PE(0,0).
    result_d = np.zeros(K, dtype=np.float32)
    result_i = np.zeros(K, dtype=np.uint32)

    runner.memcpy_d2h(
        result_d,
        sym_result_d,
        0,
        0,
        1,
        1,
        K,
        streaming=False,
        data_type=dt,
        order=order,
        nonblock=False,
    )

    runner.memcpy_d2h(
        result_i,
        sym_result_i,
        0,
        0,
        1,
        1,
        K,
        streaming=False,
        data_type=dt,
        order=order,
        nonblock=False,
    )

    runner.stop()

    # Validate indices exactly and distances within float tolerance.
    result_i_signed = result_i.astype(np.int32)
    ok_idx = np.array_equal(result_i_signed, idx_oracle)
    ok_dist = np.allclose(result_d, dist_oracle, atol=1e-3, rtol=1e-3)

    if not ok_idx:
        print("FAIL: indices mismatch", file=sys.stderr)
        print(f"  ours[:20]   = {result_i_signed[:20].tolist()}", file=sys.stderr)
        print(f"  oracle[:20] = {idx_oracle[:20].tolist()}", file=sys.stderr)

    if not ok_dist:
        print("FAIL: distances mismatch", file=sys.stderr)
        print(f"  ours[:20]   = {result_d[:20].tolist()}", file=sys.stderr)
        print(f"  oracle[:20] = {dist_oracle[:20].tolist()}", file=sys.stderr)

    if ok_idx and ok_dist:
        print(f"PASS: {args.case}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())