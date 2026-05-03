# Top-K k-NN on a $P \times P$ Wafer — Design Memo

**Started:** 2026-05-02
**Author:** Angad Singh

## 1. Routing topology

```
        col 0   col 1   col 2   col 3
row 0   (0,0)──q──▶(1,0)──q──▶(2,0)──q──▶(3,0)     stages 0+1: q broadcast
          │         │         │         │           through mpi_x, then mpi_y
row 1   (0,1)     (1,1)     (2,1)     (3,1)
          │         │         │         │           stage 3: mpi_x.gather
row 2   (0,2)     (1,2)     (2,2)     (3,2)         across each row into col 0
          │         │         │         │
row 3   (0,3)     (1,3)     (2,3)     (3,3)         stage 5: mpi_y.gather
                                                     up col 0 into PE(0,0)
```

Routing is handled through `<collectives_2d>` rather than custom fabric code. Row collectives use `c2d_x_color_0/1` on slots `0` and `1`, while column collectives use `c2d_y_color_0/1` on slots `4` and `5`. Local task IDs `10–13` are reserved for the c2d collective entrypoints, and slot `14` drives the application state machine.

Each gather transmits `2*K` `u32` wavelets per PE. The payload is stored as an interleaved packed array:

```
bitcast(f32, dist_0), idx_0, bitcast(f32, dist_1), idx_1, ...
```

This keeps each `(distance, index)` pair synchronized on a single color and avoids splitting metadata across separate streams.

## 2. Local top-K algorithm

Each PE computes its local candidates using a bounded max-heap of size `K`. The ordering is lexicographic on `(dist, idx)`, so smaller distance wins, and smaller original row index breaks ties.

The same heap implementation supports all tested values of `K`. For `K = 1`, the heap effectively becomes a single running best candidate. For small `K`, the heap fills quickly, and most later rows are rejected with one comparison against `heap[0]`. For the large case, where `K = rows_per_pe = 256`, every local row is accepted and insertion costs are bounded by `O(log K) = 8` sift levels.

Distance computation is vectorized in column-major order across the rows local to one PE:

```
for j in 0..d_dim:
    diff[:]        = D[:, j] - q[j]
    diff[:]        = diff * diff
    local_dist[:] += diff
```

This gives three vector operations per dimension over `rows_per_pe` elements, so the approximate local compute cost is

```
3 * d_dim * rows_per_pe
```

For the baseline case with `d_dim = 32` and `rows_per_pe = 128`, this is about `12K` element-cycles per PE. That dominates heap maintenance, which is only on the order of a few hundred cycles. In uneven sharding cases, PEs with fewer than `rows_per_pe` valid rows insert sentinel candidates `(+inf, 0xFFFFFFFF)`, which always lose under the lexicographic comparator.

## 3. Fabric bandwidth accounting

During each row gather, every non-root PE sends `2K` wavelets toward column `0`. The busiest row edge, entering column `0`, carries

```
(P - 1) * 2K
```

wavelets. This is `96` wavelets in the baseline case and `1024` wavelets in the `k_large` case. The column gather then contributes the same worst-case load on the busiest column-0 link.

So the total worst-edge traffic is approximately

```
4K(P - 1)
```

which is about `192` wavelet crossings for the baseline. The kernel is therefore compute-bound: the roughly `12K` local distance-computation cycles are much larger than the roughly `200` reduction cycles per PE.

## 4. Tie-break correctness

The comparator is defined as

```
(d_a, i_a) < (d_b, i_b)
```

if and only if

```
d_a < d_b
```

or

```
d_a == d_b and i_a < i_b.
```

Since squared L2 distances over finite `f32` inputs are finite and nonnegative, this defines a deterministic total order over `(f32, u32)` candidates. Top-K selection under a total order depends only on the candidate multiset, not on the order in which candidates are merged.

Thus, for any candidate groups `A`, `B`, and `C`,

```
select_K(merge(A, B), C) = select_K(A, merge(B, C)).
```

Therefore the final output depends only on the global set of `P^2 * K` candidates and the comparator, not on PE finish order or wavelet arrival timing. This also handles `all_equal` and `duplicates` cases correctly, because equal distances are resolved deterministically by the original row index.

## 5. SRAM budget

Worst case: `k_large`, with `P = 2`, `d = 16`, and `rows_per_pe = K = 256`.

| Buffer                                         |               Size |
| ---------------------------------------------- | -----------------: |
| `D` shard                                      |              16 KB |
| `local_dist + diff`                            |               2 KB |
| Heap, storing `dist + idx`                     |               2 KB |
| `packed_topk + recv_buf`, storing `2KP` values |               6 KB |
| `result_d + result_i`                          |               2 KB |
| `q + code + stack + collectives runtime`       |             ~15 KB |
| **Total**                                      | **~43 KB / 48 KB** |

The worst case is tight but still fits within the 48 KB PE SRAM limit. The baseline case is much smaller, at roughly 22 KB.

## 6. Improvements with more time

1. **Replace flat gathers with a streaming distributed merge.**
   The current design gathers all local top-K candidates and performs software selection afterward. A hand-rolled two-pointer merge along each row and column chain could discard losing candidates at every hop, keeping only `K` candidates live instead of forwarding all `2K` pairs. This would reduce reduction traffic and save roughly 100 cycles in the baseline reduction path.

2. **Specialize the `K = 1` path.**
   The current heap logic works for `K = 1`, but it is more general than necessary. A specialized path could use a single-register running argmin and scalar collective reduction instead of packing and gathering `2K` values. That would save heap overhead and reduce communication cost significantly in the `k_eq_1` case.
