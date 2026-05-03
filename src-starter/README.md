<<<<<<< HEAD
# Getting started

Place your CSL implementation in this directory. A good reading order is:

1. `examples/tutorials/gemv-00-basic-syntax` — quick CSL syntax overview in about 100 lines
2. `examples/tutorials/gemv-05-multiple-pes` — multi-PE layouts and memcpy setup
3. `examples/tutorials/gemv-06-routes-1` through `gemv-08-routes-3` — routing basics
4. `examples/tutorials/topic-11-collectives` — scatter, broadcast, gather, and reduce
5. `examples/tutorials/topic-05-sentinels` — end-of-stream markers
6. `examples/benchmarks/gemv-collectives_2d` — closest template for a $P \times P$ PE grid
7. `examples/benchmarks/histogram-torus` — useful reference for wavelet packing and termination detection

Common issues to watch for:

* **Task IDs and colors share the same 32-slot namespace.** Several slots are already reserved by memcpy and collectives, so check the color-map comment in `topic-11-collectives/layout.csl` before choosing IDs.
* **Each PE only has about 48 KB of SRAM.** Large per-PE arrays may pass parsing but fail during linking.
* **SdkRuntime builds require `--memcpy --channels=1`.** Without these flags, `cslc` defaults to the older CSELFRunner path, which can reject the compile.
* **Collective callbacks run on every PE in the collective group.** They do not only fire on the root or destination PE, so the state machine needs to handle all participants correctly.
=======
# Start here

Your CSL goes in this directory. Suggested starting reading order:

1. `examples/tutorials/gemv-00-basic-syntax` — CSL in ~100 lines
2. `examples/tutorials/gemv-05-multiple-pes` — multi-PE layout + memcpy
3. `examples/tutorials/gemv-06-routes-1` through `gemv-08-routes-3` — routing
4. `examples/tutorials/topic-11-collectives` — scatter / broadcast / gather / reduce
5. `examples/tutorials/topic-05-sentinels` — marking end-of-stream
6. `examples/benchmarks/gemv-collectives_2d` — best template for a P×P grid
7. `examples/benchmarks/histogram-torus` — wavelet bit-packing and
   termination detection, if you end up needing those

Common pitfalls (not a full list, just the ones we've seen):

- **Task IDs live in a 32-slot namespace shared with colors.** Many IDs are
  reserved by memcpy and collectives. Check the color-map comment in
  `topic-11-collectives/layout.csl` before picking task IDs.
- **PE SRAM is ~48 KB.** Oversized per-PE buffers fail at link, not compile.
- **`cslc` invocations need `--memcpy --channels=1`** for SdkRuntime. The
  default is the deprecated CSELFRunner and will reject your compile.
- **Collective callbacks fire on every PE in the group**, not just the
  destination. Structure your task state machine accordingly.
>>>>>>> b2e48fe8c7977ffe052ad94f35b8722255977279
