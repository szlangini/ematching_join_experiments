# GPU benchmark (CUDA)

Same pattern as the CPU benchmark in `../src` — `f(α, g(α))` compiled to
`Q(root, α) ← R_f(root, α, x), R_g(x, α)` — but **one binary runs six matchers
over identical data and times them all on the same machine**, so CPU-vs-GPU is a
fair comparison. Writes everything to `results_gpu.csv`:

| matcher | where | work |
|---|---|---|
| `cpu_backtracking` | CPU | nested loop, structural test then equality — Θ(N²) (the baseline) |
| `cpu_hash_join` | CPU | `std::unordered_set` index on the join key `(x, α)`, then probe — Θ(N) |
| `cpu_lftj` | CPU | **Leapfrog Triejoin** (worst-case-optimal); sorts to tries, leapfrog per variable — Θ(N log N) |
| `gpu_nested_loop` | GPU | one thread per f-row, inner loop over all g-rows — Θ(N²) |
| `gpu_build` + `gpu_probe` | GPU | open-addressing table on `(x, α)`, then probe — Θ(N) |
| `gpu_lftj` | GPU | **Leapfrog Triejoin**, parallelized over the variable `α` — Θ(N) given trie-sorted input |

Takeaways it's built to show: (1) the asymptotic gap is **algorithmic** — the hash
join *and* the leapfrog triejoin pull away from the nested loop as N grows, on
*both* CPU and GPU; (2) on the same (linear) algorithm, the GPU shows a hardware
speedup. Parallelism lowers the constant, not the exponent — the paper's point.

**On LFTJ here:** for this single *acyclic* 2-relation query, hash join and LFTJ
are both Θ(N) (LFTJ pays an extra `log N` to sort if input isn't pre-indexed).
LFTJ's distinctive *worst-case-optimal* advantage appears on **cyclic / multi-way**
queries (e.g. triangles), where binary-join plans blow up — that's the natural
follow-up query to add. It's included here as the genuine "generic join" the
paper points at, and to confirm it agrees with the others.

The CPU matchers are C++ reimplementations of the same algorithms; the canonical
versions live in the Rust crate `../src`. One binary = same-machine, fair timing.

## Verification & credibility

The benchmark is checked against an **independent analytical ground truth**, not
against itself: the answer is known in closed form to be exactly
`{(I_F, k) : k = 1..N}` (N matches). On every N, each of the six matchers has its
**entire output set** (not just the count) verified against this truth, and the
program **aborts** on any mismatch. Six independent implementations (3 algorithms
× CPU/GPU) all reproducing an independent closed-form answer is strong evidence of
correctness — much stronger than implementations merely agreeing with each other.

What was validated where:
- **CPU matchers, incl. `cpu_lftj`** — compiled and run with a host C++ compiler
  (`-Wall -Wextra`, clean) against the ground truth, **including on shuffled
  input** (so LFTJ isn't accidentally relying on generation order).
- **`gpu_lftj` per-thread logic** — validated as a host simulation against the
  ground truth (the CUDA execution mechanics — thread indexing, `atomicAdd` — are
  standard but unrun; see below).
- **Asymptotics are quantified** — the Colab notebook fits `time ∝ N^slope` per
  matcher and reports the measured exponent + R²: O(N) matchers land near slope 1,
  O(N²) near slope 2, turning the qualitative claim into a number.

⚠️ **Not yet compiled on a GPU.** This was authored on a machine with no GPU and
no `nvcc`, so the CUDA execution paths have not run on hardware. They use only the
stable CUDA Runtime API and self-verify on every run. The first clean Colab run is
the real end-to-end test — if `nvcc` flags anything, send the error and I'll fix it.

## Build & run

```sh
# from the repo root
nvcc -O3 -std=c++14 -arch=native -o ematching_gpu cuda/ematching_gpu.cu
./ematching_gpu                  # default: O(N) matchers to N=2^21, O(N^2) capped at N=2^15
./ematching_gpu 2097152 65536    # custom: max-N for O(N) matchers, cap for O(N^2) matchers
```

`-arch=native` needs nvcc ≥ 11.5; otherwise pass your GPU's compute capability
explicitly, e.g. `-arch=sm_70` (V100), `sm_75` (T4), `sm_80` (A100),
`sm_86` (RTX 30xx), `sm_89` (RTX 40xx / L4), `sm_90` (H100).

The O(N²) matchers are capped (default N=2^15) because the quadratic CPU baseline
gets slow; the O(N) matchers run the full sweep. Output: a table to stdout plus
`results_gpu.csv` with columns
`n, matches, cpu_backtracking_secs, cpu_hashjoin_secs, cpu_lftj_secs, gpu_nested_secs, gpu_hashjoin_kernel_secs, gpu_hashjoin_total_secs, gpu_lftj_kernel_secs, gpu_lftj_total_secs`
(the two O(N²) columns are empty past the cap). The GPU `*_kernel_secs` are
compute only (data resident); `*_total_secs` add the H2D/D2H PCIe transfers — for
operations this cheap, transfer often dominates, itself a useful GPU lesson.

## Easiest way to run it: Google Colab (free GPU)

No local GPU needed. The repo root has **`colab_ematching_gpu.ipynb`** — a
ready-to-run notebook that compiles this file, runs all six matchers, fits the
empirical scaling exponents (the verification step), and produces a two-panel
diagram (log-log scaling + measured-exponent-vs-theory). Upload it to
[Colab](https://colab.research.google.com), pick a T4 GPU runtime, and Run all.

## How it's "fed from the Arrow batches"

The kernels consume plain `int64_t` device arrays, one per column. That is exactly
what an Arrow `Int64Array` already *is* on the host — a contiguous, non-null
`i64` buffer — so the host→device feed is a straight `cudaMemcpy` of the Arrow
column buffer, with zero conversion. This standalone program regenerates the data
itself for portability, but in the Coln pipeline you would hand it the existing
Arrow batches built by `../src/arrow_store.rs`:

```rust
use ematching_bench::arrow_store::col_i64; // (expose the module via lib.rs)

let rf = build_rf(&rf_rows);
let f_id:  &[i64] = col_i64(&rf, 0).values(); // zero-copy view of the Arrow buffer
let f_a1:  &[i64] = col_i64(&rf, 1).values();
let f_a2:  &[i64] = col_i64(&rf, 2).values();
// cudaMemcpy(d_f_id, f_id.as_ptr(), f_id.len() * 8, cudaMemcpyHostToDevice);
// ... launch gpu_build / gpu_probe over the device pointers ...
```

So the column layout the kernels expect (`id, arg1, arg2`) is identical to the
Arrow schema, and the GPU path slots straight onto the columnar representation.

## Next steps

* **A cyclic / multi-way query.** The leapfrog triejoin is implemented
  (`cpu_lftj`, `gpu_lftj`), but on this acyclic 2-relation query it has no
  asymptotic edge over the hash join. Its worst-case-optimal advantage shows on
  patterns whose conjunctive query is cyclic (e.g. a triangle `R(a,b),S(b,c),T(c,a)`),
  where any binary-join plan is asymptotically beaten. Adding such a query is the
  way to actually demonstrate WCOJ rather than just run it.
* **Real e-graph benchmarks.** Drive the matchers with the actual egg/egglog
  suites instead of the synthetic worst case.
* **Wire into the Rust binary.** I can add a feature-gated FFI path (a `build.rs`
  that compiles this `.cu` via `nvcc` and an `extern "C"` entry point) so
  `cargo run --release --features cuda` feeds the GPU directly from the Arrow
  batches and compares against the CPU matchers in one process. Left out for now
  because that build glue is the one part I can't verify without a GPU here.
