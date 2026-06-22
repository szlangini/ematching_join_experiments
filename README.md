# ematching-bench

A standalone Rust smoke test reproducing the **core qualitative result** of
*Relational E-matching* (Yihong Zhang, Yisu Remy Wang, Max Willsey, Zachary
Tatlock, POPL 2022, [arXiv:2108.02290](https://arxiv.org/abs/2108.02290)):

> **E-matching is a relational join, and evaluating it as one beats naive
> backtracking search _asymptotically_.**

E-matching is the operation that pattern-matches a term pattern against an
e-graph; theorem provers and equality-saturation engines spend the *majority* of
their time in it. The paper's insight is that the problem is really a
conjunctive query, so the decades of database research on join algorithms apply.

This project is a smoke test for a larger effort on the **Coln** database system,
so it is written in Rust and stores its data in **Apache Arrow** record batches
(the columnar representation Coln will use).

---

## What e-matching is, briefly

An **e-graph** compactly represents a (possibly huge) set of equivalent terms.
It can be viewed as a **database of terms**:

* Each function symbol becomes a **relation** (a table).
* A function `f` of arity `k` becomes a table with `k + 1` integer columns: the
  first column is the **e-class id of the result**, and the remaining `k` columns
  are the e-class ids of the **children**.

So a row `(c, a₁, …, a_k)` in table `R_f` means "there is an `f`-application
`f(a₁, …, a_k)` and it belongs to e-class `c`."

**E-matching** a pattern means finding every substitution of the pattern's
variables that makes the pattern equal to some term already in the e-graph.

## The pattern used here

We match exactly one pattern:

```
f(α, g(α))
```

In words: *find every application of `f` whose second argument is an application
of `g`, where `f`'s first argument equals `g`'s argument.* The variable `α`
appears **twice**, which is an **equality constraint** (the two occurrences must
resolve to the same e-class).

Following the paper, this pattern compiles to a **conjunctive query**:

```
Q(root, α)  ←  R_f(root, α, x),  R_g(x, α)
```

* `root` and `α` are the output variables (the substitution we want).
* `x` is the **join variable** linking `f`'s second child to `g`'s result e-class.
* `α` appearing in both atoms is the equality constraint, now just a shared
  query variable.

This is the running example from Section 2 of the paper.

## The synthetic data — the separating worst case

The generator (`src/generator.rs`) produces, for size `N`:

```
R_f = { (I_F, k, I_G) : k in 1..=N }      columns: id, arg1 (= α), arg2 (= x)
R_g = { (I_G, k)      : k in 1..=N }      columns: id (= x), arg1 (= α)
```

`I_F` and `I_G` are two fixed, distinct e-class ids (the result classes of all
`f`- and `g`-applications respectively).

Why this is the case that *separates* the two algorithms:

* Every `g`-application lives in the **single** result e-class `I_G`. Every
  `f`-row's second child is also `I_G`. So the **structural** test — "is `f`'s
  second child a `g`-application?" — is satisfied by **all `N × N`** (f-row,
  g-row) pairs.
* Only the **equality** constraint on `α` (`f.arg1 == g.arg1`) cuts this down: it
  holds for exactly the `N` pairs where `k_f == k_g`.

So the answer set has size exactly **`N`** — it is `{ (I_F, k) : k in 1..=N }` —
but a search that checks structure first and equality last must wade through
**`N²`** candidate pairs to find them. This is deliberately the worst case from
the paper that drives the two approaches apart.

## The two matchers

Both live in `src/matchers.rs` and both read from the Arrow batches (not the raw
vectors). Both return the **set** of substitutions `(root, α)`.

### 1. Backtracking matcher — `Θ(N²)` (the prover-style baseline)

Nested iteration: for every `f`-row, scan **every** `g`-row, check the
**structural** constraint first, and check the **equality** constraint *last*.
This mirrors how a naive prover does e-matching — bind the head symbol, descend
structurally, and discover the shared-variable conflict only at the leaf. On
this data the structural test always passes, so the equality test is *reached*
`N²` times even though only `N` pairs survive.

### 2. Relational hash-join matcher — `Θ(N)`

Evaluate the conjunctive query directly. The join binds two variables at once —
`x` (structural: `R_f.arg2 = R_g.id`) and `α` (equality: `R_f.arg1 = R_g.arg1`) —
so we fold **both** into one composite key `(x, α)`, index `R_g` on it, and probe
with each `f`-row. The structural and equality constraints are resolved
*together* in a single hash lookup. Build + probe is linear in `N`.

## What the benchmark demonstrates

`src/main.rs` runs two phases:

1. **Correctness phase** (runs first, gates everything): for small `N`, it
   asserts the two matchers return the **identical set** of substitutions — not
   just the same count, the same set — and that the count is exactly `N`.
2. **Timing phase**: for a doubling range of `N`, it times both matchers (minimum
   of several runs), confirms their counts agree, prints a table, and writes
   `results.csv`.

The result is the asymptotic gap: **backtracking time grows ~`N²`, hash-join
time grows ~`N`.** The machine-independent signature is in the ratios — each time
`N` doubles, backtracking time roughly **quadruples** while hash-join time
roughly **doubles**, so the speedup column itself roughly **doubles** per step.

Illustrative run (numbers are host-dependent; the *ratios* are the point):

```
       N   matches     backtrack (s)      hashjoin (s)     speedup
------------------------------------------------------------------
     256       256       0.000074520       0.000024046        3.1x
    1024      1024       0.000991277       0.000098154       10.1x
    4096      4096       0.015782885       0.000407754       38.7x
   16384     16384       0.249438231       0.001869634      133.4x
   32768     32768       1.011504058       0.004070688      248.5x
```

## Running it

```sh
cargo run --release        # release matters: the gap is asymptotic, but debug
                           # overhead muddies the small-N timings
cargo test                 # unit tests, incl. the set-equality property
```

The sweep stops doubling `N` once a single backtracking run exceeds ~0.5 s, so
the whole thing finishes in a few seconds on any machine. Output goes to stdout
and to `results.csv` (columns: `n, matches, backtracking_secs, hashjoin_secs,
speedup`) for later plotting.

## Project layout

```
src/
  generator.rs    synthetic e-graph: builds R_f and R_g as vectors of tuples
  arrow_store.rs  Arrow layer: loads each relation into a RecordBatch (Int64 cols)
  matchers.rs     the two matchers (backtracking + relational hash join)
  main.rs         the runner: correctness phase, timing phase, table + CSV
cuda/
  ematching_gpu.cu  CPU+GPU benchmark: all six matchers (incl. Leapfrog Triejoin) in one binary (see cuda/README.md)
  README.md         build/run/verify + how it feeds from the Arrow columns
colab_ematching_gpu.ipynb   ready-to-run Google Colab notebook (free GPU) for the cuda/ benchmark
```

## Honest scope and limitations

This is a **smoke test**, not a reproduction of the paper's full evaluation.

* **Synthetic data, isolating the worst case.** The input is hand-crafted to be
  the single pattern + data shape that maximally separates backtracking from a
  join. It is *not* the real [egg](https://egraphs-good.github.io/) /
  [egglog](https://github.com/egraphs-good/egglog) benchmark suites. Running the
  actual benchmark suites (math, lambda, rational, etc.) is a later step.
* **A hash join, not a full generic join.** We implement a single two-relation
  hash join, which is all the pattern `f(α, g(α))` needs. The paper's general
  result uses a **worst-case-optimal generic join** (variable-at-a-time Generic
  Join / Leapfrog Triejoin) to handle arbitrary multi-relation patterns,
  including cyclic ones where even a tree of binary joins is suboptimal. The spot
  where that algorithm slots in is marked with a comment in
  `src/matchers.rs::hashjoin_match`.
* **The point is the exponent, not the constant.** Absolute timings depend on
  the machine, the Arrow access path, and the optimizer; the `O(N)` vs `O(N²)`
  separation is what is language- and implementation-independent.

These are the obvious next steps toward the full Coln implementation: swap the
hash join for a generic join, and drive it with the real e-graph benchmarks.

There is also a **GPU port** under `cuda/` (CUDA) that mirrors this benchmark and
shows the same O(N) vs O(N²) separation persisting on the GPU — parallelism
changes the constant, not the exponent. It is written but, lacking a GPU on the
authoring machine, not yet compiled; it self-verifies its output on every run.
See `cuda/README.md`.
