//! Benchmark runner for the "e-matching is a relational join" smoke test.
//!
//! Reproduces the core qualitative result of *Relational E-matching* (Zhang,
//! Wang, Willsey, Tatlock, POPL 2022): e-matching is a relational join, and a
//! relational approach beats naive backtracking search *asymptotically*.
//!
//! Phase 1 (correctness): for small `N`, assert the backtracking and hash-join
//! matchers return the *identical set* of substitutions. Timing does not start
//! until this passes.
//!
//! Phase 2 (timing): for a doubling range of `N` (stopping once the `O(N²)`
//! backtracking matcher gets expensive), time both matchers, confirm their match
//! counts agree, and emit a table to stdout plus `results.csv`.

mod arrow_store;
mod generator;
mod matchers;

use std::collections::HashSet;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use arrow_store::{build_rf, build_rg};
use generator::{expected_match_count, generate};
use matchers::{Subst, backtracking_match, hashjoin_match};

/// Smallest problem size in the timing sweep.
const N_START: i64 = 16;
/// Hard safety cap on `N` (the adaptive stop below normally fires first).
const N_MAX: i64 = 1 << 16;
/// Timing repetitions; we report the fastest run. The cheap linear matcher gets
/// more reps for a stable minimum; the quadratic one gets fewer to bound wall time.
const REPS_BT: u32 = 3;
const REPS_HJ: u32 = 9;
/// Stop doubling `N` once a single backtracking run exceeds this many seconds,
/// keeping the whole benchmark to a few seconds of wall time regardless of host.
const BT_STOP_SECS: f64 = 0.5;

/// Run `f` `reps` times, returning its (result, fastest observed duration).
fn time_matcher<F>(f: F, reps: u32) -> (HashSet<Subst>, Duration)
where
    F: Fn() -> HashSet<Subst>,
{
    let mut best = Duration::MAX;
    let mut result = None;
    for _ in 0..reps {
        let start = Instant::now();
        let r = f();
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
        result = Some(r);
    }
    (result.unwrap(), best)
}

/// Phase 1: identical-set correctness, asserted before any timing.
fn correctness_phase() {
    println!("== correctness phase: set equality of the two matchers ==");
    for n in [1_i64, 2, 4, 8, 16, 32, 64, 128] {
        let (rf, rg) = generate(n);
        let bf = build_rf(&rf);
        let bg = build_rg(&rg);

        let bt = backtracking_match(&bf, &bg);
        let hj = hashjoin_match(&bf, &bg);

        // Non-negotiable: identical *sets*, not merely equal counts.
        assert_eq!(
            bt, hj,
            "MATCHERS DISAGREE at N={n}: backtracking and hash join returned different sets"
        );
        // Independent check that this really is the n-match worst case.
        assert_eq!(
            bt.len(),
            expected_match_count(n),
            "unexpected match count at N={n}"
        );
        println!("  N={n:<4}  sets identical \u{2713}   matches={}", bt.len());
    }
    println!("correctness OK: both matchers compute the same substitution set.\n");
}

/// One row of benchmark results.
struct Row {
    n: i64,
    matches: usize,
    bt_secs: f64,
    hj_secs: f64,
}

/// Phase 2: time both matchers over a doubling range of `N`.
fn timing_phase() -> Vec<Row> {
    println!("== timing phase: backtracking (min of {REPS_BT}) vs hash join (min of {REPS_HJ}) ==");
    let mut rows = Vec::new();
    let mut n = N_START;
    while n <= N_MAX {
        let (rf, rg) = generate(n);
        let bf = build_rf(&rf);
        let bg = build_rg(&rg);

        // Data generation and Arrow construction are outside the timed region:
        // we measure only the matching algorithms, reading the same batches.
        let (bt_set, bt_dur) = time_matcher(|| backtracking_match(&bf, &bg), REPS_BT);
        let (hj_set, hj_dur) = time_matcher(|| hashjoin_match(&bf, &bg), REPS_HJ);

        // Counts must agree at every N (a cheap guard alongside the timing).
        assert_eq!(bt_set.len(), hj_set.len(), "match counts diverged at N={n}");

        let bt_secs = bt_dur.as_secs_f64();
        let hj_secs = hj_dur.as_secs_f64();
        rows.push(Row {
            n,
            matches: bt_set.len(),
            bt_secs,
            hj_secs,
        });
        println!("  N={n:<6} done  (backtrack {bt_secs:.6}s, hashjoin {hj_secs:.6}s)");

        if bt_secs > BT_STOP_SECS {
            println!("  (backtracking exceeded {BT_STOP_SECS}s — stopping the sweep)");
            break;
        }
        n <<= 1;
    }
    rows
}

fn speedup_ratio(r: &Row) -> Option<f64> {
    (r.hj_secs > 0.0).then(|| r.bt_secs / r.hj_secs)
}

fn print_table(rows: &[Row]) {
    println!("\n== results ==");
    println!(
        "{:>8}  {:>8}  {:>16}  {:>16}  {:>10}",
        "N", "matches", "backtrack (s)", "hashjoin (s)", "speedup"
    );
    println!("{}", "-".repeat(66));
    for r in rows {
        let speedup = match speedup_ratio(r) {
            Some(s) => format!("{s:.1}x"),
            None => "n/a".to_string(),
        };
        println!(
            "{:>8}  {:>8}  {:>16.9}  {:>16.9}  {:>10}",
            r.n, r.matches, r.bt_secs, r.hj_secs, speedup
        );
    }
}

fn write_csv(rows: &[Row], path: &str) -> std::io::Result<()> {
    let mut s = String::new();
    writeln!(s, "n,matches,backtracking_secs,hashjoin_secs,speedup").unwrap();
    for r in rows {
        let speedup = speedup_ratio(r).map(|s| format!("{s:.3}")).unwrap_or_default();
        writeln!(
            s,
            "{},{},{:.9},{:.9},{}",
            r.n, r.matches, r.bt_secs, r.hj_secs, speedup
        )
        .unwrap();
    }
    std::fs::write(path, s)
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("note: debug build — run `cargo run --release` for meaningful timings.\n");
    }

    // Phase 1 must pass before any timing happens.
    correctness_phase();

    // Phase 2.
    let rows = timing_phase();
    print_table(&rows);

    let csv_path = "results.csv";
    write_csv(&rows, csv_path).expect("failed to write results.csv");
    println!("\nwrote {csv_path} ({} data rows)", rows.len());
    println!("interpretation: backtracking time grows ~N² while hash-join time grows ~N,");
    println!("so the speedup column roughly doubles each time N doubles.");
}
