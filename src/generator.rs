//! Synthetic e-graph generator.
//!
//! An e-graph is a database of terms: each function symbol becomes a relation,
//! and a function of arity `k` becomes a table with `k + 1` integer columns —
//! the first is the e-class id of the result, the rest are the children's
//! e-class ids.
//!
//! We generate the two relations needed for the pattern `f(α, g(α))`:
//!
//! ```text
//!   R_f = { (I_F, k, I_G) : k in 1..=N }     (id, arg1=α, arg2=g-application)
//!   R_g = { (I_G, k)      : k in 1..=N }     (id, arg1=α)
//! ```
//!
//! This is deliberately the separating worst case from the paper. Matching
//! `f(α, g(α))` yields exactly `N` substitutions (one per `k`), yet *every* one
//! of the `N × N` (f-row, g-row) pairs passes the structural test "f's second
//! child is a g-application", because all g-applications share the single result
//! e-class `I_G`. Only the equality constraint on `α` prunes `N²` down to `N`.

/// Result e-class id shared by every `f`-application.
///
/// Chosen far above the `α` id range (`1..=N`) purely so the three roles never
/// visually collide when you eyeball the data. Correctness does not depend on
/// this: the relations are matched positionally, column by column, so the ids
/// could just as well overlap (as they routinely do in a real e-graph).
pub const I_F: i64 = 1_000_000;

/// Result e-class id shared by every `g`-application.
pub const I_G: i64 = 2_000_000;

/// A row of `R_f`: `(id = result e-class, arg1 = first child α, arg2 = second child)`.
pub type RowF = (i64, i64, i64);

/// A row of `R_g`: `(id = result e-class, arg1 = only child α)`.
pub type RowG = (i64, i64);

/// Build the two relations for a problem of size `n`.
pub fn generate(n: i64) -> (Vec<RowF>, Vec<RowG>) {
    assert!(n >= 1, "n must be >= 1");
    let r_f: Vec<RowF> = (1..=n).map(|k| (I_F, k, I_G)).collect();
    let r_g: Vec<RowG> = (1..=n).map(|k| (I_G, k)).collect();
    (r_f, r_g)
}

/// The number of matches the pattern *should* produce on `generate(n)`.
///
/// Used as an independent sanity check in the runner: it pins down that this is
/// really the `n`-match worst case rather than something that accidentally
/// collapsed or blew up.
pub fn expected_match_count(n: i64) -> usize {
    n as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_contents() {
        let (rf, rg) = generate(3);
        assert_eq!(rf, vec![(I_F, 1, I_G), (I_F, 2, I_G), (I_F, 3, I_G)]);
        assert_eq!(rg, vec![(I_G, 1), (I_G, 2), (I_G, 3)]);
    }

    #[test]
    fn lengths_scale_with_n() {
        let (rf, rg) = generate(100);
        assert_eq!(rf.len(), 100);
        assert_eq!(rg.len(), 100);
        assert_eq!(expected_match_count(100), 100);
    }
}
