//! Two matchers for the pattern `f(α, g(α))`.
//!
//! The pattern compiles to the conjunctive query
//!
//! ```text
//!   Q(root, α) ← R_f(root, α, x), R_g(x, α)
//! ```
//!
//! `α` appears twice, which is an equality constraint; `x` is the join variable
//! linking f's second child to g's result e-class. Both matchers return the
//! *set* of substitutions `(root, α)`.
//!
//! Column layout (see [`crate::arrow_store`]):
//!
//! ```text
//!   R_f : [0]=id(root)  [1]=arg1(α)  [2]=arg2(x)
//!   R_g : [0]=id(x)     [1]=arg1(α)
//! ```

use arrow::record_batch::RecordBatch;
use std::collections::HashSet;

use crate::arrow_store::col_i64;

/// A substitution for the pattern's output variables: `(root, α)`.
pub type Subst = (i64, i64);

/// Prover-style baseline: nested backtracking search — `Θ(N²)` on this data.
///
/// For every f-row we scan *every* g-row, first checking the structural
/// constraint (f's second child `x` must be the result e-class of this
/// g-application) and only then — last — the equality constraint that the two
/// occurrences of `α` agree. This ordering is exactly how a naive prover does
/// e-matching: bind the head symbol, descend structurally, and discover the
/// shared-variable conflict only at the leaf.
///
/// On the synthetic data the structural test passes for all `N × N` pairs (every
/// g-application lives in the one e-class `I_G` that f points at), so the
/// equality test is *reached* `N²` times even though only `N` pairs survive it.
pub fn backtracking_match(r_f: &RecordBatch, r_g: &RecordBatch) -> HashSet<Subst> {
    let f_id = col_i64(r_f, 0);
    let f_a1 = col_i64(r_f, 1); // α
    let f_a2 = col_i64(r_f, 2); // x
    let g_id = col_i64(r_g, 0); // x
    let g_a1 = col_i64(r_g, 1); // α

    let mut out = HashSet::new();
    for i in 0..r_f.num_rows() {
        let root = f_id.value(i);
        let f_alpha = f_a1.value(i);
        let f_x = f_a2.value(i);
        for j in 0..r_g.num_rows() {
            // 1. structural: is f's second child a g-application in that e-class?
            if f_x == g_id.value(j) {
                // 2. equality (checked last): do the two α occurrences agree?
                if f_alpha == g_a1.value(j) {
                    out.insert((root, f_alpha));
                }
            }
        }
    }
    out
}

/// Relational hash join: evaluate the conjunctive query directly — `Θ(N)`.
///
/// The join between `R_f` and `R_g` binds two variables at once: `x` (structural:
/// `R_f.arg2 = R_g.id`) and `α` (equality: `R_f.arg1 = R_g.arg1`). We fold *both*
/// into one composite key `(x, α)` and index `R_g` on it, so the structural and
/// equality constraints are resolved *together* in a single hash lookup instead
/// of one-then-the-other. Build + probe is linear in `N`.
pub fn hashjoin_match(r_f: &RecordBatch, r_g: &RecordBatch) -> HashSet<Subst> {
    let f_id = col_i64(r_f, 0);
    let f_a1 = col_i64(r_f, 1); // α
    let f_a2 = col_i64(r_f, 2); // x
    let g_id = col_i64(r_g, 0); // x
    let g_a1 = col_i64(r_g, 1); // α

    // Build phase: index R_g on the join key (x, α) = (g.id, g.arg1).
    //
    // R_g carries no columns beyond the two join variables, so a *set* of keys
    // is a sufficient index. A general join would store `HashMap<key, Vec<row>>`
    // here, keeping the inner relation's remaining (payload) columns for output.
    let mut index: HashSet<(i64, i64)> = HashSet::with_capacity(r_g.num_rows());
    for j in 0..r_g.num_rows() {
        index.insert((g_id.value(j), g_a1.value(j)));
    }

    // Probe phase: for each f-row, look up (x, α) = (f.arg2, f.arg1).
    let mut out = HashSet::new();
    for i in 0..r_f.num_rows() {
        let key = (f_a2.value(i), f_a1.value(i));
        if index.contains(&key) {
            out.insert((f_id.value(i), f_a1.value(i))); // (root, α)
        }
    }

    // ---- GENERIC JOIN GOES HERE ------------------------------------------
    // This is a single two-relation hash join, which is all the pattern
    // f(α, g(α)) needs. The paper's result is the *general* case: an arbitrary
    // pattern compiles to a multi-relation conjunctive query, evaluated with a
    // worst-case-optimal *generic join* (egg / egglog use a variable-at-a-time
    // Generic Join / Leapfrog-Triejoin). That algorithm picks a global variable
    // order, builds a trie index per relation, and intersects the relations one
    // variable at a time. Replacing this hash join with that generic join is the
    // next step toward the full Coln implementation.
    // ----------------------------------------------------------------------

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_store::{build_rf, build_rg};
    use crate::generator::{I_F, generate};

    #[test]
    fn matchers_agree_and_are_correct() {
        for n in [1_i64, 2, 5, 16, 50] {
            let (rf, rg) = generate(n);
            let bf = build_rf(&rf);
            let bg = build_rg(&rg);

            let bt = backtracking_match(&bf, &bg);
            let hj = hashjoin_match(&bf, &bg);

            // Identical sets — the non-negotiable correctness property.
            assert_eq!(bt, hj, "matchers disagree at n={n}");
            // Exactly the worst-case n matches.
            assert_eq!(bt.len(), n as usize, "expected n matches at n={n}");
            // And specifically the set {(I_F, k) : k in 1..=n}.
            let expected: HashSet<Subst> = (1..=n).map(|k| (I_F, k)).collect();
            assert_eq!(bt, expected, "wrong match set at n={n}");
        }
    }
}
