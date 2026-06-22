//! Arrow layer.
//!
//! The relations live in Arrow [`RecordBatch`]es with named `Int64` columns, and
//! every matcher reads from these batches rather than the raw vectors. This
//! mirrors the columnar, Arrow-based storage the larger Coln system will use, so
//! the matchers are exercised against the representation that matters.

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::generator::{RowF, RowG};

/// Project tuple field `f` of every row into an Arrow `Int64` column.
fn column<R>(rows: &[R], f: impl Fn(&R) -> i64) -> ArrayRef {
    let values: Vec<i64> = rows.iter().map(f).collect();
    Arc::new(Int64Array::from(values)) as ArrayRef
}

/// Load `R_f` into a 3-column `Int64` batch: `id, arg1, arg2`.
pub fn build_rf(rows: &[RowF]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("arg1", DataType::Int64, false),
        Field::new("arg2", DataType::Int64, false),
    ]);
    let columns = vec![
        column(rows, |r| r.0),
        column(rows, |r| r.1),
        column(rows, |r| r.2),
    ];
    RecordBatch::try_new(Arc::new(schema), columns).expect("R_f columns are equal length")
}

/// Load `R_g` into a 2-column `Int64` batch: `id, arg1`.
pub fn build_rg(rows: &[RowG]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("arg1", DataType::Int64, false),
    ]);
    let columns = vec![column(rows, |r| r.0), column(rows, |r| r.1)];
    RecordBatch::try_new(Arc::new(schema), columns).expect("R_g columns are equal length")
}

/// Borrow column `idx` of `batch` as a typed `Int64Array`.
///
/// Panics if the column is not `Int64` — acceptable here because we build every
/// batch ourselves with the fixed `Int64` schemas just above.
pub fn col_i64(batch: &RecordBatch, idx: usize) -> &Int64Array {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column should be Int64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::generate;

    #[test]
    fn batches_have_expected_schema_and_values() {
        let (rf, rg) = generate(4);
        let bf = build_rf(&rf);
        let bg = build_rg(&rg);

        assert_eq!(bf.num_columns(), 3);
        assert_eq!(bg.num_columns(), 2);
        assert_eq!(bf.num_rows(), 4);
        assert_eq!(bg.num_rows(), 4);

        assert_eq!(bf.schema().field(0).name(), "id");
        assert_eq!(bf.schema().field(1).name(), "arg1");
        assert_eq!(bf.schema().field(2).name(), "arg2");
        assert_eq!(bg.schema().field(0).name(), "id");
        assert_eq!(bg.schema().field(1).name(), "arg1");

        // Round-trip a value through the columnar layer.
        assert_eq!(col_i64(&bf, 1).value(0), 1); // first α in R_f
        assert_eq!(col_i64(&bg, 1).value(3), 4); // last α in R_g
    }
}
