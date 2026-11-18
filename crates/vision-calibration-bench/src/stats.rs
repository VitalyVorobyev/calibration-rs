//! Summary statistics shared by the harness's records and reports.

/// The median of `values`, sorted in place by [`f64::total_cmp`]: the middle
/// value of an odd count, the mean of the middle pair of an even one; `None`
/// when empty.
pub(crate) fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_even_unsorted_and_empty_lists() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }
}
