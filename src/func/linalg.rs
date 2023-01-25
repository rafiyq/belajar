use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Norm;
use crate::FType;

/// Input:
///     a: a vector
///     b: another vector 
/// Output:
///     numerical number representing the cosine similarity between a and b.
pub fn cosine_similarity(a: &Array1<FType>, b: &Array1<FType>) -> FType {
    a.dot(b) / (a.norm_l2() * b.norm_l2())
}
/// Input:
///     a: a vector
///     b: another vector 
/// Output:
///     umerical number representing the Euclidean distance between a and b.
pub fn euclidean(a: &Array1<FType>, b: &Array1<FType>) -> FType {
    (a-b).mapv(|f|f.powi(2)).sum().sqrt()
}
/// Estimate a covariance matrix, given data. 
/// Covariance indicates the level to which two variables vary together.
/// Parameters:
///     x: A 2-D array containing multiple variables and observations.
///     rowvar: If `rowvar` is true, then each row represents a variable,
///     with observations in the columns. Otherwise, the relationship
///     is transposed: each column represents a variable, while the rows
///     contain observations.
pub fn covariance(mut x: Array2<FType>, rowvar: bool) -> Array2<FType> {
    if !rowvar && x.nrows() != 1 { x.swap_axes(0, 1); }
    if x.nrows() == 0 { return x }
    let ddof = 1;
    let mut fact: FType = x.ncols() as FType - ddof as FType;
    if fact <= 0.0 { fact = 0.0}
    x -= &x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    x = x.dot(&x.t());
    x * (1. / fact)
}