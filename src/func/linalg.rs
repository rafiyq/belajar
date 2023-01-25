use ndarray::Array1;
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