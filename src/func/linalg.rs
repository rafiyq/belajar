use ndarray::Array1;
use ndarray_linalg::Norm;
use crate::FType;

/// Input:
///     a: a vector
///     b: another vector 
/// Output:
///     numerical number representing the cosine similarity between a and b.
pub fn cosine_similarity_vec(a: Vec<FType>, b: Vec<FType>) -> FType {
    let a_v = Array1::from_vec(a);
    let b_v = Array1::from_vec(b);
    a_v.dot(&b_v) / (a_v.norm_l2() * b_v.norm_l2())
}
/// Input:
///     A: a numpy array which corresponds to a word vector
///     B: A numpy array which corresponds to a word vector
/// Output:
///     d: numerical number representing the Euclidean distance between A and B.
pub fn euclidean_vec(a: Vec<FType>, b: Vec<FType>) -> FType {
    let x = Array1::from_vec(a);
    let y = Array1::from_vec(b);
    (x-y).mapv(|f|f.powi(2)).sum().sqrt()
}