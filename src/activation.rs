use ndarray::Array2;
use crate::FType;

extern crate blas_src;

/// ```text
///                 1
/// sigmoid(z) = ――――――――
///              1 + exp⁻ᶻ
/// ``` 
/// Input:
///     z: is the input (can be a scalar or an array)
/// Output:
///     z with the sigmoid implemented
pub fn sigmoid(z: Array2<FType>) -> Array2<FType> {
    1. / (1. + (-z).mapv(FType::exp))
}