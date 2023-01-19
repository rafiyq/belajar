use ndarray::Array2;

/// ```text
///                 1
/// sigmoid(z) = ――――――――
///              1 + exp⁻ᶻ
/// ``` 
/// Input:
///     z: is the input (can be a scalar or an array)
/// Output:
///     z with the sigmoid implemented
pub fn sigmoid(z: Array2<f32>) -> Array2<f32> {
    1. / (1. + (-z).mapv(f32::exp))
}