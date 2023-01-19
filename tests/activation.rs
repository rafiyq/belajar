use belajar::{activation::sigmoid, FType};
use ndarray::array;

#[test]
fn sigmoid_test() {
    assert_eq!(sigmoid(array![[0.]]), array![[0.5]]);
    // positive_check
    assert!(sigmoid(array![[4.92]]).abs_diff_eq(&array![[0.9927537604041685]], FType::EPSILON));
    // negative_check
    assert_eq!(sigmoid(array![[-1.]]), array![[0.2689414213699951]]);
    // larger_neg_check
    assert_eq!(sigmoid(array![[-20.]]), array![[2.0611536181902037e-09]]);
}