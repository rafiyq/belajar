use ndarray::Array2;
use crate::{func::activation::sigmoid, func::loss::binary_cross_entropy, FType};

/// Input:
///     x: matrix of features which is (m,n+1)
///     y: corresponding labels of the input matrix x, dimensions (m,1)
///     theta: weight vector of dimension (n+1,1)
///     alpha: learning rate
///     num_iters: number of iterations you want to train your model for
/// Output:
///     J: the final cost
///     theta: your final weight vector
pub fn gradient_descent(x: &Array2<FType>, y: &Array2<FType>, mut theta: Array2<FType>, alpha: FType, num_iters:i32) -> (FType, Array2<FType>) {
    // the number of rows in matrix x
    let m = x.nrows() as FType;
    let mut cost = 0 as FType;
    for _ in 0..num_iters {
        let z = x.dot(&theta);
        // sigmoid of z
        let h = sigmoid(z);
        // println!("h: {}", h);
        // calculate the cost function
        cost = binary_cross_entropy(&h, y, "mean").last().unwrap().to_owned();
        // println!("J: {}", cost);
        // update the weights theta
        theta = theta - alpha/m * (x.clone().reversed_axes().dot(&(h-y)));
        //theta = theta - theta2;
    }
    (cost, theta)
}