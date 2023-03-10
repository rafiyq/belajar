use belajar::{optim::gradient_descent, FType};
use ndarray::{array, Array2};

#[test]
fn gradient_descent_default_check() {
    let x = array![
        [1.00000000e00, 8.34044009e02, 1.44064899e03],
        [1.00000000e00, 2.28749635e-01, 6.04665145e02],
        [1.00000000e00, 2.93511782e02, 1.84677190e02],
        [1.00000000e00, 3.72520423e02, 6.91121454e02],
        [1.00000000e00, 7.93534948e02, 1.07763347e03],
        [1.00000000e00, 8.38389029e02, 1.37043900e03],
        [1.00000000e00, 4.08904499e02, 1.75623487e03],
        [1.00000000e00, 5.47751864e01, 1.34093502e03],
        [1.00000000e00, 8.34609605e02, 1.11737966e03],
        [1.00000000e00, 2.80773877e02, 3.96202978e02],
    ];
    let y = array![
        [1.0],
        [1.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ];
    let theta: Array2<FType> = Array2::zeros((3, 1));
    let alpha = 1e-8;
    let num_iters = 700;
    let (cost, theta_after) = gradient_descent(&x, &y, theta, alpha, num_iters);
    assert!(array![cost].abs_diff_eq(&array![0.6709497038162118], 1e-8));
    assert!(theta_after.abs_diff_eq(&array![[4.10713435e-07], [3.56584699e-04], [7.30888526e-05]], 1e-8));
}
#[test]
fn gradient_descent_larger_check() {
    let x = array![
        [1.0, 435.99490214, 25.92623183, 549.66247788],
        [1.0, 435.32239262, 420.36780209, 330.334821],
        [1.0, 204.64863404, 619.27096635, 299.65467367],
        [1.0, 266.8272751, 621.13383277, 529.14209428],
        [1.0, 134.57994534, 513.57812127, 184.43986565],
        [1.0, 785.33514782, 853.97529264, 494.23683738],
        [1.0, 846.56148536, 79.64547701, 505.24609012],
        [1.0, 65.28650439, 428.1223276, 96.53091566],
        [1.0, 127.1599717, 596.74530898, 226.0120006],
        [1.0, 106.94568431, 220.30620707, 349.826285],
        [1.0, 467.78748458, 201.74322626, 640.40672521],
        [1.0, 483.06983555, 505.23672002, 386.89265112],
        [1.0, 793.63745444, 580.00417888, 162.2985985],
        [1.0, 700.75234661, 964.55108009, 500.00836117],
        [1.0, 889.52006395, 341.61365267, 567.14412763],
        [1.0, 427.5459633, 436.74726303, 776.559185],
        [1.0, 535.6041735, 953.74222694, 544.20816015],
        [1.0, 82.09492228, 366.34240168, 850.850504],
        [1.0, 406.27504305, 27.20236589, 247.177239],
        [1.0, 67.14437074, 993.85201142, 970.58031338],
    ];
    let y = array![
        [1.0],
        [1.0],
        [1.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
    ];
    let theta: Array2<FType> = Array2::zeros((4, 1));
    let alpha = 1e-4;
    let num_iters = 30;
    let (cost, theta_after) = gradient_descent(&x, &y, theta, alpha, num_iters);
    assert!(array![cost].abs_diff_eq(&array![6.5044107216556135], 1e-8));
    assert!(theta_after.abs_diff_eq(&array![                        
        [9.45211976e-05],
        [2.40577958e-02],
        [-1.77876847e-02],
        [1.35674845e-02],
    ], 1e-8));
}