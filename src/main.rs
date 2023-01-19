use std::{vec, time::Instant};
use belajar::{nlp::{read_tweets, build_freqs, extract_features}, optim::gradient_descent, FType};
use ndarray::{Array2, Array, Axis};

fn main() {
    println!("Loading twitter sample...");
    let start_loading_data = Instant::now();

    let mut all_positive_tweets = read_tweets("datasets/twitter_samples/positive_tweets.json");
    let mut all_negative_tweets = read_tweets("datasets/twitter_samples/negative_tweets.json");
    println!("\tpositive tweet: {}", all_positive_tweets.len()); println!("\tnegative tweet: {}", all_negative_tweets.len());
    println!("\t-----------------------");

    // Create training set
    let train_pos_len = 4000; let train_neg_len = 4000;
    let test_pos_len = all_positive_tweets.len() - train_pos_len; let test_neg_len = all_negative_tweets.len() - train_neg_len;
    let test_x = [all_positive_tweets.split_off(train_pos_len), all_negative_tweets.split_off(train_neg_len)].concat();
    // Create test set
    let train_x = [all_positive_tweets, all_negative_tweets].concat();
    println!("\ttrain_x: {}", train_x.len()); println!("\ttest_x: {}", test_x.len());
    println!("\t-----------------------");

    let train_y = [vec![1; train_pos_len], vec![0; train_neg_len]].concat();
    let test_y = [vec![1; test_pos_len], vec![0; test_neg_len]].concat();
    println!("\ttrain_y: {}", train_y.len()); println!("\ttest_y: {}", test_y.len());
    println!("Time elapsed {}ms", start_loading_data.elapsed().as_millis());

    println!("Building frequency table...");
    let start_build_freqs = Instant::now();
    let freqs = build_freqs(&train_x, &train_y);
    println!("\tfreqs length: {}", freqs.keys().len());
    let mut counter = 5;
    for (key, value) in &freqs {
        println!("\t{}: {}", key, value);
        if counter < 0 {break;}
        counter -= 1;
    }
    println!("Time elapsed {}ms", start_build_freqs.elapsed().as_millis());

    println!("Training the model...");
    let mut features: Array2<FType> = Array2::zeros((train_x.len(), 3));

    for i in 0..train_x.len() {
        let feat = extract_features(&train_x[i], &freqs);
        features[[i, 0]] = feat[0];
        features[[i, 1]] = feat[1];
        features[[i, 2]] = feat[2];
    }
    let _labels: Array2<FType>= Array::from_vec(train_y).mapv(|x| x as FType).insert_axis(Axis(1));

    let (costs, theta) = gradient_descent(&features, &_labels, Array2::zeros((3, 1)), 1e-9, 1500);
    println!("The cost after training is {}", costs);
    println!("The resulting vector of weights is {}", theta);
}

// use ndarray::{Array2, Axis, array, AssignElem};
// fn main() {
//     let mut a: Array2<f32> = Array2::zeros((5, 3));
//     for i in a.axis_iter(Axis(0)) {
//         // let b = a.axis_iter(Axis(0));
//         println!("{:?}", i);

//     }
// }