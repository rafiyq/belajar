use std::{vec, time::Instant};
use belajar::nlp::{nlp::{read_tweets, build_freqs, extract_features, test_logistic_regression, predict_tweet}, optim::gradient_descent, FType};
use ndarray::{Array2, Array, Axis};

fn main() {
    println!("Loading twitter sample...");
    let time_loading_data = Instant::now();

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
    println!("Time elapsed {}ms", time_loading_data.elapsed().as_millis());

    println!("Building frequency table...");
    let time_build_freqs = Instant::now();
    let freqs = build_freqs(&train_x, &train_y);
    let mut counter = 5;
    println!("sample from frequency table:");
    println!("\t-----------------------");
    for (key, value) in &freqs {
        println!("\t{} -> {}", key, value);
        if counter < 0 {break;}
        counter -= 1;
    }
    println!("\t-----------------------");
    println!("\tfreqs length: {}", freqs.keys().len());
    println!("Time elapsed {}ms", time_build_freqs.elapsed().as_millis());
    
    println!("Training the model...");
    let time_train_model = Instant::now();
    let mut features: Array2<FType> = Array2::zeros((train_x.len(), 3));
    counter = 1;
    println!("\tsample score table:");
    println!("\t-----------------------");
    println!("\tNo Positive Negative");
    println!("\t-----------------------");
    for i in 0..train_x.len() {
        let feat = extract_features(&train_x[i], &freqs);
        features[[i, 0]] = feat[[0, 0]];
        features[[i, 1]] = feat[[0, 1]];
        features[[i, 2]] = feat[[0, 2]];
        if counter <= 5 {
            println!("\t{i:.*}{p:.*}{n:.*}", 2,8,8, i=counter, p=(feat[[0, 1]] as i32), n=(feat[[0, 2]] as i32));
            counter += 1;
        }
    }
    println!("\t-----------------------\n");
    let labels: Array2<FType>= Array::from_vec(train_y).mapv(|x| x as FType).insert_axis(Axis(1));
    let (costs, theta) = gradient_descent(&features, &labels, Array2::zeros((3, 1)), 1e-9, 1500);
    println!("\tThe cost after training is {}", costs);
    println!("\tThe resulting vector of weights is\n\t{:?}", theta);
    println!("Time elapsed {}ms", time_train_model.elapsed().as_millis());

    println!("Testing the model...");
    let time_test_model = Instant::now();
    let tmp_accuracy = test_logistic_regression(test_x, test_y, &freqs, &theta);
    println!("\tLogistic regression model's accuracy = {:.4}", tmp_accuracy);
    println!("\tsample of prediction:");
    println!("\t-----------------------");
    let tweet_list = vec!["I am happy", "I am bad", "this movie should have been great.", "great", "great great", "great great great", "great great great great"];
    for i in tweet_list {
        let prob = predict_tweet(i, &freqs, &theta);
        let pred_text;
        if  prob > 0.5{
            pred_text = "Positive Sentiment"; 
        } else {
            pred_text = "Negative Sentiment";
        }
        println!("{} -> {:.2} ({})", i, prob, pred_text);
    }
    println!("\t-----------------------");
    println!("Time elapsed {}ms", time_test_model.elapsed().as_millis());
    println!("Total time {}ms", time_loading_data.elapsed().as_millis());
}