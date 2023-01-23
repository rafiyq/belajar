use std::collections::HashMap;
use belajar::nlp::nlp::{twitter_datasets, build_freqs, train_naive_bayes, naive_bayes_predict, test_naive_bayes, get_ratio, get_word_by_treshold, process_tweet};

fn main() {
    let (train_x, train_y, test_x, test_y) = twitter_datasets();
    let initial_freqs = HashMap::new();
    let freqs = build_freqs(initial_freqs, &train_x, &train_y);
    let (logprior, loglikelihood) = train_naive_bayes(&freqs, train_y);
    println!("logprior = {}", logprior);
    println!("length loglikelihood = {}", loglikelihood.len());
    let words = ["She smiled.", "he Laught", "I am happy", "I am bad", "this movie should have been great.", "great", "great great", "great great great", "great great great great"];
    println!("sample prediction:");
    for  word in words {
        let prob = naive_bayes_predict(word, &logprior, &loglikelihood);
        println!("\t{} -> {:.2}", word, prob);
    }
    println!("Model accuracy = {}", test_naive_bayes(&test_x, &test_y, &logprior, &loglikelihood));
    let word = "happi";
    println!("ratio of word {:?}:\n\t{:?}", word, get_ratio(&freqs, word));
    let below_threshold = get_word_by_treshold(&freqs, 0, 0.05);
    let above_threshold = get_word_by_treshold(&freqs, 1, 10.);
    println!("negative word below 0.05 = {:?}", below_threshold.len());
    println!("positive word above 10 = {:?}", above_threshold.len());
    println!("Error Analysis:");
    println!("\ttruth predicted tweet");
    for (x, y) in test_x.iter().zip(&test_y) {
        let y_hat = naive_bayes_predict(x, &logprior, &loglikelihood);
        if (*y == 1) != y_hat.is_sign_positive() {
            println!("\t{}\t{:.2}\t{}", y, y_hat.is_sign_positive() as i32, process_tweet(x, true, true, true).join(" "));
        }
    }
}