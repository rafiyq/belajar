use std::collections::HashMap;
use belajar::nlp::{twitter_datasets, build_freqs, train_naive_bayes};

fn main() {
    let (train_x, train_y, _test_x, _test_y) = twitter_datasets();
    let initial_freqs = HashMap::new();
    let freqs = build_freqs(initial_freqs, &train_x, &train_y);
    let (logprior, loglikelihood) = train_naive_bayes(&freqs, train_y);
    println!("logprior = {}", logprior);
    println!("length loglikelihood = {}", loglikelihood.len())
    // println!("{:?}", freq);
}