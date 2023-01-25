use std::time::Instant;
use belajar::nlp::wordv::{load_embeddings_subset, get_accuracy};

fn main() {
    println!("hello vectors");
    let time = Instant::now();
    let word_embeddings = load_embeddings_subset();
    println!("length: {:?}", word_embeddings.len());
    println!("dimension: {:?}", word_embeddings["Spain"].len());
    let path = "datasets/GoogleNews/capitals.txt";
    let accuracy = get_accuracy(&word_embeddings, path);
    println!("Accuracy is {:.2}", accuracy);
    println!("Time elapsed {}ms", time.elapsed().as_millis());
}