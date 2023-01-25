use std::io::{BufReader, BufRead};
use std::fs::File;
use std::collections::{BTreeMap, HashSet};
use ndarray::{Array, Array1};

use crate::FType;
use crate::func::linalg::cosine_similarity;

pub fn load_embeddings_subset() -> BTreeMap<String, Array1<FType>> {
    let file_path = "datasets/GoogleNews/list_word_embeddings_subset.pickle";
    let rdr = BufReader::new(File::open(file_path).unwrap());
    let word_embeddings_vec: BTreeMap<String, Vec<f64>> = serde_pickle::from_reader(rdr, Default::default()).unwrap();
    let mut word_embeddings: BTreeMap<String, Array1<FType>> = BTreeMap::new();
    for (key, v) in word_embeddings_vec {
        word_embeddings.insert(key, Array::from_vec(v));
    }
    word_embeddings
}
/// Input:
///     city1: a string (the capital city of country1)
///     country1: a string (the country of capital1)
///     city2: a string (the capital city of country2)
///     embeddings: a dictionary where the keys are words and
/// Output:
///     countries: a dictionary with the most likely country and its similarity score
pub fn get_country(city1: &str, country1: &str, city2: &str, embeddings: &BTreeMap<String, Array1<FType>>) -> Option<(String, FType)>{
    let group = HashSet::from([city1, country1, city2]);
    for key in group.iter() {
        if !embeddings.contains_key(*key) {
            panic!("word {} isn't contains in the dictionary", key)
        }
    }
    let vec = &embeddings[country1] - &embeddings[city1] + &embeddings[city2];
    let mut similarity = -1.;
    let mut found_word = String::new();
    for word in embeddings.keys() {
        if !group.contains(word.as_str()) {
            let cur_similarity = cosine_similarity(&vec, &embeddings[word]);
            if cur_similarity > similarity {
                similarity = cur_similarity;
                found_word = word.to_string();
            }
        }
    }
    match found_word.is_empty() {
        false => Some((found_word, similarity)),
        true => None
    }
}
/// Input:
///     word_embeddings: a dictionary where the key is a word and the value is its embedding
///     filepath: path to file location
pub fn get_accuracy(word_embeddings: &BTreeMap<String, Array1<FType>>, filepath: &str) -> FType{
    let mut num_correct = 0;
    let mut m = 0;
    let file = File::open(filepath).expect("Unable to open file.");
    let rdr = BufReader::new(file).lines();
    for r in rdr {
        m += 1;
        let s = r.expect(format!("Unable to read line {}", m).as_str());
        let words: Vec<&str> = s.split(" ").collect();
        let city1 = words[0];
        let country1 = words[1];
        let city2 = words[2];
        let country2 = words[3];
        match get_country(city1, country1, city2, word_embeddings) {
            Some((c2, _)) => if c2 == country2 { num_correct += 1},
            None => (),
        }       
    }
    num_correct as FType / m as FType
} 
