use belajar::nlp::wordv::load_embeddings_subset;
use belajar::func::linalg::{cosine_similarity_vec, euclidean_vec};

#[test]
fn cosine_similarity_vec_tests() {
    let word_embeddings = load_embeddings_subset();
    assert_eq!(cosine_similarity_vec(&word_embeddings[&"king".to_string()], &word_embeddings[&"queen".to_string()]), 0.6510956835386662);
    assert_eq!(cosine_similarity_vec(&word_embeddings[&"Japan".to_string()], &word_embeddings[&"Tokyo".to_string()]), 0.700225388633558);
    assert_eq!(cosine_similarity_vec(&word_embeddings[&"Germany".to_string()], &word_embeddings[&"Beirut".to_string()]), 0.17339969518582382);
    assert_eq!(cosine_similarity_vec(&word_embeddings[&"China".to_string()], &word_embeddings[&"Chile".to_string()]), 0.3801231691333899);
}
#[test]
fn euclidean_vec_tests() {
    let word_embeddings = load_embeddings_subset();
    assert_eq!(euclidean_vec(&word_embeddings[&"king".to_string()], &word_embeddings[&"queen".to_string()]), 2.4796923748357105);
    assert_eq!(euclidean_vec(&word_embeddings[&"Japan".to_string()], &word_embeddings[&"Tokyo".to_string()]), 2.434534503534425);
    assert_eq!(euclidean_vec(&word_embeddings[&"Germany".to_string()], &word_embeddings[&"Beirut".to_string()]), 4.041651831350544);
    assert_eq!(euclidean_vec(&word_embeddings[&"China".to_string()], &word_embeddings[&"Chile".to_string()]), 3.2326782055238845);
}