use std::{fs, collections::{HashMap, HashSet}, iter::zip};
use html_escape::decode_html_entities;
use ndarray::{Array2, array, Array, Axis};
use serde_json::Value;
use lazy_static::lazy_static;
use regex::Regex;
use crate::{FType, activation::sigmoid};

macro_rules! fc_regex {
    ($re:expr) => {
        ::fancy_regex::Regex::new($re).unwrap()
    };
}

pub fn read_tweets(path:&str) -> Vec<String> {
    let mut tweets:Vec<String> = Vec::new();
    let data = fs::read_to_string(path).expect("Unable to read file");
    for line in data.lines() {
        let tweet: Value = serde_json::from_str(line).unwrap();
        // println!("{}", tweet.get("text").unwrap().as_str().unwrap().to_string());
        tweets.push(tweet.get("text").unwrap().as_str().unwrap().to_string());
        // break;
    }
    tweets
}
/// Process tweet function.
/// TODO: make PorterStemmer to behave like ntlk's version. 
/// Input:
///     tweet: a string containing a tweet
/// Output:
///     tweets_clean: a list of words containing the processed tweet
pub fn process_tweet(
    tweet: &str, 
    sentiment: bool, 
    strip_handles: bool,
    reduce_len: bool,
) -> Vec<String> {
    // ref: https://github.com/nltk/nltk/blob/175929bc47f818b5fbd8475daf831ff748b74170/nltk/tokenize/casual.py
    lazy_static! {
        static ref SEMTIMEN: Vec<Regex> = vec![
            Regex::new(r"\$\w*").unwrap(),                  // stock market tickers like $GE
            Regex::new(r"^RT[\s]+").unwrap(),               // old style retweet text "RT"
            Regex::new(r"https?://[^\s\n\r]+").unwrap(),    // hyperlinks
            Regex::new(r"#").unwrap(),                      // the hash # sign from the word
        ];
        static ref HANDLES: Vec<fancy_regex::Regex> = vec![ // Twitter username handles from text.
            fc_regex!(r"(?<![A-Za-z0-9_!@#\$%&*])@"),
            fc_regex!(r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"),
            fc_regex!(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"),
        ];
        static ref REDUCE_LEN: fancy_regex::Regex = fc_regex!(r"(.)\1{2,}");
        static ref HANG_RE: fancy_regex::Regex = fc_regex!(r"([^a-zA-Z0-9])\1{3,}");
        static ref PHONE_RE: Regex =  Regex::new(
            r"(?:(?:\+?[01][ *\-.\)]*)?(?:[\(]?\d{3}[ *\-.\)]*)?\d{3}[ *\-.\)]*\d{4})"
        ).unwrap();
        static ref EMOTICONS: Regex = Regex::new(&format!("{}{}{}{}|{}{}{}{}|{}",
            r"(?:[<>]?",
            r"[:;=8]",                    // eyes
            r"[\-o\*']?",                 // optional nose
            r"[\)\]\(\[dDpP/:\}\{@\|\\]", // mouth
            r"[\)\]\(\[dDpP/:\}\{@\|\\]", // mouth
            r"[\-o\*']?",                 // optional nose
            r"[:;=8]",                    // eyes
            r"[<>]?",
            r"</?3)"                      // heart
        )).unwrap();
        static ref FLAGS: Regex = Regex::new(&format!(r"(?:{}|{}|{}|{})",
            r"[\U0001F1E6-\U0001F1FF]{2}",  // all enclosed letter pairs
            r"\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006e\U000E0067\U000E007F",
            r"\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
            r"\U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F"
        )).unwrap();
        static ref WORD_RE: Regex = Regex::new(&format!("{}|{}|{}|{}|{}",
            r"(?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_])",  // Words with apostrophes or dashes.
            r"(?:[+\-]?\d+[,/.:-]\d+[+\-]?)",             // Numbers, including fractions, decimals.
            r"(?:[\w_]+)",                                // Words without apostrophes or dashes.
            r"(?:\.(?:\s*\.){1,})",                       // Ellipsis dots.
            r"(?:\S)"
        )).unwrap();
        static ref REGEXPS: Regex = Regex::new(&format!("{}|{}|{}",
            EMOTICONS.as_str(),
            FLAGS.as_str(),
            WORD_RE.as_str()
        )).unwrap();
    }
    let mut clean_tweet = tweet.to_owned();
    if sentiment {
        for i in 0..SEMTIMEN.len() {
            clean_tweet = SEMTIMEN[i].replace(&clean_tweet, "").to_string();
        }
    }
    if strip_handles {
        for i in 0..HANDLES.len() {
            clean_tweet = HANDLES[i].replace(&clean_tweet, "").to_string();
        }
    }
    clean_tweet = decode_html_entities(&clean_tweet).to_string();
    if reduce_len {
        // Replace repeated character sequences of length 3 or greater with sequences of length 3.
        clean_tweet = REDUCE_LEN.replace(&clean_tweet, "$1$1$1").to_string();
    }
    // Shorten problematic sequences of characters
    clean_tweet = HANG_RE.replace(&clean_tweet, "$1$1$1").to_string();

    // Extract tokens from tweet
    let tweet_tokens: Vec<&str> = REGEXPS.find_iter(&clean_tweet).map(|mat| mat.as_str()).collect();
    let mut tweets_clean :Vec<String>= Vec::new();
    let stopwords_english = stopwords("english");
    for w in tweet_tokens {
        let mut word = w.to_owned();
        if !EMOTICONS.is_match(w) {
            word.make_ascii_lowercase();
        }
        if !stopwords_english.iter().any(|i| i == &word) && 
            (word.len() != 1 || !(word.chars().nth(0).unwrap() as u8).is_ascii_punctuation()) {
            match stem::get(&word) {
                Ok(s) => tweets_clean.push(s),
                Err(_) => tweets_clean.push(word),
            };
            // if word.is_ascii(){
            //     tweets_clean.push(stem::get(&word).unwrap());
            //     println!("alive");
            // } else {
            //     tweets_clean.push(word)
            // }
        }
    }
    tweets_clean
}
pub fn stopwords<'life>(lang:&str) -> Vec<String> {
    let path = format!("datasets/twitter_samples/stopwords/{}", lang);
    let mut words:Vec<String> = Vec::new();
    let data = fs::read_to_string(path).expect("Unable to read file");
    for line in data.lines() {
        words.push(line.to_owned());
    }
    words
}
/// Build frequencies.
/// Input:
///     freqs: a dictionary that will be used to map each pair to its frequency
///     tweets: a list of tweets
///     label: an m x 1 array with the sentiment label of each tweet
///     (either 0 or 1)
/// Output:
///     freqs: a dictionary mapping each (word, sentiment) pair to its
///     frequency
pub fn build_freqs(
    mut freqs: HashMap<(String, i32), i32>, 
    tweets: &Vec<String>, 
    labels: &Vec<i32>
) -> HashMap<(String, i32), i32> {
    for (label, tweet) in zip(labels, tweets) {
        let tokens = process_tweet(tweet, true, true, true);
        for word in tokens {
            let pair = (word, *label);
            freqs.entry(pair).and_modify(|v| *v += 1).or_insert(1);
        }
    }
    freqs
}
/// Input: 
///     tweet: a list of words for one tweet
///     freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
/// Output: 
///     a feature vector of dimension (1,3)
pub fn extract_features(tweet: &str, freqs: &HashMap<(String, i32), i32>) -> Array2<FType> {
    // process_tweet tokenizes, stems, and removes stopwords
    let word_list = process_tweet(tweet, true, true, true);
    // 3 elements in the form of a 1 x 3 vector
    let mut feature: Array2<FType> = Array2::zeros((1, 3));
    //let mut feature = vec![0.; 3];
    // bias term is set to 1
    feature[[0, 0]] = 1.;

    for word in word_list {
        for (i, k) in [(word.clone(), 1), (word, 0)].iter().enumerate() {
            match freqs.get(k) {
                // increment the word count
                Some(score) => feature[[0, i]] += *score as FType,
                None => ()
            }
        }
    }
    assert!(feature.shape() == [1, 3]);
    feature
}
/// Input: 
///     tweet: a string
///     freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
///     theta: (3,1) vector of weights
/// Output: 
///     the probability of a tweet being positive or negative
pub fn predict_tweet(tweet: &str, freqs: &HashMap<(String, i32), i32>, theta: &Array2<FType>) -> FType {
    let feature = extract_features(tweet, freqs);
    sigmoid(feature.dot(theta)).last().unwrap().to_owned()
}
/// Input: 
///     test_x: a list of tweets
///     test_y: (m, 1) vector with the corresponding labels for the list of tweets
///     freqs: a dictionary with the frequency of each pair (or tuple)
///     theta: weight vector of dimension (3, 1)
/// Output: 
///     accuracy: (# of tweets classified correctly) / (total # of tweets)
pub fn test_logistic_regression(
    test_x: Vec<String>, 
    test_y: Vec<i32>, 
    freqs: &HashMap<(String, i32), i32>, 
    theta: &Array2<FType>
) -> FType {
    let mut y_hat: Vec<i32> = Vec::new();
    for tweet in test_x.iter() {
        let prediction = predict_tweet(tweet, freqs, theta);
        if prediction > 0.5 {
            y_hat.push(1);
        } else {
            y_hat.push(0);
        }
    }
    y_hat.iter().zip(&test_y).filter(|(a,b)| *a==*b).count() as FType / test_x.len() as FType
}
/// Return train_x, train_y, test_x, test_y datasets from twitter_samples
pub fn twitter_datasets() -> (Vec<String>, Vec<i32>, Vec<String>, Vec<i32>) {
    let mut all_positive_tweets = read_tweets("datasets/twitter_samples/positive_tweets.json");
    let mut all_negative_tweets = read_tweets("datasets/twitter_samples/negative_tweets.json");
    let test_x = [all_positive_tweets.split_off(4000), all_negative_tweets.split_off(4000)].concat();
    let train_x = [all_positive_tweets, all_negative_tweets].concat();
    let train_y = [vec![1; 4000], vec![0; 4000]].concat();
    let test_y = [vec![1; 1000], vec![0; 1000]].concat();
    (train_x, train_y, test_x, test_y)
}
/// Input:
///     freqs: dictionary from (word, label) to how often the word appears
///     texts: a list of text
///     labels: a list of labels correponding to the text (0,1)
/// Output:
///     logprior: the log prior.
///     loglikelihood: the log likelihood of you Naive bayes equation.
pub fn train_naive_bayes(freqs: &HashMap<(String, i32), i32>, labels: Vec<i32>) -> (FType, HashMap<String, FType>) {
    let mut loglikelihood: HashMap<String, FType> = HashMap::new();
    let mut vocab: HashSet<String> = HashSet::new();
    let mut total_word_pos: FType = 0. ;
    let mut total_word_neg: FType = 0. ;
    for pair in freqs.keys() {
        let (word, label) = pair;
        vocab.insert(word.to_string());
        if *label > 0 {
            total_word_pos += *freqs.get(pair).unwrap() as FType;
        } else {
            total_word_neg += *freqs.get(pair).unwrap() as FType;
        }
    }
    let total_uniq_word = vocab.len() as FType;
    let mut text_pos: FType = 0. ;
    let mut text_neg: FType = 0. ;
    for label in labels {
        if label == 1 { text_pos += 1. } else { text_neg += 1. }
    }
    let logprior = text_pos.ln() - text_neg.ln();
    for word in vocab {
        let freq_word_pos = freqs.get(&(word.clone(), 1)).or(Some(&0)).unwrap();
        let freq_word_neg = freqs.get(&(word.clone(), 0)).or(Some(&0)).unwrap();
        let prob_word_pos = (freq_word_pos + 1) as FType / (total_word_pos + total_uniq_word);
        let prob_word_neg = (freq_word_neg + 1) as FType / (total_word_neg + total_uniq_word);
        loglikelihood.insert(word, (prob_word_pos/prob_word_neg).ln());
    }
    (logprior, loglikelihood)
}
/// Input:
///     text: a string
///     logprior: a number
///     loglikelihood: a dictionary of words mapping to numbers
/// Output:
///     the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
pub fn naive_bayes_predict(text: &str, logprior: &FType, loglikelihood: &HashMap<String, FType>) -> FType {
    let word_list = process_tweet(text, true, true, true);
    let mut sum = *logprior;
    for word in word_list {
        if loglikelihood.contains_key(&word) { sum += loglikelihood.get(&word).unwrap()}
    }
    sum
}
/// Input:
///     test_x: A list of tweets
///     test_y: the corresponding labels for the list of tweets
///     logprior: the logprior
///     loglikelihood: a dictionary with the loglikelihoods for each word
/// Output:
///     accuracy (# of tweets classified correctly)/(total # of tweets)
pub fn test_naive_bayes(texts: Vec<String>, labels: Vec<i32>, logprior: &FType, loglikelihood: &HashMap<String, FType>) -> FType {
    let labels_vec: Array2<FType>= Array::from_vec(labels).mapv(|x| x as FType).insert_axis(Axis(1));
    let mut predictions = Array2::<FType>::zeros((0, 1));
    for text in texts {
        if naive_bayes_predict(&text, logprior, loglikelihood) > 0. {
            predictions.push_row(array![1.].view()).unwrap();
        } else { predictions.push_row(array![0.].view()).unwrap() }
    }
    1. - (predictions - labels_vec).mapv(FType::abs).mean().unwrap()
}