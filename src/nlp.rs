use std::{fs, collections::HashMap, iter::zip};
use html_escape::decode_html_entities;
use serde_json::Value;
use lazy_static::lazy_static;
use regex::Regex;

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
/// TODO: Make regex expand to multiline!
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
    lazy_static! {
        static ref SEMTIMEN: Vec<Regex> = vec![
            Regex::new(r"\$\w*").unwrap(),                  // stock market tickers like $GE
            Regex::new(r"^RT[\s]+").unwrap(),               // old style retweet text "RT"
            Regex::new(r"https?://[^\s\n\r]+").unwrap(),    // hyperlinks
            Regex::new(r"#").unwrap(),                      // the hash # sign from the word
        ];
        static ref HANDLES: Vec<fancy_regex::Regex> = vec![ // Twitter username handles from text.
            fc_regex!(r"(?<![A-Za-z0-9_!@#\$%&*])@"),
            fc_regex!(r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))")
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
    println!("{}", clean_tweet);

    // Extract tokens from tweet
    let tweet_tokens: Vec<&str> = REGEXPS.find_iter(&clean_tweet).map(|mat| mat.as_str()).collect();
    println!("{:?}", tweet_tokens);

    let mut tweets_clean :Vec<String>= Vec::new();
    let stopwords_english = stopwords("english");
    for word in tweet_tokens {
    println!("{}", word);
    if !stopwords_english.iter().any(|i| i==word) && 
        word.len() > 1 || !(word.chars().nth(0).unwrap() as u8).is_ascii_punctuation() {
            if EMOTICONS.is_match(word) {
                tweets_clean.push(stem::get(word).unwrap())
            } else {
                tweets_clean.push(stem::get(word).unwrap().to_ascii_lowercase());
            }
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
#[derive(Eq, Hash, PartialEq)]
pub struct Pair(String, i32);
/// Build frequencies.
/// Input:
///     tweets: a list of tweets
///     label: an m x 1 array with the sentiment label of each tweet
///     (either 0 or 1)
/// Output:
///     freqs: a dictionary mapping each (word, sentiment) pair to its
///     frequency
pub fn build_freqs(tweets: Vec<String>, labels: Vec<i32>) -> HashMap<Pair, i32> {
    let mut freqs: HashMap<Pair, i32> = HashMap::new();
    for (label, tweet) in zip(labels, tweets) {
        for word in process_tweet(tweet.as_ref(), true, true, true) {
            let pair = Pair(word, label);
            freqs.entry(pair).or_insert(1);
        }
    }
    freqs
}