use std::fs;
use serde_json::Value;

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