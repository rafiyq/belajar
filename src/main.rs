use belajar::nlp::{process_tweet, read_tweets};

fn main() {
    println!("Hello");

    let _tweets = read_tweets("datasets/twitter_samples/positive_tweets.json");
    let tweet = String::from("@staybubbly69 as +14158586273 Loveeeee Matt would say. WELCOME TO ADULTHOOD.... :) http://t.co/zHQy0iyaCP");
    println!("{}", tweet);
    let clean_tweet = process_tweet(tweet.as_ref(), true, true, true);
    println!("{:?}", clean_tweet);

}