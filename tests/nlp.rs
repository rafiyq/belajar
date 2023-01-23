use std::collections::HashMap;
use belajar::nlp::nlp::{process_tweet, build_freqs};

macro_rules! str {
    ($s:expr) => {
        String::from($s)
    };
}
#[test]
fn process_tweet_test_1() {
    let left = process_tweet("@dotlung indeed :-)", true, true, true);
    let right: Vec<String> = vec![String::from(r"inde"), String::from(r":-)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_test_2() {
    let left = process_tweet("Just passing by :) 🚂 (@ Dewsbury Railway Station (DEW) - @nationalrailenq in Dewsbury, West Yorkshire) https://t.co/DvBssHbrfx", true, true, true);
    let right: Vec<String> = vec![str!("pass"), str!(":)"), str!("🚂"), str!("dewsburi"), str!("railwai"), str!("station"), str!("dew"), str!("dewsburi"), str!("west"), str!("yorkshir")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_test_3() {
    let left = process_tweet("@CV_UHB the patients. Children crying in x ray dept...not to leave lol. Must say its a lovely, very pleasant hospital. Have a grt weekend :)", true, true, true);
    let right: Vec<String> = vec![str!("patient"), str!("children"), str!("cry"), str!("x"), str!("rai"), str!("dept"), str!("..."), str!("leav"), str!("lol"), str!("must"), str!("sai"), str!("love"), str!("pleasant"), str!("hospit"), str!("grt"), str!("weekend"), str!(":)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_test_4() {
    let left = process_tweet("@_nicapapa  follow @jnlazts &amp; http://t.co/RCvcYYO0Iq follow u back :)", true, true, true);
    let right: Vec<String> = vec![str!("follow"), str!("follow"), str!("u"), str!("back"), str!(":)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_test_5() {
    let left = process_tweet("@lawrenceispichu oh gosh what did you say? And aw hun :( *cuddles*", true, true, true);
    let right: Vec<String> = vec![str!("oh"), str!("gosh"), str!("sai"), str!("aw"), str!("hun"), str!(":("), str!("cuddl")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_test_6() {
    let left = process_tweet("No comment aing :(", true, true, true);
    let right: Vec<String> = vec![str!("comment"), str!("a"), str!(":(")];
    assert_eq!(left, right);
}

#[test]
fn build_freq_default_test() {
    let freqs: HashMap<(String, i32), i32> = HashMap::new();
    let tweets = vec![
        str!("i am happy"),
        str!("i am tricked"),
        str!("i am sad"),
        str!("i am tired"),
        str!("i am tired"),

    ];
    let labels = vec![1, 0, 0, 0, 0];
    let expected = HashMap::from([
        ((str!("happi"), 1), 1),
        ((str!("trick"), 0), 1),
        ((str!("sad"), 0), 1),
        ((str!("tire"), 0), 2),
    ]);
    assert_eq!(build_freqs(freqs, &tweets, &labels), expected);
}
#[test]
fn build_freq_larger_test() {
    let freqs: HashMap<(String, i32), i32> = HashMap::new();
    let tweets = vec![
        str!("i am happy"),
        str!("i am tricked"),
        str!("i am sad"),
        str!("i am tired"),
        str!("i am tired but proud today"),
        str!("i am you are"),
        str!("you are happy"),
        str!("he was sad"),

    ];
    let labels = vec![1, 0, 0, 0, 1, 0, 1, 0];
    let expected = HashMap::from([
                ((str!("happi"), 1), 2),
                ((str!("trick"), 0), 1),
                ((str!("sad"), 0), 2),
                ((str!("tire"), 0), 1),
                ((str!("tire"), 1), 1),
                ((str!("proud"), 1), 1),
                ((str!("todai"), 1), 1),
    ]);
    assert_eq!(build_freqs(freqs, &tweets, &labels), expected);
}
#[test]
fn build_freq_noempty_dict_test() {
    let freqs = HashMap::from([
        ((str!("happi"), 1), 3),
        ((str!("sad"), 0), 1),
        ((str!("tire"), 0), 2),
        ((str!("tire"), 1), 1),
    ]);
    let tweets = vec![
        str!("i am happy"),
        str!("i am tricked"),
        str!("i am sad"),
        str!("i am tired"),
        str!("i am tired but proud today"),
        str!("i am you are"),
        str!("you are happy"),
        str!("he was sad"),

    ];
    let labels = vec![1, 0, 0, 0, 1, 0, 1, 0];
    let expected = HashMap::from([
                ((str!("happi"), 1), 5),
                ((str!("trick"), 0), 1),
                ((str!("sad"), 0), 3),
                ((str!("tire"), 0), 3),
                ((str!("tire"), 1), 2),
                ((str!("proud"), 1), 1),
                ((str!("todai"), 1), 1),
    ]);
    assert_eq!(build_freqs(freqs, &tweets, &labels), expected);
}