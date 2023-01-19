use belajar::nlp::process_tweet;

macro_rules! str {
    ($s:expr) => {
        String::from($s)
    };
}
#[test]
fn process_tweet_tests_1() {
    let left = process_tweet("@dotlung indeed :-)", true, true, true);
    let right: Vec<String> = vec![String::from(r"inde"), String::from(r":-)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_tests_2() {
    let left = process_tweet("Just passing by :) 🚂 (@ Dewsbury Railway Station (DEW) - @nationalrailenq in Dewsbury, West Yorkshire) https://t.co/DvBssHbrfx", true, true, true);
    let right: Vec<String> = vec![str!("pass"), str!(":)"), str!("🚂"), str!("dewsburi"), str!("railwai"), str!("station"), str!("dew"), str!("dewsburi"), str!("west"), str!("yorkshir")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_tests_3() {
    let left = process_tweet("@CV_UHB the patients. Children crying in x ray dept...not to leave lol. Must say its a lovely, very pleasant hospital. Have a grt weekend :)", true, true, true);
    let right: Vec<String> = vec![str!("patient"), str!("children"), str!("cry"), str!("x"), str!("rai"), str!("dept"), str!("..."), str!("leav"), str!("lol"), str!("must"), str!("sai"), str!("love"), str!("pleasant"), str!("hospit"), str!("grt"), str!("weekend"), str!(":)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_tests_4() {
    let left = process_tweet("@_nicapapa  follow @jnlazts &amp; http://t.co/RCvcYYO0Iq follow u back :)", true, true, true);
    let right: Vec<String> = vec![str!("follow"), str!("follow"), str!("u"), str!("back"), str!(":)")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_tests_5() {
    let left = process_tweet("@lawrenceispichu oh gosh what did you say? And aw hun :( *cuddles*", true, true, true);
    let right: Vec<String> = vec![str!("oh"), str!("gosh"), str!("sai"), str!("aw"), str!("hun"), str!(":("), str!("cuddl")];
    assert_eq!(left, right);
}
#[test]
fn process_tweet_tests_6() {
    let left = process_tweet("No comment aing :(", true, true, true);
    let right: Vec<String> = vec![str!("comment"), str!("a"), str!(":(")];
    assert_eq!(left, right);
}