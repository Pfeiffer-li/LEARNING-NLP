import re
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    # the same as the processing of preprocess.py
    """Process Tweet Function
    Input:
        tweet: a string containing a tweet
    Output:
        tweet_clean: a list of word contain the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash and sign from the word
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweet_clean.append(stem_word)

    return tweet_clean


def build_freqs(tweets, ys):
    """Build Frequency Dictionary.
    Input:
        tweets: a list of tweet
        ys: an m x 1 with the sentiment array label of each tweet

    Output:
        freqs: a dictionary mapping each (word, frequency) pair to its frequency
    """
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs


if __name__ == "__main__":
    # Prepare Some Datas.
    # nltk.download('twitter_samples')
    # nltk.download('stopwords')
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    tweets = all_positive_tweets + all_negative_tweets
    print("Number of tweets: ", len(tweets))

    # The labels are used to build frequency dictionary.
    labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))
    freqs = build_freqs(tweets, labels)
    print(f"the length of freqs: {len(freqs)}")
    # print(freqs)

    # take example: I'm Happy, because I love you.
    # You can directly select multiple words in freqs to form a list for demonstration
    example = "I'm Happy, because I love you."
    processed_example = process_tweet(example)
    print(processed_example)
    data = []
    for word in processed_example:
        pos, neg = 0, 0
        if (word, 1) in freqs:
            pos = freqs[(word, 1)]
        if (word, 0) in freqs:
            pos = freqs[(word, 0)]
        data.append([word, pos, neg])
    print(data)

    # Visualized display
    fig, ax = plt.subplots(figsize=(8, 8))
    x = np.log([x[1] + 1 for x in data])
    y = np.log([x[2] + 1 for x in data])
    ax.scatter(x, y)
    plt.xlabel("Log Positive count")
    plt.ylabel("Log Negative count")
    for i in range(0, len(data)):
        ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)
    ax.plot([0, 9], [0, 9], color='red')
    plt.show()
    # You can obviously see that these words are on the positive side, so you can roughly judge that this sentence is a positive emotion