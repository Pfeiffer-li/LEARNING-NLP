import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random


# nltk.download('twitter_samples')

all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

# 查看原本这个数据集的状态
print("Number of Positive tweets: ", len(all_positive_tweets))
print("Number of Negative tweets: ", len(all_negative_tweets))

print("\nThe type of all_positive tweet is: ", type(all_positive_tweets))
print("The type of a tweet entry is: ", type(all_positive_tweets[0]))

# 用可视化工具来看这个数据集的情况
fig = plt.figure(figsize=(5, 5))
labels = 'Positive', 'Negative'
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.show()

# 随机打印积极和消极数据集中的一条数据来检测数据情况，打印出绿色是积极的，红色是消极的
print("\033[92m" + all_positive_tweets[random.randint(0, 5000)])
print("\033[91m" + all_negative_tweets[random.randint(0, 5000)] + "\n")
# 到这里面对这两条数据回顾一下预处理的5个步骤

# 选取这个足够复杂的例子来演示我们的预处理代码
tweet = all_positive_tweets[2277]
print(tweet + "\n")

# nltk.download('stopwords')
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
print('\033[92m' + tweet)

# 预处理数据 -- 清洗数据
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)
tweet2 = re.sub(r'#', '', tweet2)
print(tweet2)

# 预处理数据 -- 分词，这个步骤一般与变小写化一起操作
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)
print("\033[90m" + "Tokenized string: ", tweet_tokens)

# 预处理数据 -- 去除停顿词和标点符号
stopwords_english = stopwords.words('english')
print("\033[90m\nStop words: ", stopwords_english)
print("\033[90m\nPunctuation: ", string.punctuation)
tweet_clean = []
for word in tweet_tokens:
    if word not in stopwords_english or word not in string.punctuation:
        tweet_clean.append(word)
print("\n\033[92m" + "removed stop words and punctuation: ", tweet_clean)

# 预处理数据 -- 做词干处理
stemmer = PorterStemmer()
tweets_stem = []
for word in tweet_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)
print("\n\033[92m" + "stemmed words: ", tweets_stem)