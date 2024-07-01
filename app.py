from flask import Flask, request, jsonify, render_template
import tweepy
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
import re
import string
import time
import logging

nltk.download('stopwords')
nltk.download('punkt')

# Twitter API credentials
ckey = '5PPEWvduV7xM9TU6tl1FLUjjI'
csecret = 'MTp8s2ShcJOmzHSBac0eMx726jCI6R5TwxiowHt1dpLOMM0PKQ'
atoken = '1766163812120469504-rmSvh95LjJSGJLHJeYEszKcZzD1Jnd'
asecret = 'omqxJUPi0jyU227QIjClmZOsSGzfz85HTxyK7egMOOoQ9'

# Initialize Tweepy API
auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

app = Flask(__name__)

# Preprocessing functions
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def preproc(s):
    s = unidecode(s)
    POSTagger = preprocess(s)
    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in POSTagger if w not in stop_words]
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = [i for i in preProcessed if 'http' not in i and not i.isdigit()]
    temp1 = ' '.join(c for c in final)
    return temp1

def getTweets(user):
    tweets = []
    max_id = None
    try:
        for _ in range(4):  # Adjust the range as needed
            while True:
                try:
                    if max_id:
                        fetched_tweets = api.user_timeline(screen_name=user, count=200, include_rts=True, tweet_mode="extended", max_id=max_id)
                    else:
                        fetched_tweets = api.user_timeline(screen_name=user, count=200, include_rts=True, tweet_mode="extended")
                    if not fetched_tweets:
                        break
                    max_id = fetched_tweets[-1].id - 1
                    break  # Break the loop if the request is successful
                except tweepy.errors.RateLimitError:
                    logger.warning("Rate limit reached. Sleeping for 15 minutes.")
                    time.sleep(15 * 60)
                except tweepy.errors.TweepyException as e:
                    logger.error(f"Error fetching tweets: {e}")
                    return []
            for status in fetched_tweets:
                tw = preproc(status.full_text)
                if tw.find(" ") == -1:
                    tw = "blank"
                tweets.append(tw)
    except Exception as e:
        print("Failed to run the command on that user, Skipping...", e)
    return tweets

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form['username']
    tweets = getTweets(username)
    
    if not tweets:
        return jsonify({'error': 'Could not fetch tweets for this user or user has no tweets'})
    
    # Load frequency dictionary
    with open('newfrequency300.csv', 'rt') as f:
        csvReader = csv.reader(f)
        mydict = {rows[1]: int(rows[0]) for rows in csvReader}
    
    vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
    x = vectorizer.fit_transform(tweets).toarray()
    df = pd.DataFrame(x)
    
    model_IE = pickle.load(open("BNIEFinal.sav", 'rb'))
    model_SN = pickle.load(open("BNSNFinal.sav", 'rb'))
    model_TF = pickle.load(open('BNTFFinal.sav', 'rb'))
    model_PJ = pickle.load(open('BNPJFinal.sav', 'rb'))
    
    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)
    
    b = Counter(IE)
    value = b.most_common(1)
    answer.append("I" if value[0][0] == 1.0 else "E")
    
    b = Counter(SN)
    value = b.most_common(1)
    answer.append("S" if value[0][0] == 1.0 else "N")
    
    b = Counter(TF)
    value = b.most_common(1)
    answer.append("T" if value[0][0] == 1 else "F")
    
    b = Counter(PJ)
    value = b.most_common(1)
    answer.append("P" if value[0][0] == 1 else "J")
    
    mbti = "".join(answer)
    return jsonify({'mbti': mbti})

if __name__ == '__main__':
    app.run(debug=True)
