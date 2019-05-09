import tweepy
import math
from api.api_auth import api
import re


def remove_url(str):
    urls = re.findall(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str)
    for url in urls:
        str = str.replace(url, '')
    return str


def filter_tweets(tweets):
    for i, tweet in enumerate(tweets):
        tweet.text = remove_url(tweet.text)
        tweets[i] = tweet

    def is_meaningful(t): return len(t) > 0 and len(
        [word for word in t.split() if len(word) > 1])
    filtered = [t for t in tweets if is_meaningful(t.text)]
    return filtered


def search(query, num, geocode):
    count = 100
    page_count = math.ceil(num/count)

    all_tweets = []
    since_id = 0
    for _ in range(1, page_count + 1):
        tweets = api.search(q=query, lang="en", count=count,
                            since_id=since_id, geocode=geocode)
        all_tweets = all_tweets+tweets
        since_id = tweets[-1].id

    return filter_tweets(all_tweets)
