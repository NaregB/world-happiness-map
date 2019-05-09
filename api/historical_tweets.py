import tweepy
import math
from api.api_auth import api


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

    return all_tweets


# public_tweets = search_tweets(query="genocide", num=1000)
# print("Searched", len(public_tweets), "tweets")
# for tweet in public_tweets:
#     if tweet.place:
#         print(tweet.place.country_code)
