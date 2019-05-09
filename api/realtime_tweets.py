import tweepy
import redis
import json
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from api_auth_absolute import api
from redis_dto import DTO

analyzer = SentimentIntensityAnalyzer()
r = redis.Redis(host='localhost', port=6379, db=0)


def filter_tweet(t):
    urls = re.findall(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', t)
    for url in urls:
        t = t.replace(url, '')

    if len(t) > 0 and len(
            [word for word in t.split() if len(word) > 1]):
        return t
    else:
        return ''


class MyStreamListener(tweepy.StreamListener):
    def on_data(self, raw_data):
        global r
        try:
            all_data = json.loads(raw_data)
            text = all_data["text"]
            text = filter_tweet(text)
            if text == '':
                raise Exception('Invalid tweet content')
            country_code = all_data["place"]["country_code"]
            if(country_code == ''):
                raise Exception('Country code is empty')
            data = r.get(country_code)
            analyzer = SentimentIntensityAnalyzer()
            polarity_score_compound = analyzer.polarity_scores(all_data["text"])[
                "compound"]
            if data is None:
                # neutral
                if(polarity_score_compound > -0.05 and 0.05 > polarity_score_compound):
                    dto = DTO(0, 0, 0, 0, polarity_score_compound, 1)
                elif(polarity_score_compound >= 0.05):
                    dto = DTO(polarity_score_compound, 1, 0, 0, 0, 0)
                else:
                    dto = DTO(0, 0, polarity_score_compound, 1, 0, 0)
                json_data = dto.toJSON()
                r.set(country_code, json_data)
            else:
                json_data = json.loads(r.get(country_code))
                if(polarity_score_compound > -0.05 and 0.05 > polarity_score_compound):
                    json_data["neutral_sentiment_score"] = json_data["neutral_sentiment_score"] + \
                        polarity_score_compound
                    json_data["total_neutral_sentiment"] = json_data["total_neutral_sentiment"] + 1
                elif(polarity_score_compound >= 0.05):
                    json_data["positive_sentiment_score"] = json_data["positive_sentiment_score"] + \
                        polarity_score_compound
                    json_data["total_positive_sentiment"] = json_data["total_positive_sentiment"] + 1
                else:
                    json_data["negative_sentiment_score"] = json_data["negative_sentiment_score"] + \
                        polarity_score_compound
                    json_data["total_negative_sentiment"] = json_data["total_negative_sentiment"] + 1
                r.set(country_code, json.dumps(json_data))
        except Exception as e:
            print("Error: " + str(e))

    def on_error(self, status_code):
        print(status_code)


def start_stream():
    while True:
        try:
            my_stream_listener = MyStreamListener()
            my_stream = tweepy.Stream(
                auth=api.auth, listener=my_stream_listener)
            # realtime tweets in english with bounding box set to the whole world
            my_stream.filter(languages=['en'], locations=[-180, -90, 180, 90])
        except:
            continue


start_stream()
