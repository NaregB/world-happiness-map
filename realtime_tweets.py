import tweepy
import pycountry
from api_auth import api


class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print("Country", pycountry.countries.get(
            alpha_2=status.place.country_code).name)

    def on_error(self, status_code):
        print(status_code)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

# realtime tweets in english with bounding box set to the whole world
myStream.filter(languages=['en'], locations=[-180, -90, 180, 90])
