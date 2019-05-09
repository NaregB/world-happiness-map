import json


class DTO:
    positive_sentiment_score = 0
    total_positive_sentiment = 0
    negative_sentiment_score = 0
    total_negative_sentiment = 0
    neutral_sentiment_score = 0
    total_neutral_sentiment = 0

    def __init__(self, positive_sentiment_score, total_positive_sentiment, negative_sentiment_score, totalnegativeSentiment, neutral_sentiment_score, totalneutralSentiment):
        self.positive_sentiment_score = positive_sentiment_score
        self.total_positive_sentiment = total_positive_sentiment
        self.negative_sentiment_score = negative_sentiment_score
        self.total_negative_sentiment = totalnegativeSentiment
        self.neutral_sentiment_score = neutral_sentiment_score
        self.total_neutral_sentiment = totalneutralSentiment

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)
