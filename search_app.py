import csv
import pycountry
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from api.redis_dto import DTO
from api.historical_tweets import search
from data.bounding_boxes import country_bounding_boxes

# initialize locations
locations = []
with open('data/cities.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in list(reader)[1:]:
        locations.append(
            {'label': row[0], 'value': row[4], "latitude": row[2], "longitude": row[3]})
csvFile.close()

# add external styles and initialize app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# set up app layout
app.layout = html.Div(children=[
    html.H1(children='Country Sentiment', style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(id='query-state', type='text', placeholder='Search Query'),
        dcc.Input(id='count-state', type='number', step=100,
                  min=100, max=1000, placeholder='Count'),
        dcc.Dropdown(
            id='location-state',
            options=locations,
            placeholder="Location",
            style={'width': '310px', 'margin': 'auto', 'textAlign': 'left'}
        ),
        html.Button(id='submit-button',
                    children='Analyze', style={'padding': '0px 20px', 'color': 'white', 'backgroundColor': '#3389ea', 'marginTop': '10px'}),

    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='output-state',
    ),
])

# make a new request on each button click with
# corresponding count, query and location


@app.callback(Output('output-state', 'figure'),
              [Input('submit-button', 'n_clicks')],
              [State('query-state', 'value'),
               State('count-state', 'value'),
               State('location-state', 'value')])
def update_graph_data(n_clicks, query, count, location):
    country = [country for country in locations if country["value"] == location][0]
    latitude = country["latitude"]
    longitude = country["longitude"]
    dto = DTO(0, 0, 0, 0, 0, 0)
    results = search(query=query, num=count,
                     geocode="{},{},50mi".format(latitude, longitude))
    analyzer = SentimentIntensityAnalyzer()

    for val in results:
        polarity_score_compound = analyzer.polarity_scores(val.text)[
            "compound"]
        if -0.5 < polarity_score_compound < 0.5:
            dto.neutral_sentiment_score = dto.neutral_sentiment_score + polarity_score_compound
            dto.total_neutral_sentiment = dto.total_neutral_sentiment + 1
        elif polarity_score_compound >= 0.5:
            dto.positive_sentiment_score = dto.positive_sentiment_score + polarity_score_compound
            dto.total_positive_sentiment = dto.total_positive_sentiment + 1
        else:
            dto.negative_sentiment_score = dto.negative_sentiment_score + polarity_score_compound
            dto.total_negative_sentiment = dto.total_negative_sentiment + 1

    tweet_count = len(results)
    total_score = dto.positive_sentiment_score + \
        dto.negative_sentiment_score + \
        dto.neutral_sentiment_score
    avg_polarity_score = round(total_score/tweet_count, 4)

    avg_positive_score = round(dto.positive_sentiment_score /
                               dto.total_positive_sentiment, 4) if dto.total_positive_sentiment != 0 else 0
    avg_negative_score = round(dto.negative_sentiment_score /
                               dto.total_negative_sentiment, 4) if dto.total_negative_sentiment != 0 else 0
    avg_neutral_score = round(dto.neutral_sentiment_score /
                              dto.total_neutral_sentiment, 4) if dto.total_neutral_sentiment != 0 else 0

    bounding_box = country_bounding_boxes[location][1]
    latrange = [bounding_box[1], bounding_box[3]]
    lonrange = [bounding_box[0], bounding_box[2]]

    figure = {
        'data': [
            go.Choropleth(
                locations=[pycountry.countries.get(alpha_2=location).alpha_3],
                z=[avg_polarity_score],
                zmin=-1.0,
                zmax=1.0,
                text=(country["label"] +
                      "<br>Tweet Count: " + str(tweet_count) +
                      "<br>Average Positive Score: " + str(avg_positive_score) +
                      "<br>Average Negative Score: " + str(avg_negative_score) +
                      "<br>Average Neutral Score: " +
                      str(avg_neutral_score),
                      ),
                reversescale=True,
                colorbar=go.choropleth.ColorBar(
                    title='Sentiment Score'),
            )
        ],
        'layout': go.Layout(
            title=go.layout.Title(
                text='Search Sentiment Scores'
            ),
            geo=go.layout.Geo(
                showframe=False,
                bgcolor='rgba(255, 255, 255, 0.0)',
                countrywidth=3,

                projection=go.layout.geo.Projection(
                    type='equirectangular'
                ),
                lonaxis=go.layout.geo.Lonaxis(
                    range=lonrange
                ),
                lataxis=go.layout.geo.Lataxis(
                    range=latrange
                ),
            ),
        )
    }

    return figure


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
