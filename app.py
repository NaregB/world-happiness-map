import redis
import json
import pycountry
import pandas as pd

import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

r = redis.Redis(host='localhost', port=6379, db=0)
country_codes = [country.alpha_3 for country in pycountry.countries]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='World Happiness Map', style={'textAlign': 'center'}),

    html.Div(
        children=[
            dcc.Graph(
                id='example-graph',
                style={'width': '80%', 'float': 'left'},
                figure={},
            ),
            html.Div(
                id="basic-stats",
                style={'float': 'left'},
                children=[
                    html.Span("Total Tweets:")
                ]
            ),
        ]
    ),


    dcc.Interval(
        id='interval-component',
        interval=10*1000,
        n_intervals=0
    )
])


def get_data_frame():
    df = pd.DataFrame(columns=['Code', 'Country',
                               'Tweet_Count', 'Average_Sentiment_Score',
                               'Average_Positive_Score', 'Average_Negative_Score',
                               'Average_Neutral_Score'])
    df.set_index('Code')

    keys = r.keys()
    for i in range(len(keys)):
        key = keys[i].decode("utf-8")
        country_data = json.loads(r.get(key))

        country = pycountry.countries.get(alpha_2=key)

        country_name = country.name if country else ''
        country_code = country.alpha_3 if country else ''

        tweet_count = country_data['total_positive_sentiment'] + \
            country_data['total_negative_sentiment'] + \
            country_data['total_neutral_sentiment']
        total_score = country_data['positive_sentiment_score'] + \
            country_data['negative_sentiment_score'] + \
            country_data['neutral_sentiment_score']
        avg_polarity_score = round(total_score/tweet_count, 4)

        avg_positive_score = round(country_data['positive_sentiment_score'] /
                                   country_data['total_positive_sentiment'], 4) if country_data['total_positive_sentiment'] != 0 else 0
        avg_negative_score = round(country_data['negative_sentiment_score'] /
                                   country_data['total_negative_sentiment'], 4) if country_data['total_negative_sentiment'] != 0 else 0
        avg_neutral_score = round(country_data['neutral_sentiment_score'] /
                                  country_data['total_neutral_sentiment'], 4) if country_data['total_neutral_sentiment'] != 0 else 0
        df.loc[i] = [country_code, country_name,
                     tweet_count, avg_polarity_score, avg_positive_score,
                     avg_negative_score, avg_neutral_score]
    return df


@app.callback([Output('example-graph', 'figure'), Output('basic-stats', 'children')],
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    df = get_data_frame()
    figure = {
        'data': [
            go.Choropleth(
                locations=df['Code'],
                z=df['Average_Sentiment_Score'],
                zmin=-1.0,
                zmax=1.0,
                text=(df['Country'] +
                      "<br>Tweet Count: " + df['Tweet_Count'].astype(str) +
                      "<br>Average Positive Score: " + df['Average_Positive_Score'].astype(str) +
                      "<br>Average Negative Score: " + df['Average_Negative_Score'].astype(str) +
                      "<br>Average Neutral Score: " + df['Average_Neutral_Score'].astype(str)),
                reversescale=True,
                marker=go.choropleth.Marker(
                    line=go.choropleth.marker.Line(
                        color='rgb(180,180,180)',
                        width=0.5
                    )),
                colorbar=go.choropleth.ColorBar(
                    title='Sentiment Score'),
            )
        ],
        'layout': go.Layout(
            title=go.layout.Title(
                text='Realtime Sentiment Scores'
            ),
            geo=go.layout.Geo(
                showframe=False,
                showcoastlines=False,
                projection=go.layout.geo.Projection(
                    type='equirectangular'
                ),
            ),
        )
    }

    children = [
        html.Span("Tweet count: " + str(df["Tweet_Count"].sum(axis=0)))]

    return figure, children


if __name__ == '__main__':
    app.run_server(debug=True, port=8086)
