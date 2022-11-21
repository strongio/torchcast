from dash import Dash, dcc, html, Input, Output
from pathlib import Path
import plotly.express as px
import pandas as pd


app = Dash(__name__)

DATA_DIR = Path("docs/examples/electricity").resolve()


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv(DATA_DIR / "df_electricity.csv.gz")


app.layout = html.Div([
    html.Div(children=[
        html.H1(
            children='Electricity Use Forecasting',
            style={
                'textAlign': 'center',
            }
        ),

        html.Div(children='Investigating at the electricity use of 370 clients from 2011 to 2014', style={
            'textAlign': 'center',
        }),

        dcc.Graph(id="time-series-chart"),
        html.P("Select client:"),
        dcc.Dropdown(
            id="group",
            options=df['group'].unique(),
            value="MT_001",
            clearable=False,
        ),
    ]),
])

@app.callback(
    Output("time-series-chart", "figure"),
    Input("group", "value"))
def display_time_series(group):
    _df = df.loc[df["group"] == group]
    fig = px.line(_df, x='time', y='kW')

    fig.update_layout(
        plot_bgcolor='#DDDDDD',
        paper_bgcolor='#EEEEEE',
        font_color='#7FDBFF',
        font_family="Courier New",
        title_font_family="Courier New",
    )

    return fig

if __name__ == "__main__":
   app.run_server(debug=True)
