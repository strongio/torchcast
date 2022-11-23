from datetime import datetime
from typing import List
from dash import Dash, dcc, html, Input, Output
from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np


app = Dash(__name__)

DATA_DIR = Path("docs/examples/electricity").resolve()


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df: pd.DataFrame = pd.read_csv(DATA_DIR / "df_electricity.csv.gz")
all_groups: np.ndarray = df['group'].unique()
regular_groups: pd.DataFrame = pd.read_parquet(DATA_DIR / 'train_groups.pq')


app.layout = html.Div(
    [
        # col 1
        html.Div(children=[
            html.H1(children='Electricity Use Forecasting',
                    style={'textAlign': 'center',}),

            html.Div(children='Investigating at the electricity use of 370 clients from 2011 to 2014',
                     style={'textAlign': 'center',}),

            dcc.Graph(id="time-series-chart"),

            # the horizontal part with 3parts
            html.Div(children=[
                # Part 1
                html.Div(
                    dcc.Graph(id="histogram-chart"),
                    style={'padding': 10, 'flex': 1}),

                # Part 2
                html.Div(id="correlation-div",),

                # Part 3
                html.Div(
                    children=[
                        html.H6("Select ML model:"),
                        dcc.RadioItems(
                            id="prediction_toggle",
                            options=[
                                {"label": "No Predictions", "value": "none"},
                                {"label": "Exponential Smoothing", "value": "es"},
                                {"label": "Neural-Network", "value": "nn"}
                            ]
                        ),
                        html.H6("Time"),
                        dcc.Checklist(
                            id="day_night_toggle",
                            options=[
                                {"label": "Day", "value": "day"},
                                {"label": "Night", "value": "night"}
                            ],
                            value=["day", "night"],
                            inline=True,
                        ),
                        html.H6("Day"),
                        dcc.Checklist(
                            id="day_of_week_toggle",
                            options=[
                                {"label": "Weekdays", "value": "weekdays"},
                                {"label": "Weekends", "value": "weekends"},
                            ],
                            value=["weekdays", "weekends"],
                            inline=True,
                        )
                    ],
                    style={'padding': 10, 'flex': 1, 'align-items': 'center', 'justify-content': 'center'}),
            ], style={'display': 'flex', 'flex-direction': 'row'}
            )
        ], style={'padding': 10, 'flex': 2.5}),

        # col 2
        html.Div(children=[
            html.Div(children=[
                html.H6("Drop irregular clients:", style={"margin-top": "100px"}),
                dcc.RadioItems(id="drop_irregular",
                            options=[{"label": "No", "value": False},
                                        {"label": "Yes", "value": True}],
                        value=False, inline=True),
            ]),

            html.Div(children=[
                html.H6("Clients (max 2)"),
                # dcc.Dropdown(
                #     id="group_dropdown",
                #     options=df['group'].unique(),
                #     value="MT_001",
                #     clearable=False,
                # ),
                dcc.Checklist(
                    id="group_checklist",
                    options=all_groups,
                    value=["MT_001"],
                    style={"max-height": "250px", "overflow-y": "auto"},
                ),
            ]),

        ], style={'padding': 10, 'flex': 1}),
    ],
    style={'display': 'flex', 'flex-direction': 'row'}
)


# --- plotting
# main plot
@app.callback(
    Output("time-series-chart", "figure"),
    Output("histogram-chart", "figure"),
    Input("group_checklist", "value")
)
def display_time_series(groups: List[str]):
    _df = df.loc[df['group'].isin(groups)]

    fig_ts = px.line(_df, x='time', y='kW', color='group')
    fig_ts.add_vline(x=datetime(2013, 6, 1), line_width=3, line_dash="dash", line_color="green")

    fig_hist = px.histogram(_df, x='kW', color='group', opacity=0.8, histnorm='probability density')

    fig_ts.update_layout(
        plot_bgcolor='#DDDDDD',
        paper_bgcolor='#EEEEEE',
        font_color='#000000',
        font_family="Courier New",
        title_font_family="Courier New",
    )
    fig_hist\
        .update_layout(
            plot_bgcolor='#DDDDDD',
            paper_bgcolor='#EEEEEE',
            font_color='#000000',
            font_family="Courier New",
            title_font_family="Courier New",
            margin={"autoexpand": True})\
        .update_yaxes(visible=False)

    fig_hist.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    return fig_ts, fig_hist

# Correlation plot
@app.callback(
    Output("correlation-div", "children"),
    Output("correlation-div", "style"),
    Input("group_checklist", "value")
)
def correlation_plot(group_checklist_values: List[str]):
    if len(group_checklist_values) == 2:
        group1: pd.DataFrame = df.loc[df['group'] == group_checklist_values[0]]
        group2: pd.DataFrame = df.loc[df['group'] == group_checklist_values[1]]
        groups_merged: pd.DataFrame = group1.merge(
            group2,
            on=["time"],
            validate="1:1",
            how="inner",
        )
        fig_corr = px.density_heatmap(
            data_frame=groups_merged,
            x="kW_x",
            y="kW_y",
            labels={
                "kW_x": f"{group_checklist_values[0]} Power Use (kW)",
                "kW_y": f"{group_checklist_values[1]} Power Use (kW)",
            }
        )
        fig_corr\
            .update_layout(
                plot_bgcolor='#DDDDDD',
                paper_bgcolor='#EEEEEE',
                font_color='#000000',
                font_family="Courier New",
                title_font_family="Courier New",
                margin={"autoexpand": True})\
            .update_coloraxes(showscale=False)

        children=[
            dcc.Graph(id="correlation-chart", figure=fig_corr)
        ]
        style={'padding': 10, 'flex': 1 }
    else:
        children = [
            html.P(children="Select two clients to show correlation"),
        ]
        style={'padding': 10, 'display': 'flex', 'flex': 1, 'align-items': 'center', 'justify-content': 'center'}
    return children, style


# --- Station selection
# limit the number of options
_max_selected = 2
@app.callback(
    Output("group_checklist", "options"),
    Input("group_checklist", "value"),
    Input("drop_irregular", "value"),
)
def update_multi_options(groups_selected: List[str], drop_irregular: bool):
    options = all_groups
    if drop_irregular:
        options = regular_groups["train_groups"].to_list()

    if len(groups_selected) >= _max_selected:
        options = [
            {
                "label": option,
                "value": option,
                "disabled": option not in groups_selected,
            }
            for option in options
        ]
    return options

if __name__ == "__main__":
   app.run_server(debug=True)
