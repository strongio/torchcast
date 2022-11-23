from typing import Dict, List
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
from dash import Dash, dcc, html, Input, Output
from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np


app = Dash(__name__)

DATA_DIR = Path("docs/examples/electricity").resolve()


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv(DATA_DIR / "df_electricity.csv.gz")
nn_df: pd.DataFrame = pd.read_parquet(DATA_DIR / "df_forecast_nn.pq")
all_groups: np.ndarray = df['group'].unique()
regular_groups = pd.read_parquet(DATA_DIR / "train_groups.pq").squeeze().to_list()
strong_color_cycle: Dict[str, str] = dict(color_discrete_sequence=["#F4F4F4", "#B5C3FF"])


app.layout = html.Div(
    [
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H1(
                            children="Electricity Use Forecasting",
                            style={
                                "textAlign": "left",
                            },
                        ),
                        html.Div(
                            children="Investigating the electricity use of 370 clients in the grid.",
                            style={"textAlign": "left", "margin-bottom": "40px"},
                        ),
                    ],
                ),
                html.Img(
                    src="assets/strong-dark-blue-background.png",
                    style={
                        "width": "127px",
                        "height": "48px",
                        "position": "absolute",
                        "right": "75px",
                        "align-self": "center",
                    },
                ),
            ],
            style={"display": "inline-flex"},
        ),
        html.Div(
            children=[
                # col 1
                html.Div(
                    children=[
                        dcc.Graph(
                            id="time-series-chart",
                            className="card",
                            style={"margin-bottom": "30px"},
                        ),
                        # the horizontal part with 2 parts
                        html.Div(
                            children=[
                                # Part 1
                                html.Div(
                                    dcc.Graph(id="histogram-chart"),
                                    style={"flex": 1, "margin-right": "30px"},
                                    className="card",
                                ),
                                # Part 2
                                html.Div(
                                    id="correlation-div",
                                    style={"width": "323px"},
                                    className="card",
                                ),
                            ],
                            style={"display": "flex", "flex-direction": "row"},
                        ),
                    ],
                    style={"flex": 2.5},
                ),
                # col 2
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H6("Select ML model:"),
                                dcc.RadioItems(
                                    id="prediction_toggle",
                                    options=[
                                        {"label": "No Predictions", "value": "none"},
                                        {"label": "Exponential Smoothing", "value": "es",},
                                        {"label": "Neural-Network", "value": "nn"},
                                    ],
                                    value="none",
                                    className="form-item",
                                ),
                                html.H6("Time"),
                                dcc.Checklist(
                                    id="day_night_toggle",
                                    options=[
                                        {"label": "Day", "value": "day"},
                                        {"label": "Night", "value": "night"},
                                    ],
                                    value=["day", "night"],
                                    inline=True,
                                    className="form-item",
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
                                    className="form-item",
                                ),
                            ],
                            style={
                                "flex": 1,
                                "align-items": "left",
                                "justify-content": "left",
                            },
                        ),
                        html.Div(
                            children=[
                                html.H6(
                                    "Drop irregular clients:",
                                ),
                                dcc.RadioItems(
                                    id="drop_irregular",
                                    options=[
                                        {"label": "No", "value": False},
                                        {"label": "Yes", "value": True},
                                    ],
                                    value=True,
                                    inline=True,
                                    className="form-item",
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H6("Clients"),
                                # dcc.Dropdown(
                                #     id="group_dropdown",
                                #     options=df['group'].unique(),
                                #     value="MT_001",
                                #     clearable=False,
                                # ),
                                dcc.Checklist(
                                    id="group_checklist",
                                    options=all_groups,
                                    value=["MT_328"],
                                    className="form-item",
                                    style={"max-height": "190px", "overflow-y": "auto"},
                                ),
                            ]
                        ),
                    ],
                    style={"flex": 1, "margin-left": "30px"},
                    className="card",
                ),
            ],
            className="bg-strong-dark-blue",
            style={"display": "flex", "flex-direction": "row"},
        ),
        html.Div(
            id="footer",
            children=['Powered by ',
                      html.A(href="https://strong.io/", children="Strong Analytics",
                             target="_blank", rel="noopener noreferrer")],
            style={"margin-top": "16px", "padding-left": "5px"},
        )
    ],
)


@lru_cache()
def get_es_prediction_df(group: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(DATA_DIR / f"es_{group}_2.pq")
    except FileNotFoundError:
        print(f"Couldn't find the es predictions for {group}")
        return None

# --- plotting
# main plot
@app.callback(
    Output("time-series-chart", "figure"),
    Output("histogram-chart", "figure"),
    Input("group_checklist", "value"),
    Input("prediction_toggle", "value"),
)
def display_time_series(
    groups: List[str],
    prediction_toggle_value: str,
):
    _df = df.loc[df['group'].isin(groups)].copy()
    train_val_split_dt = pd.Timestamp("2013-06-01")

    if prediction_toggle_value == "none":
        fig_ts = px.line(_df, x='time', y='kW', color='group', height=308, **strong_color_cycle)
    elif prediction_toggle_value == "es":
        # Load the predictions from each group
        es_predictions: List[pd.DataFrame] = [get_es_prediction_df(group) for group in groups]
        melted_dfs: List[pd.DataFrame] = [pd.DataFrame(columns=['group', 'time', 'actual_or_mean', 'kW'])]
        for es_prediction in es_predictions:
            melted = es_prediction.melt(
                value_vars=["actual", "mean"],
                id_vars=["group", "time"],
                value_name="kW",
                var_name="actual_or_mean",
            )
            melted = melted.query("(actual_or_mean == 'actual') | (time > @train_val_split_dt)")
            melted_dfs.append(
                melted
            )
        _df = pd.concat(melted_dfs)
        fig_ts = px.line(_df, x='time', y='kW', color='group', line_dash='actual_or_mean', height=308, **strong_color_cycle)
    else:
        _df = (
            nn_df
            .query("group.isin(@groups)")
            .melt(
                value_vars=["actual", "mean"],
                id_vars=["group", "time"],
                value_name="kW",
                var_name="actual_or_mean",
            )
            .query("(actual_or_mean == 'actual') | (time > @train_val_split_dt)")
        )

        fig_ts = px.line(_df, x='time', y='kW', color='group', line_dash='actual_or_mean', height=308, **strong_color_cycle)

    fig_ts.add_vline(x=datetime(2013, 6, 1), line_width=3, line_dash="dash", line_color="white")

    fig_hist = px.histogram(_df, x='kW', nbins=80, color='group', opacity=0.8, histnorm='probability density', width=493,
                            height=275, **strong_color_cycle)

    fig_ts.update_layout(
        legend=dict(
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.50,
            orientation='h',
            title="Group",
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0,
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#F4F4F4',
        font_family="Courier New",
        title_font_family="Courier New",
    )
    fig_ts.update_xaxes(showgrid=False)
    fig_ts.update_yaxes(showgrid=False)
    fig_ts.update_traces(line=dict(width=1.0))
    fig_hist\
        .update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#F4F4F4',
            font_family="Courier New",
            title_font_family="Courier New",
            margin=dict(
                l=8,
                r=8,
                b=0,
                t=10,
                pad=0,
            ))\
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
            },
            width=440, height=275
        )
        fig_corr\
            .update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FFF',
                font_family="Courier New",
                title_font_family="Courier New",
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=24,
                    pad=0,
                ),)\
            .update_coloraxes(showscale=False)

        children=[
            dcc.Graph(id="correlation-chart", figure=fig_corr)
        ]
        style={ 'flex': 1, "width": "464px"}
    else:
        children = [
            html.P(children="Select two clients to show correlation"),
        ]
        style={ 'flex': 1, "width": "464px", "text-align": "center", "padding-top": "150px"}
    return children, style


# --- Station selection
# limit the number of options
_max_selected = 2
@app.callback(
    Output("group_checklist", "options"),
    Input("group_checklist", "value"),
    Input("drop_irregular", "value"),
)
def update_multi_options(
    groups_selected: List[str],
    drop_irregular: bool,
):
    options = all_groups
    if drop_irregular:
        options = regular_groups

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
