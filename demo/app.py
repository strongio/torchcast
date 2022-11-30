from typing import Dict, List
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
from dash import Dash, dcc, html, Input, Output
from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S',
)


app = Dash(__name__)

DATA_DIR = Path("docs/examples/electricity").resolve()
train_val_split_dt: pd.Timestamp = pd.Timestamp("2013-06-01")


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv(DATA_DIR / "df_electricity.csv.gz", parse_dates=["time"])
nn_df: pd.DataFrame = pd.read_parquet(DATA_DIR / "df_forecast_nn.pq")
# the nn_df and exponential smoothing dfs have sqrt and centered data.
df["kW_sqrt"] = np.sqrt(df["kW"])
centerings = (
    nn_df
    .sort_values("time")
    .groupby("group")
    .head(1)
    .filter(["actual", "time", "group"])
    .merge(df)
    .set_index("group")
    .pipe(
        lambda frame:
        frame["kW_sqrt"] - frame["actual"]
    )
)

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
                    src="assets/strong-logo-white.svg",
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
                            className="card main-card",
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
                                    className="card",
                                    style={"flex": "0 0 323px"}
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
                                html.H5("ML model"),
                                dcc.RadioItems(
                                    id="prediction_toggle",
                                    options=[
                                        {"label": "None", "value": "none"},
                                        {"label": "Exponential Smoothing", "value": "es",},
                                        {"label": "Neural-Network", "value": "nn"},
                                    ],
                                    value="none",
                                    className="form-item",
                                ),
                                html.H5(" "),
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
                                html.H5(" "),
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
                                    style={"max-height": "190px", "overflow-y": "scroll", "scrollbar-color": "dark"},
                                ),
                            ],
                        ),
                    ],
                    style={"flex": "0 0 250px", "margin-left": "30px"},
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


@lru_cache
def get_es_prediction_df(group: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(DATA_DIR / f"es_{group}_2.pq")
    except FileNotFoundError:
        print(f"Couldn't find the es predictions for {group}")
        return None

@lru_cache
def get_combined_df(group: str) -> pd.DataFrame:
    es_prediction_df_subset: pd.DataFrame = (
        get_es_prediction_df(group)
        .assign(ES=lambda df: (df["mean"] + centerings.at[group]).pow(2))
        .filter(["time", "ES"])
    )
    nn_df_subset: pd.DataFrame = (
        nn_df
        .query(f"group == '{group}'")
        .assign(NN=lambda df: (df["mean"] + centerings.at[group]).pow(2))
        .filter(["time", "NN"])
    )
    original_data_subset = (
        df
        .query(f"group == '{group}'")
        .assign(actual = lambda df: df["kW"])
        .filter(["group", "time", "actual"])
    )
    combined = (
        original_data_subset
        .merge(es_prediction_df_subset, how="outer")
        .merge(nn_df_subset, how="outer")
        .assign(
            is_train = lambda df: df["time"] < train_val_split_dt
        )
        .melt(
            value_vars=["actual", "ES", "NN"],
            id_vars=["group", "time", "is_train"],
            value_name="kW",
            var_name="model",
        )
    )
    logging.info(combined)
    return combined

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

    _df = pd.concat(
        [pd.DataFrame(columns=["group", "time", "is_train", "model", "kW"])]
        + [get_combined_df(group) for group in groups]
    )

    ts_fig_height_px = 400

    if prediction_toggle_value == "none":
        fig_ts = px.line(
            _df.query("model == 'actual'"),
            x='time', y='kW',
            color='group', **strong_color_cycle,
            height=ts_fig_height_px,
        )
    elif prediction_toggle_value == "es":
        # do the plotting
        fig_ts = px.line(
            _df.query("(model == 'actual') | ((model == 'ES') and (is_train == False))"),
            x='time', y='kW',
            color='group', **strong_color_cycle,
            line_dash='model',
            height=ts_fig_height_px
        )
    else:
        fig_ts = px.line(
            _df.query("(model == 'actual') | ((model == 'NN') & (is_train == False))"),
            x='time', y='kW',
            color='group', **strong_color_cycle,
            line_dash='model',
            height=ts_fig_height_px)

    # Add the vertical line between train-val split
    fig_ts.add_vline(x=datetime(2013, 6, 1), line_width=3, line_dash="dash", line_color="white")

    fig_hist = px.histogram(
        _df.query("model == 'actual'"),
        x='kW',
        nbins=80,
        color='group', **strong_color_cycle,
        opacity=0.8,
        histnorm='probability density',
        height=275
    )

    # Styling of time series
    fig_ts \
        .update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ]),
                    font=dict(
                        color="#111"
                    )
                ),
                rangeslider=dict(
                    visible=True
                ),
                range=(train_val_split_dt - pd.Timedelta("7D"),
                       train_val_split_dt - pd.Timedelta("7D") + pd.Timedelta("30D")),
                type="date"
            )
        ) \
        .update_layout(
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
        ) \
        .update_xaxes(showgrid=False) \
        .update_yaxes(showgrid=False) \
        .update_traces(line=dict(width=1.0))

    # Update looks of histogram
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
            )) \
        .update_yaxes(visible=False) \
        .update_layout(
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
            width=323, height=275
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
        style={ "flex": "0 0 323px"}
    else:
        children = [
            html.P(children="Select two clients to show correlation"),
        ]
        style={ "flex": "0 0 323px", "text-align": "center", "padding-top": "150px"}
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
   app.run_server(
       debug=True,
       host="andys-macbook-pro",
       port=80,
   )
