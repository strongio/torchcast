import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from plotly.graph_objs import Figure

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S',
)


app = Dash(__name__, eager_loading=True)
server = app.server

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
                            style={"flex": "1 1", "margin-bottom": "30px"},
                        ),
                        # the horizontal part with 2 parts
                        html.Div(
                            children=[
                                # Part 1
                                html.Div(
                                    dcc.Graph(id="histogram-chart"),
                                    style={"flex": "1 1", "margin-right": "30px"},
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
                    style={"flex": "1 1"},
                ),
                # col 2
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H6("ML model"),
                                dcc.RadioItems(
                                    id="prediction_toggle",
                                    options=[
                                        {"label": "None", "value": "none"},
                                        {"label": "Exponential Smoothing", "value": "es",},
                                        {"label": "Neural Network", "value": "nn"},
                                    ],
                                    value="none",
                                    className="form-item",
                                ),
                                daq.BooleanSwitch(id="show_future_switch", on=False, label="Show future", labelPosition="bottom"),
                                html.H5(" "),
                                html.H6("Time"),
                                dcc.RangeSlider(
                                    0, 24, 1, value=[0, 24], marks={0: "12AM", 6: "6AM", 12: "12PM", 18: "6PM", 24: "12AM"},
                                    id="time_of_day_range",
                                ),
                                html.H6("Day"),
                                dcc.Checklist(
                                    id="day_of_week_toggle",
                                    options=[
                                        {"label": label, "value": value}
                                        for label, value in zip(
                                            "MTWRFSS",
                                            range(7)
                                        )
                                    ],
                                    value=list(range(7)),
                                    inline=True,
                                    className="form-item",
                                ),
                                html.H5(" "),
                                html.H6("Clients"),
                                dcc.Dropdown(
                                    id="group_dropdown",
                                    options=all_groups,
                                    value=["MT_328"],
                                    # className="form-item",
                                    clearable=False,
                                    multi=True,
                                    # style={.Select-value-label {color: white !important;}},
                                    style={"color": "black"},
                                ),
                                # html.H6(
                                #     "Drop irregular clients",
                                # ),
                                # dcc.RadioItems(
                                #     id="drop_irregular",
                                #     options=[
                                #         {"label": "No", "value": False},
                                #         {"label": "Yes", "value": True},
                                #     ],
                                #     value=True,
                                #     inline=True,
                                #     className="form-item",
                                # ),
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
    return combined

# --- plotting
# main plot
@app.callback(
    Output("time-series-chart", "figure"),
    Output("histogram-chart", "figure"),
    Input("group_dropdown", "value"),
    Input("prediction_toggle", "value"),
    Input("show_future_switch", "on"),
    Input("time_of_day_range", "value"),
    Input("day_of_week_toggle", "value"),
    Input("time-series-chart", "figure"),
)
def display_time_series(
    groups: List[str],
    prediction_toggle_value: str,
    show_future: bool,
    time_of_day_range: List[int],
    day_of_week_toggle: List[int],
    existing_fig_ts: Optional[Figure],
):

    _df = pd.concat(
        [pd.DataFrame({
            "group": pd.Series(dtype='str'),
            "time": pd.Series(dtype='datetime64[s]'),
            "is_train": pd.Series(dtype='bool'),
            "model": pd.Series(dtype='str'),
            "kW": pd.Series(dtype='float64')
        })]
        + [get_combined_df(group) for group in groups]
    )

    # Get the y range now before it gets overwritten

    y_range = (0, _df["kW"].max() * 1.1)
    x_range = existing_fig_ts and existing_fig_ts['layout']['xaxis']['range']
    x_max_range = (_df["time"].min(), _df["time"].max())

    if not show_future:
        _df = _df.query("(is_train == True) | (model != 'actual')")

    if time_of_day_range == [0, 24]:
        pass
    else:
        valid_hours = list(range(*time_of_day_range))
        _df = _df.query("time.dt.hour.isin(@valid_hours)")

    if day_of_week_toggle == list(range(7)):
        pass
    else:
        _df = _df.query("time.dt.dayofweek.isin(@day_of_week_toggle)")



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
                    visible=True,
                    range=x_max_range,
                ),
                range=x_range or (
                    train_val_split_dt - pd.Timedelta("7D"),
                    train_val_split_dt - pd.Timedelta("7D") + pd.Timedelta("30D")
                ),
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
        .update_yaxes(range=y_range, showgrid=False) \
        .update_traces(line=dict(width=1.0))

    # Create the histogram
    fig_hist = px.histogram(
        _df.query("model == 'actual'"),
        x='kW',
        nbins=80,
        color='group', **strong_color_cycle,
        opacity=0.8,
        histnorm='probability density',
        height=275
    )
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
    Input("group_dropdown", "value")
)
def correlation_plot(group_checklist_values: List[str]):
    if len(group_checklist_values) == 2:
        group1: pd.DataFrame = get_combined_df(group_checklist_values[0]).query("model == 'actual'")
        group2: pd.DataFrame = get_combined_df(group_checklist_values[1]).query("model == 'actual'")
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
            color_continuous_scale="haline",
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
                    t=0,
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
    Output("group_dropdown", "options"),
    Input("group_dropdown", "value"),
    # Input("drop_irregular", "value"),
)
def update_multi_options(
    groups_selected: List[str],
    # drop_irregular: bool,
):
    # options = all_groups
    # if drop_irregular:
    #     options = regular_groups

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
        # debug=True,
        host="0.0.0.0",
        port=8097,
    )
