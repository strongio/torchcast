# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# # !pip install git+https://github.com/strongio/torchcast.git#egg=torchcast
from typing import Sequence, Optional

import torch
import copy

import matplotlib.pyplot as plt

from torchcast.exp_smooth import ExpSmoother
from torchcast.utils.data import (
    TimeSeriesDataset, TimeSeriesDataLoader, complete_times, nanmean
)

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

BASE_DIR = './'  # drive/MyDrive'

import os

if 'drive/MyDrive' in BASE_DIR and not os.path.exists(BASE_DIR):
    from google.colab import drive

    drive.mount('/content/drive')
# -

# # Using NN's for Long-Range Forecasts: Electricity Data
#
# In this example we'll show how to handle complex series. For this example (electricity data) there is no 'hour-in-day' component that's independent of the 'day-of-week' or 'day-in-year' component -- everything is interrelated. Here we'll show how to do this by leveraging `torchcast`'s ability to integrate with any PyTorch neural-network.
#
# We'll use a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), which consists of electricity-usage for 370 locations, taken every 15 minutes (we'll downsample to hourly).

# + nbsphinx="hidden"
try:
    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])
except FileNotFoundError:
    import requests
    from zipfile import ZipFile
    from io import BytesIO

    response = \
        requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')

    with ZipFile(BytesIO(response.content)) as f:
        df_raw = pd.read_table(f.open('LD2011_2014.txt'), sep=";", decimal=",")

    # melt, collect to hourly:
    df_elec = df_raw. \
        melt(id_vars=['Unnamed: 0'], value_name='kW', var_name='group'). \
        assign(time=lambda df_elec: df_elec['Unnamed: 0'].astype('datetime64[h]')). \
        groupby(['group', 'time']) \
        ['kW'].mean(). \
        reset_index()

    df_elec. \
        loc[df_elec['time'] >= df_elec['group'].map(df_elec.query("kW>0").groupby('group')['time'].min().to_dict()), :]. \
        reset_index(drop=True). \
        to_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), index=False)

    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])

BATCH_SIZE = 12
SUBSET = False
np.random.seed(2021 - 1 - 21)
torch.manual_seed(2021 - 1 - 21)
# -

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE

# ## Data-Cleaning
#
# Our dataset consists of hourly kW readings for multiple buildings:

df_elec.head()

# Electricity-demand data can be challenging because of its complexity. In traditional forecasting applications, we divide our model into siloed processes that each contribute to separate 'behaviors' of the time-series. For example:
#
# - Hour-in-day effects
# - Day-in-week effects
# - Season-in-year effects
# - Weather effects
#
# However, with electricity data, it's limiting to model these separately, because **these effects all interact**: the impact of hour-in-day depends on the day-of-week, the impact of the day-of-week depends on the season of the year, etc.
#
# We can plot some examples to get an initial glance at this complexity.

df_elec.query("group=='MT_001'").plot('time', 'kW', figsize=(20, 5))

# Some groups have data that isn't really appropriate for modeling -- for example, exhibiting near-zero variation:

df_elec.query("group=='MT_003'").plot('time', 'kW', figsize=(20, 5))

# For some rudimentary cleaning, we'll remove these kinds of regions of 'flatness':

# +
# calculate rolling std-dev:
df_elec['roll_std'] = 0
for g, df in tqdm(df_elec.groupby('group')):
    df_elec.loc[df.index, 'roll_std'] = df['kW'].rolling(48).std()
df_elec.loc[df_elec.pop('roll_std') < .25, 'kW'] = float('nan')

group_missingness = df_elec.assign(missing=lambda df: df['kW'].isnull()).groupby('group')['missing'].mean()

df_elec = df_elec.loc[df_elec['group'].map(group_missingness) < .01, :].reset_index(drop=True)
# -

# For simplicity we'll just drop buildings that are flat in this way for a non-trivial amount of time.
#
# We'll also subset to groups with at least 2 years of data, so we're guaranteed enough data for forecasting:

# +
df_group_summary = df_elec. \
    groupby('group') \
    ['time'].agg(['min', 'max']). \
    reset_index(). \
    assign(history_len=lambda df: (df['max'] - df['min']).dt.days)

train_groups = sorted(df_group_summary.query("history_len >= 365")['group'])

if SUBSET:
    train_groups = train_groups[:SUBSET]
df_elec = df_elec.loc[df_elec['group'].isin(train_groups), :].reset_index(drop=True)

# + [markdown] id="f9c827e5"
# We'll split the data at 2013. For half the groups, this will be used as validation data; for the other half, it will be used as test data.

# + id="01ea1964"
SPLIT_DT = np.datetime64('2013-01-01')
df_elec['_use_holdout_as_test'] = (df_elec['group'].str.replace('MT_', '').astype('int') % 2) == 0
df_elec['dataset'] = 'train'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & df_elec['_use_holdout_as_test'], 'dataset'] = 'test'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & ~df_elec.pop('_use_holdout_as_test'), 'dataset'] = 'val'
# df_elec['dataset'].value_counts()
# -

# ## A Standard Forecasting Approach
#
# First, let's try a standard exponential-smoothing algorithm on one of the series. This intentionally doesn't leverage `torchcast`'s ability to train on batches of series, so is quite slow, but will help us have a base case to improve on.

# +
from torchcast.process import LocalTrend, Season

es = ExpSmoother(
    measures=['kW_sqrt'],
    processes=[
        # seasonal processes:
        Season(id='day_in_week', period=24 * 7, dt_unit='h', K=3, fixed=True),
        Season(id='day_in_year', period=24 * 365.25, dt_unit='h', K=8, fixed=True),
        Season(id='hour_in_day', period=24, dt_unit='h', K=8, fixed=True),
        # long-running trend:
        LocalTrend(id='trend'),
    ]
)

# +
# build our dataset
df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])

from_dataframe_kwargs = {
    'dt_unit': 'h',
    'y_colnames': ['kW_sqrt'],
    'time_colname': 'time'
}

train_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec. \
        query("group == 'MT_052'"). \
        query("dataset == 'train'"),
    group_colname='group',
    **from_dataframe_kwargs
)
train_MT_052 = train_MT_052.to(DEVICE)
print(train_MT_052)

# +
es.to(DEVICE)

try:
    es.load_state_dict(torch.load(os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"), map_location=DEVICE))
except FileNotFoundError:
    es.fit(
        train_MT_052.tensors[0],
        start_offsets=train_MT_052.start_datetimes,
    )
    os.makedirs(os.path.join(BASE_DIR, "electricity_models"), exist_ok=True)
    torch.save(es.state_dict(), os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"))
# -

# ### Model-Evaluation
#
# How does this standard model perform? Plotting the forecasts vs. actual suggests serious issues:

eval_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == 'MT_052'"),
    **from_dataframe_kwargs,
    group_colname='group',
).to(DEVICE)
with torch.no_grad():
    _y = eval_MT_052.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    _pred = es(
        _y,
        start_offsets=eval_MT_052.start_datetimes,
        out_timesteps=_y.shape[1] + 24 * 365.25,
    )
    df_pred52 = _pred.to_dataframe(eval_MT_052)
df_pred52 = df_pred52.loc[~df_pred52['actual'].isnull(), :].reset_index(drop=True)
_pred.plot(df_pred52, split_dt=SPLIT_DT)


# Unfortunately, with hourly data, visualizing long-range forecasts in this way isn't very illuminating: it's just really hard to see the data! Let's try splitting it into weekdays vs. weekends and daytimes vs. nightimes:

def plot_2x2(df: pd.DataFrame,
             time_colname: str = 'time',
             pred_colname: str = 'mean',
             actual_colname: str = 'actual'):
    """
    Plot predicted vs. actual for a single group, splitting into 2x2 facets of weekday/end * day/night.
    """
    assert pred_colname in df.columns
    assert actual_colname in df.columns
    df_split = df. \
        query(f"({time_colname}.dt.hour==8) | ({time_colname}.dt.hour==20)"). \
        assign(weekend=lambda df: df[time_colname].dt.weekday.isin([5, 6]).astype('int'),
               night=lambda df: (df[time_colname].dt.hour == 8).astype('int')). \
        reset_index(drop=True). \
        rename(columns={pred_colname: 'forecast', actual_colname: 'actual'})

    _, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    for (wknd, night), df in df_split.groupby(['weekend', 'night']):
        df.plot(time_colname, 'actual', ax=axes[wknd, night], linewidth=.5, color='black')
        df.plot(time_colname, 'forecast', ax=axes[wknd, night], alpha=.75, color='red')
        axes[wknd, night].axvline(x=SPLIT_DT, color='black', ls='dashed')
        axes[wknd, night].set_title("{}, {}".format('Weekend' if wknd else 'Weekday', 'Night' if night else 'Day'))
    plt.tight_layout()


plot_2x2(df_pred52)

# The most obvious issue here is the discrepancy between the predictions on the training data (which look sane) and the validation data (which look insane). This isn't overfitting, but instead the difference between one-step-ahead predictions vs. long-range forecasts. One possibility for why the model does so poorly on the latter is that it wasn't actually trained to generate these predictions: the standard approach has us train on one-step-ahead predictions.
#
# Let's see if we can improve on this. We'll leave the model unchanged but make two changes:
#
# - Use the `n_step` argument to train our model on one-week ahead forecasts, instead of one step (i.e. hour) ahead. This improves the efficiency of training by 'encouraging' the model to 'care about' longer range forecasts vs. over-focusing on the easier problem of forecasting the next hour.
# - Split our single series into multiple groups. This is helpful to speed up training, since pytorch has a non-trivial overhead for separate tensors -- i.e., it scales well with an increasing batch-size (fewer, but bigger, tensors), but poorly with an increasing time-series length (smaller, but more, tensors).

# +
# # for efficiency of training, we split this single group into multiple groups
df_elec['gyq'] = \
    df_elec['group'] + ":" + \
    df_elec['time'].dt.year.astype('str') + "_" + \
    df_elec['time'].dt.quarter.astype('str')

# since TimeSeriesDataset pads short series, drop incomplete groups:
df_elec.loc[df_elec.groupby('gyq')['kW_sqrt'].transform('count') < 2160,'gyq'] = float('nan')

train_MT_052_2 = TimeSeriesDataset.from_dataframe(
    df_elec. \
        query("group == 'MT_052'"). \
        query("dataset == 'train'"),
    group_colname='gyq',
    **from_dataframe_kwargs
).to(DEVICE)

# +
try:
    es.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "electricity_models", "es_standard2.pt"), map_location=DEVICE)
    )
except FileNotFoundError:
    es.fit(
        train_MT_052_2.tensors[0],
        start_offsets=train_MT_052_2.start_datetimes,
        n_step=int(24 * 7.5),
        every_step=False
    )
    torch.save(es.state_dict(), os.path.join(BASE_DIR, "electricity_models", "es_standard2.pt"))

with torch.no_grad():
    _y = eval_MT_052.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    _pred = es(
        _y,
        start_offsets=eval_MT_052.start_datetimes,
        out_timesteps=_y.shape[1] + 24 * 365.25 * 2,
    )
    df_pred52_take2 = _pred.to_dataframe(eval_MT_052)
df_pred52_take2 = df_pred52_take2.loc[~df_pred52_take2['actual'].isnull(),:].reset_index(drop=True)
_pred.plot(df_pred52_take2, split_dt=SPLIT_DT)
# -

# Massive improvement! What about the other view?

plot_2x2(df_pred52_take2)

# Viewing the forecastsing this way helps us see a lingering serious issue: the annual seasonal pattern is very different for daytimes and nighttimes, but the model isn't (and can't be) capturing that. For example, it incorrectly forecasts a 'hump' during summer days and weekend nights, even though this hump is really only present on weekday nights. The model only allows for a single seasonal pattern, rather than a separate one for different times of the day and days of the week.

# ## Incorporating a Neural Network
#
# If the problem is that our time-series model isn't allowing for interactions among the components of our time-series model, then a natural approach to allowing for these interactions is a neural network.
#
# To make this work, we likely don't want build a model of a single series. Instead, we want to learn across multiple series, so that our network can build representations of patterns that are shared across multiple buildings.

# First, we'll use `torchcast.utils.add_season_features` to add dummy features that capture the annual, weekly, and daily seasonal patterns:

# +
from torchcast.utils import add_season_features

df_elec = df_elec. \
    pipe(add_season_features, K=3, period='weekly'). \
    pipe(add_season_features, K=8, period='yearly'). \
    pipe(add_season_features, K=8, period='daily')

season_cols = \
    df_elec.columns[df_elec.columns.str.endswith('_sin') | df_elec.columns.str.endswith('_cos')].tolist()


# -

# Create dataloaders:

# +
train_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec.query("dataset == 'train'"),
    group_colname='group',
    X_colnames=season_cols,
    **from_dataframe_kwargs,
    batch_size=BATCH_SIZE
)

val_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec.query("dataset == 'val'"),
    group_colname='group',
    X_colnames=season_cols,
    **from_dataframe_kwargs,
    batch_size=BATCH_SIZE
)
# -

# ### A Hybrid Model
#
# Below we specify our hybrid model. TODO EXPLAIN

# +
from torchcast.process import LinearModel, LocalTrend, Season
from torchcast.covariance import Covariance
from torchcast.kalman_filter import KalmanFilter

calendar_features_num_latent_dim = 15

calendar_feature_nn = torch.nn.Sequential(
    torch.nn.Linear(len(season_cols), 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, calendar_features_num_latent_dim)
)
processes = [
        # trend:
        LocalTrend(id='trend'),
        # static seasonality:
        LinearModel(id='season', predictors=[f'nn{i}' for i in range(calendar_features_num_latent_dim)]),
        # deviations from typical hour-in-day cycle:
        Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
    ]
kf_nn = KalmanFilter(
    measures=['kW_sqrt'],
    processes=processes,
    measure_covariance=Covariance.from_measures(['kW_sqrt'], predict_variance=True),
    process_covariance=Covariance.from_processes(processes, predict_variance=True),
)
# -

LinearModel.solve_and_predict(-torch.arange(4.)[None,:,None], torch.randn((1,4,1)))

# ### Pre-Training
#
# (we have to pre-train)
#
# To keep things consice, we'll use [PyTorch Lightning](http://pytorch-lightning.readthedocs.io).
#
# Since we'll use this again later, we first make an abstract class that works on any model with a `TimeSeriesDataset` input:

# +
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

class TimeSeriesLightningModule(LightningModule):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._module = module

    def _step(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        batch = batch.to(DEVICE) 
        pred = self(batch, **kwargs)
        return self._get_loss(pred, batch.tensors[0])

    def training_step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        loss = self._step(batch, **kwargs)
        self.log("step_train_loss", loss, on_step=True, on_epoch=False)
        self.log("epoch_train_loss", loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        loss = self._step(batch, **kwargs)
        self.log("step_val_loss", loss, on_step=True, on_epoch=False)
        self.log("epoch_val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def _get_loss(self, predicted, actual) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        raise NotImplementedError
# -

# Now we'll make a class that's more specific to the current task.

# +
class CalendarFeaturePretrainer(TimeSeriesLightningModule):
    def _get_loss(self, predicted, actual) -> torch.Tensor:
        is_valid = ~torch.isnan(actual)
        sq_err = (actual[is_valid] - predicted[is_valid]) ** 2
        return sq_err.mean()

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        y, X = batch.tensors
        y_means = torch.nanmean(y, 1, keepdim=True)
        y_cent = y - y_means
        pred_cent = LinearModel.solve_and_predict(y=y_cent, X=self._module(X))
        return pred_cent + y_means

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


calender_feature_pretrainer = CalendarFeaturePretrainer(calendar_feature_nn).to(DEVICE)
calender_feature_pretrainer
# -


try:
    calender_feature_pretrainer.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "electricity_models", f"calender_feature_pretrainer{calendar_features_num_latent_dim}.pt"))
    )
except FileNotFoundError:
#     %reload_ext tensorboard
#     %tensorboard --logdir=lightning_logs/
    Trainer(
        gpus=int(str(DEVICE)=='cuda'),
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor="epoch_val_loss", patience=10, min_delta=0.0001)]
    ).fit(calender_feature_pretrainer, train_batches, val_batches)
    torch.save(calender_feature_pretrainer.state_dict(), 
               os.path.join(BASE_DIR, "electricity_models", f"calender_feature_pretrainer{calendar_features_num_latent_dim}.pt"))


# +
calendar_feature_nn.to(DEVICE)

eval_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == 'MT_052'"),
    group_colname='group',
    **from_dataframe_kwargs,
    X_colnames=season_cols
).to(DEVICE)

# TODO: drop validation y

with torch.no_grad():
    df_MT_052 = TimeSeriesDataset.tensor_to_dataframe(
        calender_feature_pretrainer(eval_MT_052),
        times=eval_MT_052.times(),
        group_names=eval_MT_052.group_names,
        time_colname='time', group_colname='group',
        measures=['predicted_sqrt']
    ).merge(df_elec.query("group == 'MT_052'"), how='left')

plot_2x2(df_MT_052, actual_colname='kW_sqrt', pred_colname='predicted_sqrt')


# -

# ### Training the Hybrid Model
#
# **TODO:** callout box that this is faster on GPU

# +
names_to_idx = {nm:i for i,nm in enumerate(df_elec['group'].unique())}

class KalmanFilterLightningModule(TimeSeriesLightningModule):
    def _get_loss(self, predicted, actual) -> torch.Tensor:
        return -predicted.log_prob(actual).mean()
    
    def training_step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        return super().training_step(
            batch=batch, 
            batch_idx=batch_idx,
            n_step=int(24 * 7.5),
            every_step=False,
            **kwargs
        )

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        y, X = batch.tensors
        return self._module(
            y,
            season__X=self._module.calendar_feature_nn(X),
            measure_var_multi=self._module.measure_var_nn(
                torch.tensor([names_to_idx[gn] for gn in batch.group_names], device=DEVICE)
            ),
            process_var_multi=self._module.process_var_nn(
                torch.tensor([names_to_idx[gn] for gn in batch.group_names], device=DEVICE)
            ),
            start_offsets=batch.start_offsets
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=.05)


# +
# freeze:
[p.requires_grad_(False) for p in calendar_feature_nn.parameters()]

# deepcopy:
kf_nn.calendar_feature_nn = copy.deepcopy(calendar_feature_nn)


kf_nn.measure_var_nn = kf_nn.measure_var_nn = torch.nn.Sequential(
      torch.nn.Embedding(num_embeddings=len(names_to_idx), embedding_dim=1),
      torch.nn.Softplus()
)

kf_nn.process_var_nn = copy.deepcopy(kf_nn.measure_var_nn)

kf_nn_lightning = KalmanFilterLightningModule(kf_nn)

try:
    kf_nn.load_state_dict(torch.load(os.path.join(BASE_DIR, "electricity_models", f"kf_nn{calendar_features_num_latent_dim}_pvar.pt")))
except FileNotFoundError:
    Trainer(
        gpus=int(str(DEVICE)=='cuda'),
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor="epoch_train_loss", patience=2, min_delta=0.005, verbose=True)]
    ).fit(kf_nn_lightning, train_batches)
    torch.save(kf_nn.state_dict(), os.path.join(BASE_DIR, "electricity_models", f"kf_nn{calendar_features_num_latent_dim}_pvar.pt"))
# -

# ### Model Evaluation
#
# Reviewing the same example-building from before, we see the forecasts are more closely hewing to the actual seasonal structure for each time of day/week. Instead of the forecasts in each panel being essentially identical, each differs in shape. 

# + nbsphinx="hidden"
withtest_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec,
    group_colname='group',
    **dataset_kwargs,
    batch_size=5
)
df_pred_nn = []
with torch.no_grad():
    for eval_batch in tqdm(withtest_batches):
        eval_batch = eval_batch.to(DEVICE)
        #y = eval_batch.train_val_split(dt=SPLIT_DT)[0].tensors[0]
        X = eval_batch.tensors[1]
        group_ids = get_group_ids(eval_batch)
        pred_nn = es_nn(
            y,
            seasonal__X=es_nn.ts_nn((X, group_ids.view(-1, 1))),
            measure_var_multi=es_nn.mvar_nn(group_ids),
            start_offsets=eval_batch.start_datetimes,
            out_timesteps=X.shape[1],
        )
        df_pred_nn.append(pred_nn.to_dataframe(eval_batch))
df_pred_nn = pd.concat(df_pred_nn)
df_pred_nn = df_pred_nn.loc[~df_pred_nn['actual'].isnull(), :].reset_index(drop=True)

# +
# foo=pred_nn.to_dataframe(eval_batch, type='components')
# pred.plot(foo.query("group=='MT_052'"), split_dt=SPLIT_DT)

# +
# df_pred52_nn = df_pred_nn. \
#     query("group=='MT_052'"). \
#     query("(time.dt.hour==8) | (time.dt.hour==20)"). \
#     assign(weekend=lambda df: df['time'].dt.weekday.isin([5, 6]).astype('int'),
#            night=lambda df: (df['time'].dt.hour == 8).astype('int')). \
#     reset_index(drop=True)
# df_pred52_nn['forecast'] = df_pred52_nn.pop('mean')
#
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
# for (weekend, night), df in df_pred52_nn.groupby(['weekend', 'night']):
#     df.plot('time', 'actual', ax=axes[weekend, night], linewidth=.5, color='black')
#     df.plot('time', 'forecast', ax=axes[weekend, night], alpha=.75, color='red')
#     axes[weekend, night].axvline(x=SPLIT_DT, color='black', ls='dashed')
#     axes[weekend, night].set_title("{}, {}".format('Weekend' if weekend else 'Weekday', 'Night' if night else 'Day'))
# plt.tight_layout()

# + nbsphinx="hidden"
"""
- MT_029 -- temporarily dip at split_dt (probably new-year holiday) was mis-attributed to 'trend' component
- MT_018, MT_024, MT_047 -- still systematic bias in certain parts of the day. why? 
    - when we split day into 12/12 hours, don't really see it. so seems like just having trouble with *exact* shape.
- MT_045/MT_036/MT_013 -- seems like LocalLevel should be *much* more responsive. 
    - OTOH it's clearly just xmas, maybe exs like that are incredibly rare otherwise so training doesn't prioritze
""";


# -

# While it's fairly obvious from looking at the forecasts, we can confirm that the 2nd model does indeed substantially reduce forecast error, relative to the 'standard' model:

def inverse_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['mean', 'lower', 'upper', 'actual']:
        df[col] = df[col].clip(lower=0) ** 2
    return df


# +
df_nn_err = df_pred_nn. \
    pipe(inverse_transform). \
    assign(error=lambda df: (df['mean'] - df['actual']).abs(),
           validation=lambda df: df['time'] > SPLIT_DT). \
    groupby(['group', 'validation']). \
    agg(error=('error', 'mean')). \
    reset_index()

df_pred52. \
    pipe(inverse_transform). \
    assign(error=lambda df: (df['mean'] - df['actual']).abs(),
           validation=lambda df: df['time'] > SPLIT_DT). \
    groupby(['group', 'validation']). \
    agg(error=('error', 'mean')). \
    reset_index(). \
    merge(df_nn_err, on=['group', 'validation'], suffixes=('_es', '_es_nn'))
