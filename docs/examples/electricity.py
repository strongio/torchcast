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
    TimeSeriesDataset, TimeSeriesDataLoader, complete_times
)

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

BASE_DIR = 'drive/MyDrive'

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

SUBSET = False
np.random.seed(2021 - 1 - 21)
torch.manual_seed(2021 - 1 - 21)

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

# + [markdown] id="-XZIttQyetx0"
# **TODO**

# + colab={"base_uri": "https://localhost:8080/"} id="ZBgxPMOsesPh" outputId="317866c2-59ee-46eb-a5d2-7de1e371459f"
zero_counts = df_elec.query("kW==0")['time'].value_counts()
print(zero_counts)
df_elec.loc[df_elec['time'].isin(zero_counts.index[zero_counts>100]),'kW'] = float('nan')

# + [markdown] id="3b9bed7b"
# For simplicity we'll just drop buildings that are flat in this way for a non-trivial amount of time.
#
# We'll split the data **XXX**. For half the groups, this will be used as validation data; for the other half, it will be used as test data.

# + id="01ea1964"
SPLIT_DT = np.datetime64('2013-06-01')
df_elec['_use_holdout_as_test'] = (df_elec['group'].str.replace('MT_', '').astype('int') % 2) == 0
df_elec['dataset'] = 'train'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & df_elec['_use_holdout_as_test'], 'dataset'] = 'test'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & ~df_elec.pop('_use_holdout_as_test'), 'dataset'] = 'val'
# df_elec['dataset'].value_counts()
# -

# Finally, drop groups without enough data:

# +
df_group_summary = df_elec. \
    groupby(['group','dataset']) \
    ['time'].agg(['min', 'max']). \
    reset_index(). \
    assign(history_len=lambda df: (df['max'] - df['min']).dt.days)

all_groups = set(df_group_summary['group'])
train_groups = sorted(df_group_summary.query("(dataset=='train') & (history_len >= 365)")['group'])
print(f"Dropping {len(all_groups - set(train_groups)):,} groups")

if SUBSET:
    train_groups = train_groups[:SUBSET]
df_elec = df_elec.loc[df_elec['group'].isin(train_groups), :].reset_index(drop=True)
# -

# TODO
example_group = 'MT_052'

# + colab={"base_uri": "https://localhost:8080/", "height": 360} id="52d1daed" outputId="bf4c04da-1fd6-45ec-a109-2c6fb396f29d"
df_ex = df_elec.query("group==@example_group")
df_ex.loc[df_ex['time'].between('2013-05-01','2013-05-02'),:].plot('time', 'kW', figsize=(20, 5), title=df_ex['group'].iloc[0]);

# + [markdown] tags=[] id="c1ea8dcd-43cd-4216-bcca-3c248f35a4c1"
# ## A Standard Forecasting Approach

# + [markdown] id="niU-YrfdpEYy"
# ### Attempt 1
#
# First, let's try a standard exponential-smoothing algorithm on one of the series. This intentionally doesn't leverage `torchcast`'s ability to train on batches of series, so is quite slow, but will help us have a base case to improve on.

# +
from torchcast.process import LocalTrend, Season

es = ExpSmoother(
    measures=['kW_sqrt_c'],
    processes=[
        # seasonal processes:
        Season(id='day_in_week', period=24 * 7, dt_unit='h', K=3, fixed=True),
        Season(id='day_in_year', period=24 * 365.25, dt_unit='h', K=8, fixed=True),
        Season(id='hour_in_day', period=24, dt_unit='h', K=8, fixed=True),
        # long-running trend:
        LocalTrend(id='trend'),
    ]
)

# + id="ifgdUy1uNQLk"
# transform and center:
df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])
group_means = df_elec.query("dataset=='train'").groupby('group')['kW_sqrt'].mean().to_dict()
df_elec['kW_sqrt_c'] = df_elec['kW_sqrt'] - df_elec['group'].map(group_means) 

# + colab={"base_uri": "https://localhost:8080/"} id="441291b1" outputId="4cd9ff18-9556-480c-b1ce-e1f972f11936"
# build our dataset
from_dataframe_kwargs = {
    'dt_unit': 'h',
    'y_colnames': ['kW_sqrt_c'],
    'time_colname': 'time'
}

train_example = TimeSeriesDataset.from_dataframe(
    df_elec. \
        query("group == @example_group"). \
        query("dataset == 'train'"),
    group_colname='group',
    **from_dataframe_kwargs
)
train_example = train_example.to(DEVICE)
print(train_example)

# + id="4aa3b95f" colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["ecc350a5583942e2a86041ff5b17fc85", "88e97e9a884545629c627d9ca04ae982", "4bc2ae7dd62046be9e601bdc4f9e212d", "273fce2b4367401b9b4e536fc2c1d8f8", "89eeac40db5a439f97869cb80754c734", "c5806e23bd2047e891b9b2233401d725", "4f38e0667cb444db842d9a4bb22eb516", "e2aea428d3894e09a151ce4cf0689247", "d881c181bbfd4f26ace0cae049f5ecfb", "63cf707b3b6547fea4c2bcaa45f1683f", "4d65e5826a784839aef8c44ab58834ee"]} outputId="5ff4feb9-9f22-4b71-e2fa-a200cabf86ca"
es.to(DEVICE)

try:
    es.load_state_dict(torch.load(os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"), map_location=DEVICE))
except FileNotFoundError:
    es.fit(
        train_example.tensors[0],
        start_offsets=train_example.start_datetimes,
    )
    os.makedirs(os.path.join(BASE_DIR, "electricity_models"), exist_ok=True)
    torch.save(es.state_dict(), os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"))

# + [markdown] id="ce66af6b"
# How does this standard model perform? Plotting the forecasts vs. actual suggests serious issues:

# + colab={"base_uri": "https://localhost:8080/", "height": 550} id="4f0bc6e7" outputId="07e237de-00f8-44ce-b286-94058b2c3109"
eval_example = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == @example_group"),
    **from_dataframe_kwargs,
    group_colname='group',
).to(DEVICE)
with torch.no_grad():
    _y = eval_example.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    _pred = es(
        _y,
        start_offsets=eval_example.start_datetimes,
        out_timesteps=_y.shape[1] + 24 * 365.25 * 2,
    )
    df_pred52 = _pred.to_dataframe(eval_example)
df_pred52 = df_pred52.loc[~df_pred52['actual'].isnull(), :].reset_index(drop=True)
_pred.plot(df_pred52, split_dt=SPLIT_DT)

# + [markdown] id="72b64170"
# The most obvious issue here is the discrepancy between the predictions on the training data (which look sane) and the validation data (which look insane). This isn't overfitting, but instead the difference between one-step-ahead predictions vs. long-range forecasts. One possibility for why the model does so poorly on the latter is that it wasn't actually trained to generate these predictions: the standard approach has us train on one-step-ahead predictions.

# + [markdown] id="ZCmH_vByo2M6"
# ### Attempt 2
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

train_example_2 = TimeSeriesDataset.from_dataframe(
    df_elec. \
        query("group == @example_group"). \
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
        train_example_2.tensors[0],
        start_offsets=train_example_2.start_datetimes,
        n_step=int(24 * 7.5),
        every_step=False
    )
    torch.save(es.state_dict(), os.path.join(BASE_DIR, "electricity_models", "es_standard2.pt"))

with torch.no_grad():
    _y = eval_example.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    _pred = es(
        _y,
        start_offsets=eval_example.start_datetimes,
        out_timesteps=_y.shape[1] + 24 * 365.25 * 2,
    )
    df_pred52_take2 = _pred.to_dataframe(eval_example)
df_pred52_take2 = df_pred52_take2.loc[~df_pred52_take2['actual'].isnull(),:].reset_index(drop=True)
_pred.plot(df_pred52_take2, split_dt=SPLIT_DT)


# + [markdown] id="d54b5e0f"
# Massive improvement! Unfortunately, with hourly data, visualizing long-range forecasts in this way isn't very illuminating: it's just really hard to see the data! Let's try splitting it into weekdays vs. weekends and daytimes vs. nightimes:

# + id="be3744d7"
def plot_2x2(df: pd.DataFrame,
             time_colname: str = 'time',
             pred_colname: str = 'mean',
             actual_colname: str = 'actual',
             day_hour: int = 14,
             night_hour: int = 2):
    """
    Plot predicted vs. actual for a single group, splitting into 2x2 facets of weekday/end * day/night.
    """
    assert pred_colname in df.columns
    assert actual_colname in df.columns
    df_split = df. \
        query(f"({time_colname}.dt.hour=={day_hour}) | ({time_colname}.dt.hour=={night_hour})"). \
        assign(weekend=lambda df: df[time_colname].dt.weekday.isin([5, 6]).astype('int'),
               night=lambda df: (df[time_colname].dt.hour == night_hour).astype('int')). \
        reset_index(drop=True). \
        rename(columns={pred_colname: 'forecast', actual_colname: 'actual'})

    _, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    for (wknd, night), df in df_split.groupby(['weekend', 'night']):
        df.plot(time_colname, 'actual', ax=axes[wknd, night], linewidth=.5, color='black')
        df.plot(time_colname, 'forecast', ax=axes[wknd, night], alpha=.75, color='red')
        axes[wknd, night].axvline(x=SPLIT_DT, color='black', ls='dashed')
        axes[wknd, night].set_title("{}, {}".format('Weekend' if wknd else 'Weekday', 'Night' if night else 'Day'))
    plt.tight_layout()


plot_2x2(df_pred52_take2)

# + [markdown] id="7840bc87"
# Viewing the forecastsing this way helps us see a lingering serious issue: the annual seasonal pattern is very different for daytimes and nighttimes, but the model isn't (and can't be) capturing that. For example, it incorrectly forecasts a 'hump' during summer days and weekend nights, even though this hump is really only present on weekday nights. The model only allows for a single seasonal pattern, rather than a separate one for different times of the day and days of the week.

# + [markdown] id="71398c7f"
# ## Incorporating a Neural Network
#
# If the problem is that our time-series model isn't allowing for interactions among the components of our time-series model, then a natural approach to allowing for these interactions is a neural network.
#
# To make this work, we likely don't want build a model of a single series. Instead, we want to learn across multiple series, so that our network can build representations of patterns that are shared across multiple buildings.

# + [markdown] id="c9d7a811"
# First, we'll use `torchcast.utils.add_season_features` to add dummy features that capture the annual, weekly, and daily seasonal patterns:

# + id="2442c37c"
from torchcast.utils import add_season_features

df_elec = df_elec. \
    pipe(add_season_features, K=3, period='weekly'). \
    pipe(add_season_features, K=8, period='yearly'). \
    pipe(add_season_features, K=10, period='daily')

season_cols = \
    df_elec.columns[df_elec.columns.str.endswith('_sin') | df_elec.columns.str.endswith('_cos')].tolist()


# + [markdown] id="6317f3b6"
# Create dataloaders:

# + id="c89c2fe6"
def make_dataloader(type: str, 
                    batch_size: int,
                    group_colname: str = 'group',
                    frac: float = 1,
                    **kwargs) -> TimeSeriesDataLoader:
    assert type in {'train','val','all'}
    data = df_elec if type == 'all' else df_elec.query(f"dataset == '{type}'")
    if frac < 1:
        _keep = data[group_colname].isin(data[group_colname].drop_duplicates().sample(frac=frac))
        data = data.loc[_keep, :]
    kwargs = {**from_dataframe_kwargs, **kwargs}
    return TimeSeriesDataLoader.from_dataframe(
        data,
        group_colname=group_colname,
        X_colnames=season_cols,
        **kwargs,
        batch_size=batch_size
    )


# + [markdown] id="ab89e8c0"
# ### A Hybrid Model
#
# Below we specify our hybrid model. TODO EXPLAIN

# + id="Y47xgP3MSTKA"
calendar_features_num_latent_dim = 15

# + id="5e4fd4d9"
from torchcast.process import LinearModel, LocalTrend, Season
from torchcast.covariance import Covariance
from torchcast.kalman_filter import KalmanFilter

es_nn = ExpSmoother(
#kf_nn = KalmanFilter(
    measures=['kW_sqrt_c'],
    processes=[
        # trend:
        LocalTrend(id='trend'),
        # static seasonality:
        LinearModel(id='season', predictors=['nn_output']),
        #LinearModel(id='season', predictors=[f'nn_output{i}' for i in range(calendar_features_num_latent_dim)]),
        # deviations from typical hour-in-day cycle:
        Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
    ],
    measure_covariance=Covariance.from_measures(['kW_sqrt_c'], predict_variance=True),
)

# + [markdown] tags=[]
# ### Pre-Training
#
# _(we have to pre-train)_
#
# To keep things consice, we'll use [PyTorch Lightning](http://pytorch-lightning.readthedocs.io).
#
# Since we'll use this again later, we first make an abstract class that works on any model with a `TimeSeriesDataset` input:

# +
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch_optimizer import Adahessian

class TimeSeriesLightningModule(LightningModule):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._module = module

    def _step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        batch = batch.to(DEVICE) 
        pred = self(batch, **kwargs)
        return self._get_loss(pred, batch.tensors[0])

    def training_step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        loss = self._step(batch, batch_idx=batch_idx, **kwargs)
        self.log("step_train_loss", loss, on_step=True, on_epoch=False)
        self.log("epoch_train_loss", loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        loss = self._step(batch, batch_idx=batch_idx, **kwargs)
        self.log("step_val_loss", loss, on_step=True, on_epoch=False)
        self.log("epoch_val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def _get_loss(self, predicted, actual) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        raise NotImplementedError
# -

# Now we'll make a class that's more specific to the current task.

# + colab={"base_uri": "https://localhost:8080/"} id="b6cf75c5" outputId="e62fd1f0-400e-4670-ff59-f06dd06131d2"
names_to_idx = {nm:i for i,nm in enumerate(df_elec['group'].unique())}

class CalendarFeatureNN(TimeSeriesLightningModule):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module=module)
        with torch.no_grad():
            self._module.emb_nn.weight *= .1
    
    def _get_loss(self, predicted, actual) -> torch.Tensor:
        is_valid = ~torch.isnan(actual)
        sq_err = (actual[is_valid] - predicted[is_valid]) ** 2
        return sq_err.mean()
    
    def backward(self,
                loss: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                optimizer_idx: int,
                *args,
                **kwargs):
                return super().backward(loss, optimizer, optimizer_idx, create_graph=True)

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        y, X = batch.tensors

        #pred = LinearModel.solve_and_predict(y=y, X=self._module(X))
        group_names = [gn.split(":")[0] for gn in batch.group_names]
        group_ids = [names_to_idx[gn] for gn in group_names]

        # use NNs to get latent features and group-specific coefs on those features:
        coefs = self._module.emb_nn(torch.tensor(group_ids, device=DEVICE))
        feats = self._module.features_nn(X)

        # multiply features by coefs (as in a linear-model)
        # NOTE: no bias term -- assumes centered y
        pred = (coefs.unsqueeze(1) * feats).sum(-1, keepdim=True)

        return pred

    def configure_optimizers(self) -> torch.optim.Optimizer:
        pars = [p for p in self.parameters() if p.requires_grad]
        return Adahessian(pars, lr=.15)

calendar_feature_nn = CalendarFeatureNN(
    torch.nn.ModuleDict({'features_nn': 
                         torch.nn.Sequential(
                            torch.nn.Linear(len(season_cols), 48),
                            torch.nn.Tanh(),
                            torch.nn.Linear(48, 48),
                            torch.nn.Tanh(),
                            torch.nn.Linear(48, calendar_features_num_latent_dim)
            )
        ,'emb_nn': 
        torch.nn.Embedding(num_embeddings=len(names_to_idx), embedding_dim=calendar_features_num_latent_dim)
    })
).to(DEVICE)

try:
    calendar_feature_nn.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "electricity_models", f"calendar_feature_nn{calendar_features_num_latent_dim}.pt"))
    )
except FileNotFoundError:
#     %reload_ext tensorboard
#     %tensorboard --logdir=lightning_logs/
    Trainer(
        gpus=int(str(DEVICE) == 'cuda'),
        logger=CSVLogger(os.path.join(BASE_DIR, "electricity_models"), f'calendar_feature_nn{calendar_features_num_latent_dim}'),
        log_every_n_steps=1,
        min_epochs=100,
        callbacks=[EarlyStopping(monitor="epoch_val_loss", patience=10, min_delta=.0001, verbose=False)]
    ).fit(calendar_feature_nn, make_dataloader('train', batch_size=45), make_dataloader('val', batch_size=55))
    calendar_feature_nn.trainer = None
    torch.save(calendar_feature_nn.state_dict(),
               os.path.join(BASE_DIR, "electricity_models", f"calendar_feature_nn{calendar_features_num_latent_dim}.pt"))



# +
calendar_feature_nn.to(DEVICE)

eval_example = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == @example_group"),#.assign(kW_sqrt_c=lambda df: df['kW_sqrt_c'].where(df['time']<=SPLIT_DT)),
    group_colname='group',
    **from_dataframe_kwargs,
    X_colnames=season_cols
).to(DEVICE)

with torch.no_grad():
    df_example = TimeSeriesDataset.tensor_to_dataframe(
        calendar_feature_nn(eval_example),
        times=eval_example.times(),
        group_names=eval_example.group_names,
        time_colname='time', group_colname='group',
        measures=['predicted_sqrt']
    ).merge(df_elec.query("group == @example_group"), how='left')

# from plotnine import *
# print(
#     ggplot(df_example.loc[df_example['time'].dt.year==2013,:].assign(month=lambda df: df['time'].dt.month), 
#            aes(x='time.dt.hour', y='kW_sqrt_c', color='time.dt.day_name()')) +
#     stat_summary(fun_y=np.mean, geom='line') +
#     stat_summary(aes(y='predicted_sqrt'), linetype='dashed', size=1.5, fun_y=np.mean, geom='line') +
#     facet_wrap("~month") +
#     theme(figure_size=(10,10))
# )

plot_2x2(df_example, actual_colname='kW_sqrt_c', pred_colname='predicted_sqrt')

# + [markdown] tags=[]
# ### Training the Hybrid Model
#
# **TODO:** callout box that this is faster on GPU

# + id="AXvwLsaScg0l"
_df_train = df_elec.query("dataset=='train'").reset_index(drop=True)
_df_train['_diff'] = _df_train.groupby('group')['kW_sqrt'].diff()
group_resid_devs = _df_train.groupby('group')['_diff'].std().to_dict()


# + id="1c-0M6tTlqty"
class HybridForecaster(TimeSeriesLightningModule):
    def _get_loss(self, predicted, actual) -> torch.Tensor:
        return -predicted.log_prob(actual).mean()

    def _step(self, batch: TimeSeriesDataset, batch_idx: int, **kwargs) -> torch.Tensor:
        # self.print(batch_idx)
        return super()._step(batch=batch, batch_idx=batch_idx, n_step=int(24 * 15), every_step=False, **kwargs)

    def forward(self, batch: TimeSeriesDataset, **kwargs) -> torch.Tensor:
        y, X = batch.tensors
        group_names = [gn.split(":")[0] for gn in batch.group_names]
        group_ids = [names_to_idx[gn] for gn in group_names]
        
        # features of the time-series predict mvar:
        means = torch.as_tensor([group_means[gn] for gn in group_names], device=DEVICE)
        devs = torch.as_tensor([group_resid_devs[gn] for gn in group_names], device=DEVICE)
        cvs = devs / means
        mvar_X = torch.stack([means, devs, cvs], 1).log()

        return self._module(
            y,
            season__X=self._module.calendar_feature_nn(batch),
            #season__X=self._module.calendar_feature_nn.features_nn(X),
            measure_var_multi=self._module.measure_var_nn(mvar_X),
            start_offsets=batch.start_offsets,
            **kwargs
        )
    
    def backward(self,
                loss: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                optimizer_idx: int,
                *args,
                **kwargs):
                return super().backward(loss, optimizer, optimizer_idx, create_graph=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adahessian([p for p in self.parameters() if p.requires_grad], lr=.10)
        
# # freeze the "features" while keeping the per-building embeddings unfrozen:
[p.requires_grad_(False) for p in calendar_feature_nn._module.features_nn.parameters()]
# [p.requires_grad_(False) for p in calendar_feature_nn.parameters()]

# XXX:
es_nn.calendar_feature_nn = copy.deepcopy(calendar_feature_nn)

# XXX
# es_nn.measure_var_nn = torch.nn.Sequential(
#     torch.nn.Embedding(num_embeddings=len(names_to_idx), embedding_dim=1),
#     torch.nn.Softplus()
# )
# with torch.no_grad():
#     es_nn.measure_var_nn[0].weight /= 10
es_nn.measure_var_nn = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Softplus()
)

es_nn_lightning = HybridForecaster(es_nn)

# -

try:
    es_nn.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "electricity_models", f"es_nn{calendar_features_num_latent_dim}X.pt"))
    )
except FileNotFoundError:
    Trainer(
        gpus=int(str(DEVICE) == 'cuda'),
        logger=CSVLogger(os.path.join(BASE_DIR, "electricity_models"), f'es_nn{calendar_features_num_latent_dim}', flush_logs_every_n_steps=1),
        log_every_n_steps=1,
        min_epochs=10,
        callbacks=[EarlyStopping(monitor="epoch_val_loss", patience=10, min_delta=0.001, verbose=True)]
    ).fit(es_nn_lightning, 
          make_dataloader('train', batch_size=144, group_colname='gyq', shuffle=True), 
          make_dataloader('val', batch_size=55)
          )
    torch.save(es_nn.state_dict(), os.path.join(BASE_DIR, "electricity_models", f"es_nn{calendar_features_num_latent_dim}.pt"))
# ### Model Evaluation
#
# Reviewing the same example-building from before, we see the forecasts are more closely hewing to the actual seasonal structure for each time of day/week. Instead of the forecasts in each panel being essentially identical, each differs in shape. 

# + nbsphinx="hidden"
es_nn_lightning.to(DEVICE)
df_pred_nn = []
with torch.no_grad():
    for batch in tqdm(make_dataloader('all', batch_size=50)):
        batch = batch.to(DEVICE)
        _y = batch.train_val_split(dt=SPLIT_DT)[0].tensors[0]
        _X = batch.tensors[1]
        pred_nn = es_nn_lightning(
            batch.with_new_tensors(_y, _X), 
            out_timesteps=_X.shape[1]
        )
        df_pred_nn.append(pred_nn.to_dataframe(batch))
df_pred_nn = pd.concat(df_pred_nn)

df_pred_nn = df_pred_nn.loc[~df_pred_nn['actual'].isnull(), :].reset_index(drop=True)


plot_2x2(df_pred_nn.query("group==@example_group"))

# +
#pred_nn.plot(pred_nn.to_dataframe(batch, type='components'), split_dt=SPLIT_DT)

# # MT_363 is has nice season-structure, better example to use?
# # MT_276 looks good
# _df = df_pred_nn.query("group=='MT_363'")
# print(_df['group'].iloc[0])
# plot_2x2(_df)

# df_pred52_nn = df_pred_nn. \
#     query("group==@example_group"). \
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
