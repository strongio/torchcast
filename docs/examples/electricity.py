# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# # !pip install git+https://github.com/strongio/torch-kalman.git@feature/rename#egg=torchcast
# # !pip install torch_optimizer

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

BASE_DIR = './'#drive/MyDrive'

import os
if 'drive/MyDrive' in BASE_DIR and not os.path.exists(BASE_DIR):
    from google.colab import drive
    drive.mount('/content/drive')
# -

# # Using NN's for Long-Range Forecasts: Electricity Data
#
# In this example we'll show how to handle complex series, in which we don't want to treat individual components of the series as independent. For this example (electricity data) there is no 'hour-in-day' component that's independent of the 'day-of-week' or 'day-in-year' component -- everything is interrelated. Here we'll show how to do this by leveraging `torchcast`'s ability to integrate with any PyTorch neural-network. 
#
# This example will also showcase how to handle a very large number of series. We will train in batches, and use supporting `torch.nn.Embedding` models to allow the model to express differences across different series.
#
# We'll use a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), which consists of electricity-usage for 370 locations, taken every 15 minutes (we'll downsample to hourly).

# + nbsphinx="hidden"
try:
    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])
except FileNotFoundError:
    import requests
    from zipfile import ZipFile
    from io import BytesIO
    response =\
        requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')

    with ZipFile(BytesIO(response.content)) as f:
        df_raw = pd.read_table(f.open('LD2011_2014.txt'), sep=";", decimal=",")

    # melt, collect to hourly:
    df_elec = df_raw.\
        melt(id_vars=['Unnamed: 0'], value_name='kW', var_name='group').\
        assign(time = lambda df_elec: df_elec['Unnamed: 0'].astype('datetime64[h]')).\
        groupby(['group','time'])\
        ['kW'].mean().\
        reset_index()

    df_elec.\
        loc[df_elec['time']>=df_elec['group'].map(df_elec.query("kW>0").groupby('group')['time'].min().to_dict()),:].\
        reset_index(drop=True).\
        to_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), index=False)
        
    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])

SUBSET = 50
np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)
# -

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

df_elec.query("group=='MT_001'").plot('time','kW',figsize=(20,5))

# Some groups have data that isn't really appropriate for modeling -- for example, exhibiting near-zero variation:

df_elec.query("group=='MT_003'").plot('time','kW',figsize=(20,5))

# For some rudimentary cleaning, we'll remove these kinds of regions of 'flatness':

# +
# calculate rolling std-dev:
df_elec['roll_std'] = 0
for g, df in tqdm(df_elec.groupby('group')):
    df_elec.loc[df.index, 'roll_std'] = df['kW'].rolling(48).std()
df_elec.loc[df_elec.pop('roll_std') < .25, 'kW'] = float('nan')

group_missingness = df_elec.assign(missing=lambda df: df['kW'].isnull()).groupby('group')['missing'].mean()

df_elec = df_elec.loc[df_elec['group'].map(group_missingness) < .01,:].reset_index(drop=True)
# -

# For simplicity we'll just drop buildings that are flat in this way for a non-trivial amount of time.
#
# We'll also subset to groups with at least 2 years of data, so we're guaranteed enough data for forecasting:

# +
df_group_summary = df_elec.\
  groupby('group')\
    ['time'].agg(['min','max']).\
    reset_index().\
    assign(history_len = lambda df: (df['max'] - df['min']).dt.days)

train_groups = sorted(df_group_summary.query("history_len >= 730")['group'])

#
train_groups = train_groups[:SUBSET or -1]
df_elec = df_elec.loc[df_elec['group'].isin(train_groups),:].reset_index(drop=True)
# -

# We'll split the data at 2014. For half the groups, this will be used as validation data; for the other half, it will be used as test data.

SPLIT_DT = np.datetime64('2014-01-01')
df_elec['_use_2014_as_test'] = (df_elec['group'].str.replace('MT_','').astype('int') % 2) == 0
df_elec['dataset'] = 'train'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & df_elec['_use_2014_as_test'],'dataset'] = 'test'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & ~df_elec.pop('_use_2014_as_test'),'dataset'] = 'val'
#df_elec['dataset'].value_counts()

# ## A Standard Forecasting Approach
#
# First, let's try a forecasting approach that's fairly 'standard':
#
# - We'll build a model for a single series (instead of sharing a model across multiple series)
# - Each 'process' (e.g. day-in-week, day-in-year, hour-in-day) we'll be assumed to have a separate, independent influence on the data.

# +
from torchcast.process import LocalLevel, LocalTrend, Season

es = ExpSmoother(
    measures=['kW_sqrt'], 
    processes=[
        # seasonal processes:
        Season(id='day_in_week', period=24*7, dt_unit='h', K=3, fixed=True),
        Season(id='day_in_year', period=24*365.25, dt_unit='h', K=8, fixed=True),
        Season(id='hour_in_day', period=24, dt_unit='h', K=8, fixed=True),
        # long-running trend:
        LocalTrend(id='trend'),
#         # 'local' processes: allow temporary deviations that decay 
#         LocalLevel(id='level', decay=True),
#         Season(id='hour_in_day2', period=24, dt_unit='h', K=6, decay=True),
    ]
)

# +
# build our dataset

df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])

# for efficiency of training, we split this single group into multiple groups
df_elec['gym'] =\
     df_elec['group'] + ":" +\
     df_elec['time'].dt.year.astype('str') + "_" +\
     df_elec['time'].dt.month.astype('str').str.zfill(2)

train_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.\
        query("group == 'MT_052'").\
        query("dataset == 'train'").\
        assign(group_year = lambda df: df['group'] + df_elec['time'].dt.year.astype('str')),
    group_colname='group', 
    time_colname='time',
    measure_colnames=['kW_sqrt'],
    dt_unit='h'
)
train_MT_052 = train_MT_052.to(DEVICE)
train_MT_052

# +
es.to(DEVICE)

try:
    es.load_state_dict(torch.load(os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"), map_location=DEVICE))
except FileNotFoundError:
    es.fit(
        train_MT_052.tensors[0],
        start_offsets=train_MT_052.start_datetimes,
    )
    #torch.save(es.state_dict(), os.path.join(BASE_DIR, "electricity_models", "es_standard.pt"))
# -

# XXX
#
# Despite this being the 'standard' approach, we are still using some nonstandard tricks here:
#
# - We are using the `n_step` argument to train our model on one-week ahead forecasts, instead of one step (i.e. hour) ahead. This improves the efficiency of training by 'encouraging' the model to 'care about' longer range forecasts vs. over-focusing on the easier problem of forecasting the next hour.
# - We are splitting our single series into multiple groups. This is helpful since pytorch has a non-trivial overhead for separate tensors -- i.e., it scales well with an increasing batch-size (fewer, but bigger, tensors), but poorly with an increasing time-series length (smaller, but more, tensors).

# ### Model-Evaluation
#
# How does this standard model perform? Plotting the forecasts vs. actual suggests some issues:

eval_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == 'MT_052'"),
    group_colname='group', 
    time_colname='time',
    measure_colnames=['kW_sqrt'],
    dt_unit='h'
).to(DEVICE)
with torch.no_grad():
    y = eval_MT_052.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    pred = es(
            y, 
            start_offsets=eval_MT_052.start_datetimes,
            out_timesteps=y.shape[1] + 24*365.25,
        ) 
    df_pred52 = pred.to_dataframe(eval_MT_052)
df_pred52 = df_pred52.loc[~df_pred52['actual'].isnull(),:].reset_index(drop=True)
pred.plot(df_pred52, split_dt=SPLIT_DT)

# Unfortunately, with hourly data, visualizing long-range forecasts in this way isn't very illuminating: it's just really hard to see the data! Let's try splitting it into weekdays vs. weekends and daytimes vs. nightimes:

# +
df_pred52_split = df_pred52.\
           query("(time.dt.hour==8) | (time.dt.hour==20)").\
           assign(weekend = lambda df: df['time'].dt.weekday.isin([5,6]).astype('int'),
                  night = lambda df: (df['time'].dt.hour == 8).astype('int')).\
        reset_index(drop=True)
df_pred52_split['forecast'] = df_pred52_split.pop('mean')


fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
for (weekend, night), df in df_pred52_split.groupby(['weekend','night']):
    df.plot('time','actual', ax=axes[weekend,night], linewidth=.5, color='black')
    df.plot('time', 'forecast', ax=axes[weekend,night], alpha=.75, color='red')
    axes[weekend,night].axvline(x=SPLIT_DT, color='black', ls='dashed')
    axes[weekend,night].set_title("{}, {}".format('Weekend' if weekend else 'Weekday', 'Night' if night else 'Day'))
plt.tight_layout()
# -

# Viewing the forecastsing this way helps us understand the problem: the annual seasonal pattern is very different for daytimes and nighttimes, but the model isn't (and can't be) capturing that. For example, it incorrectly forecasts a 'hump' during summer days and weekend nights, even though this hump is really only present on weekday nights. The model only allows for a single seasonal pattern, rather than a separate one for different times of the day and days of the week.

# ## Incorporating a Neural Network
#
# If the problem is that our time-series model isn't allowing for interactions among the components of our time-series model, then a natural approach to allowing for these interactions is a neural network.
#
# To make this work, we likely don't want build a model of a single series. Instead, we want to learn across multiple series, so that our network can build representations of patterns that are shared across multiple buildings.

# +
# prepare our dataset: add dummy features that capture the annual, weekly, and daily seasonal patterns:
from torchcast.utils import add_season_features

df_elec = df_elec.\
    pipe(add_season_features, K=3, period='weekly').\
    pipe(add_season_features, K=8, period='yearly').\
    pipe(add_season_features, K=8, period='daily')

season_cols =\
    df_elec.columns[df_elec.columns.str.endswith('_sin')|df_elec.columns.str.endswith('_cos')]

df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])

dataset_kwargs = dict(
    dt_unit='h',
    y_colnames=['kW_sqrt'],
    X_colnames=season_cols,
    time_colname='time'
)

trainval_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec.query("dataset != 'test'"), 
    group_colname='group', 
    **dataset_kwargs,
    batch_size=18
)

val_batch = TimeSeriesDataset.from_dataframe(
    df_elec.query("dataset == 'val'"), 
    group_colname='group', 
    **dataset_kwargs
)
# -

# ### Neural-Network for Time-Series
#
# Since `torchcast` is built on top of PyTorch, we can train a model end-to-end: i.e. we could start with a neural network with random-inits, plug it into a `KalmanFilter`, and train the whole thing.
#
# In practice, we'll generally want to take a neural-network that has already been partially or fully trained to generate sensible predictions. A standard feedforward network will take many many more epochs to train than a `KalmanFilter` (which itself will be slower per-epoch), so it's helpful to keep the two models separate intially.
#
# Here we'll use a network that combines a per-building embedding with a standard architecture that's shared across buildings. This allows the network's shared layers to find representations that are shared across many buildings.
#
# Note the network here uses only calendar features (fourier-transforms on daily/weekly/yearly seasonal periods) to generate predictions. This could be expanded depending on the available data: e.g. holiday-indicators, weather-data, etc.

# +
# mapping from group-names to integers for torch.nn.Embedding:
group_id_mapping = {gn : i for i, gn in enumerate(np.unique(df_elec['group']))}

def get_group_ids(dataset):
    group_names = pd.Series(dataset.group_names).str.split(":", expand=True)[0]
    return torch.as_tensor([group_id_mapping[gn] for gn in group_names], device=DEVICE)

# scaling
group_means = df_elec.query("time < @SPLIT_DT").groupby('group')['kW_sqrt'].mean().to_dict()
group_stds = df_elec.query("time < @SPLIT_DT").groupby('group')['kW_sqrt'].mean().to_dict()
def standardize_by_group(tensor, dataset):
    group_names = pd.Series(dataset.group_names).str.split(":", expand=True)[0]
    means = torch.as_tensor([group_means[gn] for gn in group_names], device=DEVICE)
    stds = torch.as_tensor([group_stds[gn] for gn in group_names], device=DEVICE)
    return (tensor - means[:,None,None]) / stds[:,None,None]


# -

class MatmulNN(torch.nn.Module):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def forward(self, inputs):
        lhs_in, rhs_in = inputs
        lhs_out = self.lhs(lhs_in)
        rhs_out = self.rhs(rhs_in)
        assert len(lhs_out.shape) == len(rhs_out.shape)
        return (lhs_out.unsqueeze(-2) @ rhs_out.unsqueeze(-1)).squeeze(-1)


# +
ts_nn = MatmulNN(
    torch.nn.Sequential(
        torch.nn.Linear(len(season_cols), 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 20)
    ),
    torch.nn.Embedding(
        num_embeddings=len(group_id_mapping),
        embedding_dim=20
    )
)
with torch.no_grad():
    ts_nn.rhs.weight[:] = .01*torch.randn_like(ts_nn.rhs.weight)
from IPython import display
ts_nn.optimizer = torch.optim.Adam(ts_nn.parameters())

ts_nn.loss_history = []
ts_nn.val_history = []
for epoch in range(1500):
    epoch_loss = 0
    for batch in trainval_batches:
        batch, _ = batch.train_val_split(dt=SPLIT_DT)
        y, X = batch.to(DEVICE).tensors
        X[torch.isnan(X)] = 0.
        y = standardize_by_group(y, batch)
        nan_mask = torch.isnan(y)
        group_ids = get_group_ids(batch)
        try:
            pred = ts_nn((X, group_ids.view(-1,1)))
            loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
            loss.backward()
            ts_nn.optimizer.step()
            if torch.isnan(ts_nn.rhs.weight).any():
                raise Exception("foo")
            epoch_loss += loss.item()
        finally:
            ts_nn.optimizer.zero_grad(set_to_none=True)
    ts_nn.loss_history.append(epoch_loss / len(trainval_batches))

    with torch.no_grad():
        y, X = val_batch.to(DEVICE).tensors
        y = standardize_by_group(y, val_batch)
        nan_mask = torch.isnan(y)
        group_ids = get_group_ids(val_batch)
        pred = ts_nn((X, group_ids.view(-1,1)))
        val_loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
        ts_nn.val_history.append(val_loss.item())

    if epoch > 10:
        plt.close()
        display.clear_output(wait=True)
        fig, axes = plt.subplots(ncols=2, figsize=(10,5))
        pd.Series(ts_nn.loss_history[10:]).plot(ax=axes[0], logy=True)
        pd.Series(ts_nn.val_history[10:]).plot(ax=axes[1], logy=True)
        display.display(plt.gcf())
torch.save(ts_nn.state_dict(), os.path.join(BASE_DIR, "electricity_models", "ts_nn.pt"))


# +
# helper class for combining shared layers with per-building embedding:
class PerGroupNN(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module, embedding: torch.nn.Module):
        super().__init__()
        self.shared = shared
        self.embedding = embedding

    def forward(self, X, group_ids):
        shared_weights = self.shared(X)
        per_group = self.embedding(group_ids)
        per_group_w, per_group_b = torch.split(
            per_group.unsqueeze(1), 
            [self.embedding.embedding_dim - 1, 1], 
            -1
        )
        outw = shared_weights.unsqueeze(-2) @ per_group_w.unsqueeze(-1)
        return outw.squeeze(-1) + per_group_b 

per_group_nn = PerGroupNN(
    shared = torch.nn.Sequential(
        torch.nn.Linear(len(season_cols), 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 20)
    ),
    embedding = torch.nn.Embedding(
        num_embeddings=len(group_id_mapping),
        embedding_dim=21
    )
)
per_group_nn.to(DEVICE)

try:
    per_group_nn.load_state_dict(torch.load(os.path.join(BASE_DIR, "electricity_models", "per_group_nn.pt"), map_location=DEVICE))
except FileNotFoundError:
    from IPython import display
    per_group_nn.optimizer = torch.optim.Adam(per_group_nn.parameters())

    per_group_nn.loss_history = []
    per_group_nn.val_history = []
    for epoch in range(1500):
        epoch_loss = 0
        for batch in trainval_batches:
            batch, _ = batch.train_val_split(dt=SPLIT_DT)
            y, X = batch.to(DEVICE).tensors
            y = standardize_by_group(y, batch)
            nan_mask = torch.isnan(y)
            group_ids = get_group_ids(batch)
            try:
                pred = per_group_nn(X, group_ids=group_ids)
                loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
                loss.backward()
                per_group_nn.optimizer.step()
                epoch_loss += loss.item()
            finally:
                per_group_nn.optimizer.zero_grad(set_to_none=True)
        per_group_nn.loss_history.append(epoch_loss / len(train_batches))

        with torch.no_grad():
            y, X = val_batch.to(DEVICE).tensors
            y = standardize_by_group(y, val_batch)
            nan_mask = torch.isnan(y)
            group_ids = get_group_ids(val_batch)
            pred = per_group_nn(X, group_ids=group_ids)
            val_loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
            per_group_nn.val_history.append(val_loss.item())

        if epoch > 10:
            plt.close()
            display.clear_output(wait=True)
            fig, axes = plt.subplots(ncols=2, figsize=(10,5))
            pd.Series(per_group_nn.loss_history[10:]).plot(ax=axes[0], logy=True)
            pd.Series(per_group_nn.val_history[10:]).plot(ax=axes[1], logy=True)
            display.display(plt.gcf())
    torch.save(per_group_nn.state_dict(), os.path.join(BASE_DIR, "electricity_models", "per_group_nn.pt"))
# -

# ### Training our Hybrid Forecasting Model
#
# Now that we have a network that has the representational capacity to transform datetimes into interacting seasonal structure, we can plug this network into a new `KalmanFilter` model.
#
# Additionally, this model is using a few tricks to support training across the many diverse time-serieses in this dataset:
#
# - **Predicting Variance:** We use a `torch.nn.Embedding` model to predict a separate variance-structure for each building. This incorporates both the measure-variance -- the amount of white-noise in the data -- as well as the process-variance -- how variable each component of the series is. 
# - **Predicting Initial Values:** We are still splitting each series into multiple sub-series to aid in efficiency of training. This means that we need to let each series start off with its own unique internal state that encodes its unique seasonal and random-walk structure. 

# +
from torchcast.process import NN, LocalLevel, LocalTrend, Season
from torchcast.covariance import Covariance

shared_nn = copy.deepcopy(per_group_nn.shared)

processes = [
    LocalTrend(id='trend'),
    LocalLevel(id='level', decay=True),
    Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
    NN(id='nn', nn=shared_nn)
]

mvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=1,
)
pvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=Covariance.from_processes(processes).param_rank,
)

kf_nn = KalmanFilter(
    measures=['kW_sqrt'], 
    processes=processes,
    measure_covariance=Covariance.from_measures(['kW_sqrt'], predict_variance=mvar_nn),
    process_covariance=Covariance.from_processes(processes, predict_variance=pvar_nn)
)

kf_nn.initial_state_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=len(kf_nn.initial_mean),
)
with torch.no_grad():
    kf_nn.initial_state_nn.weight[:] *= .01
kf_nn.initial_mean = None

kf_nn.to(DEVICE)
# -

try:
    kf_nn.load_state_dict(torch.load(os.path.join(BASE_DIR,"electricity_models", "kf_nn.pt"), map_location=DEVICE))
except FileNotFoundError:
    from IPython import display
    train_batches = TimeSeriesDataLoader.from_dataframe(
        df_elec.query("dataset == 'train'"), 
        group_colname='gym', 
        **dataset_kwargs, 
        batch_size=100,
        shuffle=True
    )

    from torch_optimizer import Adahessian
    kf_nn.optimizer = Adahessian([
        {'params' : [p for n,p in kf_nn.named_parameters() if 'processes.nn.'     in n], 'lr' : .001},
        {'params' : [p for n,p in kf_nn.named_parameters() if 'processes.nn.' not in n], 'lr' : .50}
    ])

    kf_nn.loss_history = []
    kf_nn.val_history = []
    for epoch in range(200):
        # training:
        train_loss = 0
        for batch in tqdm(train_batches, total=len(train_batches), desc=f"Epoch {epoch}"):
            y, X = batch.to(DEVICE).tensors
            group_ids = get_group_ids(batch)
            try:
                pred = kf_nn(
                    y, 
                    X=X,
                    measure_covariance__X=group_ids,
                    process_covariance__X=group_ids,
                    initial_state=kf_nn.initial_state_nn(group_ids),
                    start_offsets=batch.start_datetimes,
                    n_step=int(24 * 7.5),
                    every_step=False
                )
                loss = -pred.log_prob(y).mean()
                loss.backward(create_graph=True)
                kf_nn.optimizer.step()
                train_loss += loss.item()
            finally:
                kf_nn.optimizer.zero_grad(set_to_none=True)
        train_loss /= len(train_batches)
        kf_nn.loss_history.append(train_loss)

        # validation:
        with torch.no_grad():
            y, X = val_batch.to(DEVICE).tensors
            group_ids = get_group_ids(val_batch)
            pred = kf_nn(
                    y, 
                    X=X,
                    measure_covariance__X=group_ids,
                    process_covariance__X=group_ids,
                    initial_state=kf_nn.initial_state_nn(group_ids),
                    start_offsets=val_batch.start_datetimes,
                    n_step=int(24 * 7.5),
                    every_step=False
                )
            val_errs = nanmean((pred.means - y).abs(), dim=1) / nanmean(y, dim=1)
            kf_nn.val_history.append(val_errs.mean().item())

        display.clear_output(wait=True)
        if epoch > 2:
            plt.close()
            fig, axes = plt.subplots(ncols=2,figsize=(10,5))
            pd.Series(kf_nn.loss_history[2:]).plot(ax=axes[0])
            pd.Series(kf_nn.val_history[2:]).plot(ax=axes[1])
            display.display(plt.gcf())
        torch.save(kf_nn.state_dict(), os.path.join(BASE_DIR,"electricity_models", "kf_nn.pt"))

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
        y = eval_batch.train_val_split(dt=SPLIT_DT)[0].tensors[0]
        X = eval_batch.tensors[1]
        group_ids = get_group_ids(eval_batch)
        pred_nn = kf_nn(
                y, 
                X=X,
                measure_covariance__X=group_ids,
                process_covariance__X=group_ids,
                initial_state=kf_nn.initial_state_nn(group_ids),
                start_offsets=eval_batch.start_datetimes,
                out_timesteps=X.shape[1],
            ) 
        df_pred_nn.append(pred_nn.to_dataframe(eval_batch))
df_pred_nn = pd.concat(df_pred_nn)
df_pred_nn = df_pred_nn.loc[~df_pred_nn['actual'].isnull(),:].reset_index(drop=True)

# +
df_pred52_nn = df_pred_nn.\
           query("group=='MT_052'").\
           query("(time.dt.hour==8) | (time.dt.hour==20)").\
           assign(weekend = lambda df: df['time'].dt.weekday.isin([5,6]).astype('int'),
                  night = lambda df: (df['time'].dt.hour == 8).astype('int')).\
        reset_index(drop=True)
df_pred52_nn['forecast'] = df_pred52_nn.pop('mean')

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
for (weekend, night), df in df_pred52_nn.groupby(['weekend','night']):
    df.plot('time','actual', ax=axes[weekend,night], linewidth=.5, color='black')
    df.plot('time', 'forecast', ax=axes[weekend,night], alpha=.75, color='red')
    axes[weekend,night].axvline(x=SPLIT_DT, color='black', ls='dashed')
    axes[weekend,night].set_title("{}, {}".format('Weekend' if weekend else 'Weekday', 'Night' if night else 'Day'))
plt.tight_layout()

# + nbsphinx="hidden"
"""
- MT_029 -- need (way) higher K for annual season?
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
df_nn_err = df_pred_nn.\
    pipe(inverse_transform).\
    assign(error = lambda df: (df['mean'] - df['actual']).abs(),
           validation = lambda df: df['time'] > SPLIT_DT).\
    groupby(['group','validation']).\
    agg(error = ('error', 'mean')).\
    reset_index()

df_pred52.\
    pipe(inverse_transform).\
    assign(error = lambda df: (df['mean'] - df['actual']).abs(),
           validation = lambda df: df['time'] > SPLIT_DT).\
    groupby(['group','validation']).\
    agg(error = ('error', 'mean')).\
    reset_index().\
    merge(df_nn_err, on=['group', 'validation'])
# -

SPLIT_DT

dfc = pred_nn.to_dataframe(eval_batch, type='components')

pred.plot(dfc.query("time.between('2013-12-26','2014-01-15')"))

from plotnine import *

foo = dfc.\
    query("time.between('2013-12-26','2014-01-30')").\
    assign(
        var=lambda df: ((df['upper'] - df['lower']) / 1.96) ** 2,
        process=lambda df: df['process'].map({'level' : 'Local Level', 
                                              'hour_in_day' : 'Local Hour-in-Day',
                                              'nn' : 'Typical Pattern (NN)'})
    ).groupby(['process','time'])\
    [['mean','var']].sum().\
    reset_index().\
    assign(mean=lambda df: df['mean'] - df.groupby('process')['mean'].transform('mean'),
           lower=lambda df: df['mean'] - df['var'] ** .5,
           upper=lambda df: df['mean'] + df['var'] ** .5)
print(
    ggplot(foo, aes(x='time', y='mean')) + 
    geom_line(color='blue') +
    geom_ribbon(aes(ymin='lower', ymax='upper'), alpha=.25) +
    facet_wrap("~process", scales='free_y', ncol=1) +
    geom_vline(xintercept=SPLIT_DT, linetype='dashed') +
    theme_bw() +
    theme(figure_size=(10,5), subplots_adjust={'wspace': 0.25}) +
    scale_y_continuous(name="State") +
    scale_x_date(name="", date_breaks="1 week", date_labels='%b %d')
)


