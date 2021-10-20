# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

# +
# # !pip install git+https://github.com/strongio/torchcast.git#egg=torchcast

import torch
import copy

import matplotlib.pyplot as plt

from torchcast.exp_smooth import ExpSmoother
from torchcast.process import LocalTrend, Season
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUBSET = 200

print(subprocess.run(["kaggle", "competitions", "download", "-c", "rossmann-store-sales"], 
                     stdout=subprocess.PIPE).stdout.decode())

# +
from zipfile import ZipFile

with ZipFile("./rossmann-store-sales.zip") as zf:
    df_store = pd.read_csv(zf.open("store.csv"))
    df_train = pd.read_csv(zf.open("train.csv"), parse_dates=['Date'], dtype={'StateHoliday' : 'str'})
    assert (df_train.loc[df_train['Open']==0,'Sales'] == 0).all()
    df_train.loc[df_train['Open']==0,'Sales'] = float('nan')
    assert (df_train.loc[df_train['Open']==0,'Customers'] == 0).all()
    df_train.loc[df_train['Open']==0,'Customers'] = float('nan')
# -

from plotnine import *

qplot(df_train['Customers'],bins=100) + scale_y_continuous(trans='log1p')

qplot(df_train['Sales'],bins=100) + scale_y_continuous(trans='log1p')

store_sales_means = df_train.groupby('Store')['Sales'].mean().to_dict()
store_cust_means = df_train.groupby('Store')['Customers'].mean().to_dict()
df_train['sales_c'] = df_train['Sales'] / df_train['Store'].map(store_sales_means)
df_train['customers_c'] = df_train['Customers'] / df_train['Store'].map(store_cust_means)
df_train.head()

dataset_train = TimeSeriesDataset.from_dataframe(
    df_train.loc[df_train['Store'].isin(df_train['Store'].drop_duplicates().sample(SUBSET)),:],
    group_colname='Store',
    time_colname='Date',
    measure_colnames=['sales_c'],#, 'customers_c'],
    dt_unit='D'
)
dataset_train

# +
# create a model:
_processes = []
for m in dataset_train.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        Season(id=f'{m}_day_in_week', period=7, dt_unit='D', K=3, measure=m),
        Season(id=f'{m}_day_in_year', period=365.25, dt_unit='D', K=5, measure=m)
    ])
es1 = ExpSmoother(measures=dataset_train.measures[0], processes=_processes)
#es1 = torch.jit.script(es1)

# fit:
es1.fit(
    dataset_train.tensors[0],
    start_offsets=dataset_train.start_datetimes,
    n_step=6,every_step=False
)
# -

torch.save(es1.state_dict(), "./rossman-es1.pt")

pred = es1(dataset_train.tensors[0], start_offsets=dataset_train.start_datetimes)
foo=pred.to_dataframe(dataset_train)

pred.plot(foo.query("measure=='sales_c' & time.dt.year==2015"))

bar=pred.to_dataframe(dataset_train, type='components', multi=None)

pred.plot(bar.query("measure=='sales_c' & time.dt.year==2015"))

bar

pred.plot(bar.query("measure=='sales_c' & time.dt.year==2015").groupby(['time','measure','process','group'])[['mean','std']].sum().reset_index())


