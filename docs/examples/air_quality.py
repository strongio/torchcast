# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
# ---

# %% {"nbsphinx": "hidden"}
import pandas as pd
import torch

from torchcast.state_space import Predictions
from torchcast.utils.datasets import load_air_quality_data
from torchcast.kalman_filter import KalmanFilter
from torchcast.utils.data import TimeSeriesDataset
import os

import numpy as np

np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)

# %% [markdown]
# # Multivariate Forecasts: Beijing Multi-Site Air-Quality Data
#
# We'll demonstrate several features of `torchcast` using a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data). It includes data on air pollutants and weather from 12 sites.

# %% {"tags": ["remove_cell"]}
df_aq = load_air_quality_data('weekly')

SPLIT_DT = np.datetime64('2016-02-22')

df_aq

# %% [markdown] {"hidePrompt": true}
# ### Univariate Forecasts
#
# Let's try to build a model to predict total particulate-matter (PM2.5 and PM10). 
#
# First, we'll make our target the sum of these two types. We'll log-transform since this is strictly positive.

# %%
from torchcast.process import LocalTrend, Season

# create a dataset:
df_aq['PM'] = df_aq['PM10'] + df_aq['PM2p5'] 
df_aq['PM_log10'] = np.log10(df_aq['PM']) 
dataset_pm_univariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PM_log10'],
    group_colname='station', 
    time_colname='week'
)
dataset_pm_univariate_train, _ = dataset_pm_univariate.train_val_split(dt=SPLIT_DT)

# create a model:
kf_pm_univariate = KalmanFilter(
    measures=['PM_log10'], 
    processes=[
        LocalTrend(id='trend'),
        Season(id='day_in_year', period=365.25 / 7, dt_unit='W', K=4, fixed=True)
    ]
)

# fit:
kf_pm_univariate.fit(
    dataset_pm_univariate_train.tensors[0],
    start_offsets=dataset_pm_univariate_train.start_datetimes
)

# %% [markdown]
# Let's see how our forecasts look:

# %%
# helper for transforming log back to original:
def inverse_transform(df):
    df = df.copy()
    # bias-correction for log-transform (see https://otexts.com/fpp2/transformations.html#bias-adjustments)
    df['mean'] = df['mean'] + .5 * (df['upper'] - df['lower']) / 1.96 ** 2
    # inverse the log10:
    df[['actual', 'mean', 'upper', 'lower']] = 10 ** df[['actual', 'mean', 'upper', 'lower']]
    df['measure'] = df['measure'].str.replace('_log10', '')
    return df

# generate forecasts:
forecast = kf_pm_univariate(
        dataset_pm_univariate_train.tensors[0],
        start_offsets=dataset_pm_univariate_train.start_datetimes,
        out_timesteps=dataset_pm_univariate.tensors[0].shape[1]
)

df_forecast = inverse_transform(forecast.to_dataframe(dataset_pm_univariate))
print(forecast.plot(df_forecast, max_num_groups=3, split_dt=SPLIT_DT))

# %% [markdown]
# #### Evaluating Performance: Expanding Window
#
#
# To evaluate our forecasts, we will not use the long-range forecasts above. Instead, we will use an [expanding window](https://eng.uber.com/forecasting-introduction#:~:text=Comparing) approach to evaluate a shorter forecast horizon. In this approach, we generate N-step-ahead forecasts at every timepoint:
#
# ![title](expanding_window.png)
#
#
# This approach is straightforward in `torchcast`, using the `n_step` argument. Here we'll generate 4-week-ahead predictions. Note that we're still separating the validation time-period.

# %%
with torch.no_grad():
    pred_4step = kf_pm_univariate(
        dataset_pm_univariate.tensors[0],
        start_offsets=dataset_pm_univariate.start_datetimes,
        n_step=4
    )

df_univariate_error = pred_4step.\
    to_dataframe(dataset_pm_univariate, group_colname='station', time_colname='week').\
    pipe(inverse_transform).\
    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
    assign(
        error = lambda df: np.abs(df['mean'] - df['actual']),
        validation = lambda df: df['week'] > SPLIT_DT
    ).\
    groupby(['station','validation'])\
    ['error'].mean().\
    reset_index()
df_univariate_error.groupby('validation')['error'].agg(['mean','std'])

# %% [markdown]
# ### Multivariate Forecasts
#
# Can we improve our model by splitting the pollutant we are forecasting into its two types (2.5 and 10), and modeling them in a multivariate manner?

# %%
# create a dataset:
df_aq['PM10_log10'] = np.log10(df_aq['PM10'])
df_aq['PM2p5_log10'] = np.log10(df_aq['PM2p5'])
dataset_pm_multivariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PM10_log10','PM2p5_log10'],
    group_colname='station', 
    time_colname='week'
)
dataset_pm_multivariate_train, _ = dataset_pm_multivariate.train_val_split(dt=SPLIT_DT)

# create a model:
_processes = []
for m in dataset_pm_multivariate.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=4, measure=m, fixed=True)
    ])
kf_pm_multivariate = KalmanFilter(measures=dataset_pm_multivariate.measures[0], processes=_processes)

# fit:
kf_pm_multivariate.fit(
    dataset_pm_multivariate_train.tensors[0],
    start_offsets=dataset_pm_multivariate_train.start_datetimes
)
# %% [markdown]
# We can generate our 4-step-ahead predictions for validation as we did before:

# %%
with torch.no_grad():
    pred_4step = kf_pm_multivariate(
        dataset_pm_multivariate.tensors[0],
        start_offsets=dataset_pm_multivariate.start_datetimes,
        n_step=4
    )
pred_4step.means.shape

# %% [markdown]
# At this point, though, we run into a problem: we we have forecasts for both PM2.5 and PM10, but we ultimately want a forecast for their *sum*. With untransformed data, we could take advantage of the fact that [sum of correlated normals is still normal](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables#Correlated_random_variables):

# %%
torch.sum(pred_4step.means, 2)

# %% [markdown]
# In our case this unfortunately won't work: we have log-transformed our measures. This seems like it was the right choice (i.e. our residuals look reasonably normal and i.i.d):

# %%
pred_4step.plot(pred_4step.to_dataframe(dataset_pm_multivariate, type='components').query("process=='residuals'"))


# %% [markdown]
# In this case, we **can't take the sum of our forecasts to get the forecast of the sum**, and [there's no simple closed-form expression for the sum of lognormals](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C14&q=SUMS+OF+LOGNORMALS&btnG=).
#
# One option that is fairly easy in `torchcast` is to use a [Monte-Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) approach: we'll just generate random-samples based on the means and covariances underlying our forecast. In that case, the sum of the PM2.5 + PM10 forecasted-samples *is* the forecasted PM sum we are looking for:

# %%
def mc_preds_to_dataframe(preds: Predictions,
                          dataset: TimeSeriesDataset,
                          inverse_transform_fun: callable, num_draws: int = 500,
                          **kwargs) -> pd.DataFrame:
    """
    Our predictions are on the transformed scale, and we'd like to sum across measures on the original scale;
    this function uses a monte-carlo approach to do this.
    """
    # generate draws from the forecast distribution, apply inverse-transform:
    mc_draws = inverse_transform_fun(torch.distributions.MultivariateNormal(*preds).rsample((num_draws,)))
    # sum across measures (e.g. 2.5 and 10), then mean across draws:
    mc_predictions = mc_draws.sum(-1, keepdim=True).mean(0)
    # convert to a dataframe
    return TimeSeriesDataset.tensor_to_dataframe(
        mc_predictions,
        times=dataset.times(),
        group_names=dataset.group_names,
        measures=['predicted'],
        **kwargs
    )


# %%
df_mv_pred = mc_preds_to_dataframe(
    pred_4step,
    dataset_pm_multivariate,
    inverse_transform_fun=lambda x: 10 ** x,
    group_colname='station',
    time_colname='week'
)
df_multivariate_error = df_mv_pred. \
    merge(df_aq.loc[:, ['station', 'week', 'PM']]). \
    assign(
        error=lambda df: np.abs(df['predicted'] - df['PM']),
        validation=lambda df: df['week'] > SPLIT_DT
    ). \
    groupby(['station', 'validation']) \
    ['error'].mean(). \
    reset_index()
df_multivariate_error.groupby('validation')['error'].agg(['mean','std'])

# %% [markdown]
# We see that this approach has reduced our error: substantially in the training period, and moderately in the validation period. We can look at the per-site differences to reduce common sources of noise and see that the reduction is consistent (it holds for all but one site):

# %%
df_multivariate_error.\
    merge(df_univariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')

# %% [markdown]
# ### Adding Predictors
#
# In many settings we have external predictors we'd like to incorporate. Here we'll use four predictors corresponding to weather conditions. Of course, in a forecasting context, we run into the problem of needing to fill in values for these predictors for future dates. For an arbitrary forecast horizon this can be a complex issue; for simplicity here we'll focus on the 4-week-ahead predictions we used above, and simply lag our weather predictors by 4.

# %%
from torchcast.process import LinearModel

# prepare external predictors:
predictors_raw = ['TEMP', 'PRES', 'DEWP']
predictors = [p.lower() + '_lag4' for p in predictors_raw]
# standardize:
predictor_means = df_aq.query("week<@SPLIT_DT")[predictors_raw].mean()
predictor_stds = df_aq.query("week<@SPLIT_DT")[predictors_raw].std()
df_aq[predictors] = (df_aq[predictors_raw] - predictor_means) / predictor_stds
# lag:
df_aq[predictors] = df_aq.groupby('station')[predictors].shift(4, fill_value=0)

# create dataset:
dataset_pm_lm = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    y_colnames=['PM10_log10','PM2p5_log10'],
    X_colnames=predictors,
    group_colname='station', 
    time_colname='week',
)
dataset_pm_lm_train, _ = dataset_pm_lm.train_val_split(dt=SPLIT_DT)
dataset_pm_lm_train

# %%
# create a model:
_processes = []
for m in dataset_pm_lm.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=4, measure=m, fixed=True),
        LinearModel(id=f'{m}_lm', predictors=predictors, measure=m)
    ])
kf_pm_lm = KalmanFilter(measures=dataset_pm_lm.measures[0], processes=_processes)

# fit:
y, X = dataset_pm_lm_train.tensors
kf_pm_lm.fit(
    y,
    X=X, # if you want to supply different predictors to different processes, you can use `{process_name}__X`
    start_offsets=dataset_pm_lm_train.start_datetimes
)

# %% [markdown]
# Here we show how to inspect the influence of each predictor:

# %%
# inspect components:
with torch.no_grad():
    y, X = dataset_pm_lm.tensors
    pred_4step = kf_pm_lm(
        y,
        X=X,
        start_offsets=dataset_pm_lm.start_datetimes,
        n_step=4
    )
pred_4step.plot(pred_4step.to_dataframe(dataset_pm_lm, type='components').query("process.str.contains('lm')"), split_dt=SPLIT_DT)

# %% [markdown]
# Now let's look at error:

# %%
# error:
df_lm_pred = mc_preds_to_dataframe(
    pred_4step,
    dataset_pm_lm,
    inverse_transform_fun=lambda x: 10 ** x,
    group_colname='station',
    time_colname='week'
)

df_lm_error = df_lm_pred. \
    merge(df_aq.loc[:, ['station', 'week', 'PM']]). \
    assign(
        error=lambda df: np.abs(df['predicted'] - df['PM']),
        validation=lambda df: df['week'] > SPLIT_DT
    ). \
    groupby(['station', 'validation']) \
    ['error'].mean(). \
    reset_index()

df_lm_error.\
    merge(df_multivariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')

# %% [markdown]
# In this setting, the lagged predictors do not help.
