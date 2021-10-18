import itertools
from unittest import TestCase

import torch
from parameterized import parameterized

import numpy as np
from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LinearModel
from torchcast.process.season import Season


class TestProcess(TestCase):
    def test_fourier_season(self):
        series = torch.cos(2. * 3.1415 * torch.arange(1., 8.) / 7.)
        data = torch.stack([series.roll(-i).repeat(3) for i in range(6)]).unsqueeze(-1)
        start_datetimes = np.array([np.datetime64('2019-04-18') + np.timedelta64(i, 'D') for i in range(6)])
        kf = KalmanFilter(
            processes=[Season(id='day_of_week', period='7D', dt_unit='D', K=3, fixed=True)],
            measures=['y']
        )
        kf._scale_by_measure_var = False
        kf.state_dict()['initial_mean'][:] = torch.tensor([1., 0., 0., 0., 0., 0.])
        kf.state_dict()['measure_covariance.cholesky_log_diag'] -= 2
        pred = kf(data, start_offsets=start_datetimes)
        for g in range(6):
            self.assertLess(torch.abs(pred.means[g] - data[g]).mean(), .01, msg=str(g))

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3]))
    @torch.no_grad()
    def test_lm(self, num_groups: int = 1, num_preds: int = 1):
        data = torch.zeros((num_groups, 5, 1))
        kf = KalmanFilter(
            processes=[
                LinearModel(id='lm', predictors=[f"x{i}" for i in range(num_preds)])
            ],
            measures=['y']
        )
        wrong_dim = 1 if num_preds > 1 else 2
        with self.assertRaises((ValueError, torch.jit.Error), msg=(num_groups, num_preds)) as cm:
            kf(data, X=torch.zeros((num_groups, 5, wrong_dim)))

        expected = f'Expected `x.shape[-1:]` to be [{num_preds}], got torch.Size([{wrong_dim}])'
        self.assertIn(expected, str(cm.exception))

        kf(data, X=torch.ones(num_groups, data.shape[1], num_preds))
