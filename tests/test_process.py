from unittest import TestCase

import torch

import numpy as np
from torchcast.kalman_filter import KalmanFilter
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
