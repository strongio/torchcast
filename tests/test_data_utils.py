import unittest
from warnings import warn

import torch

from torchcast.utils.data import TimeSeriesDataset


class TestDataUtils(unittest.TestCase):
    def test_time_series_dataset(self):
        values = torch.randn((3, 39, 2))

        batch = TimeSeriesDataset(
            values,
            group_names=['one', 'two', 'three'],
            start_times=[0, 0, 0],
            measures=[['y1', 'y2']],
            dt_unit=None
        )
        try:
            import pandas as pd
        except ImportError:
            warn("Not testing TimeSeriesDataset.to_dataframe, pandas not installed.")
            return
        df1 = batch.to_dataframe()

        df2 = pd.concat([
            pd.DataFrame(values[i].numpy(), columns=batch.all_measures).assign(group=group, time=batch.times()[0])
            for i, group in enumerate(batch.group_names)
        ])
        self.assertTrue((df1 == df2).all().all())
