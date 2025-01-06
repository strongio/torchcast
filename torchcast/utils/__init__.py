"""
This module includes:

- Data-utils for converting time-series data from a Pandas DataFrame into a PyTorch :class:`torch.utils.data.Dataset`
 and/or :class:`torch.utils.data.DataLoader`. The most common pattern is using the ``from_dataframe()`` classmethod.
- A function for handling implicit missing data
- A function for adding calendar-features: i.e. weekly/daily/yearly season dummy-features.
"""

from .features import add_season_features
from .data import TimeSeriesDataset, TimeSeriesDataLoader, complete_times
from .baseline import make_baseline
from .training import SimpleTrainer, StateSpaceTrainer, SeasonalEmbeddingsTrainer
