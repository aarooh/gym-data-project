import pandas as pd
import pytest
import numpy as np

from main import load_dataset
from configs.config import get_config


@pytest.fixture
def load_data() -> pd.DataFrame:
    config = get_config("./configs/config.yml")
    df, _ = load_dataset(config)
    return df


class TestDataset:
    def test_dataset_size(self, load_data: pd.DataFrame):
        """Test if dataset has more than 50 000 rows"""
        assert len(load_data) > 50000

    def test_time_series(self, load_data: pd.DataFrame):
        """Test if dataset has correct time series
        making sure that all of the values are between start_date and end_date
        """
        start_date = "2020-04-24"
        end_date = "2021-05-11"
        load_data = load_data[
            (load_data["time"] < start_date) & (load_data["time"] > end_date)
        ]
        assert len(load_data) == 0

    def test_column_has_positive_values(self, load_data: pd.DataFrame):
        """Test if dataset has only positive values.
        first we select all numeric columsn and flatten our dataframe
        and count all negative values
        """
        load_data = load_data.iloc[:, 1:]
        assert np.sum((load_data < 0).values.ravel()) == 0
