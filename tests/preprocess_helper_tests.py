import pandas as pd
import numpy as np
import os
import pytest
from unittest import mock
import warnings
import sys

sys.path.append("notebooks/utils/")

from preprocess_helper import *

def test_preprocess_intertie_data():
    """Test preprocess_intertie_data function."""
    
    preprocess_intertie_data()
    assert os.path.exists("data/processed/intertie.csv")

def test_process_supply_data():
    """Test process_supply_data function."""
    process_supply_data()
    assert os.path.exists("data/processed/supply_load_price.csv")

def test_merge_data():
    """Test merge_data function."""
    merge_data()
    assert os.path.exists("data/processed/preprocessed_features.csv")


if __name__ == "__main__":
    pytest.main([__file__])
