import os

import numpy as np
import pytest
import torch

from tests import _PATH_DATA



def test_data():
    train_image_path = f"{_PATH_DATA}/processed/train.py"

    test_image_path = f"{_PATH_DATA}/processed/test.py"

    N_train = 25000
    N_test = 5000
    train_images = torch.load(train_image_path)
    test_images = torch.load(test_image_path)
    assert len(train_images) == N_train
    assert len(test_images) == N_test
