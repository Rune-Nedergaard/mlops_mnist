import os

import numpy as np
import pytest
import torch

from tests import _PATH_DATA

#_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
#_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
#_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

def test_data():
    train_image_path = f"{_PATH_DATA}/processed/train.pt"

    test_image_path = f"{_PATH_DATA}/processed/test.pt"

    N_train = 25000
    N_test = 5000
    train_images = torch.load(train_image_path)
    test_images = torch.load(test_image_path)
    assert len(train_images[0]) == N_train
    assert len(test_images[0]) == N_test

