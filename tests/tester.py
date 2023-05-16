import os
import sys
from unittest import TestCase

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))

from src.suhbramaniyan-t-package.ingest_data import load_data
from src.guru_mle.train-t-package import train_model
from src.guru_mle.score-t-package import get_score

class Testingest(TestCase):
    def test_is_num(self):
        a, b, c, d = load_data()
        assert isinstance(a, pd.DataFrame)
        assert isinstance(b, pd.Series)
        assert isinstance(c, pd.DataFrame)
        assert isinstance(d, pd.Series)


class Testtrain(TestCase):
    def test_is_model(self):
        f = training()
        l = not (len(dir(f)) == len(dir(type(f)())))
        self.assertTrue(l, "not a model")


class TestJoke(TestCase):
    def test_is_num(self):
        a = scoring()
        assert type(a) is np.float64, "num must be an floating integer"
