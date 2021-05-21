from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger, DEBUG, INFO, StreamHandler, FileHandler, Formatter, Logger
import os.path
import time
from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



@dataclass
class ColumnSelector(BaseEstimator, TransformerMixin):
    feature_names: List[str]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.feature_names]



def get_logger(filepath: str, name: Optional[str] = None):
    """Get logger having console and file handler.
    """
    logger = getLogger(name or __name__)
    logger.setLevel(DEBUG)
    for h in logger.handlers:
        logger.removeHandler(h)
    file_handler = FileHandler(filepath)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter('"%(asctime)s","%(name)s","%(levelname)s","%(message)s"'))
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@contextmanager
def timer(name: str, logger: Optional[Logger] = None, level: int = DEBUG):
    '''
    Refference
    ----------
    https://amalog.hateblo.jp/entry/kaggle-snippets
    '''
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'{name}: start')
    yield
    print_(f'{name}: done in {time.time() - t0:.3f} s')


def join_prefix_suffix(name: str, prefix: str, suffix: str, sep: str = '_') -> str:
    out = name
    if prefix:
        out = prefix + sep + out
    if suffix:
        out = out + sep + suffix
    return out
