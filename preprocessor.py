import argparse
from functools import partial
import os
import os.path
import time
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from utils import get_logger, timer

START_YEAR = 1992
END_YEAR = 2013

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=
        'This module will output extracted features from given train/evaluation dataset.')
    parser.add_argument('-i1', '--input1', type=str, default=os.path.join(os.getcwd(), 'read_only', 'TrainDataSet.csv'),
                        help='Filepath of training set. Default is "<current_directory>/read_only/TrainDataSet.csv"')
    parser.add_argument('-i2', '--input2', type=str, default=os.path.join(os.getcwd(), 'read_only', 'EvaluationData.csv'),
                        help='Filepath of training set. Default is "<current_directory>/read_only/EvaluationData.csv"')
    parser.add_argument('-o', '--out_dir', type=str, default=os.path.join(os.getcwd(), 'preprocessed'),
                        help='Directory for which this module write the output data. This module will try to make directory '
                        'if given directory dose not exist. Default is "<current_directry>/preprocessed"')
    parser.add_argument('-o1', '--output1', type=str, default='TrainDataSetPreprocessed.csv',
                        help='Filename of preprocessed training set. Default is "TrainDataSetPreprocessed.csv"')
    parser.add_argument('-o2', '--output2', type=str, default='EvaluationDataPreprocessed.csv',
                        help='Filename of preprocessed test set. Default is "EvaluationDataPreprocessed.csv"')
    parser.add_argument('-l', '--log_filepath', type=str, default=os.path.join(os.getcwd(), 'log', 'preprocess.log'),
                        help='Log filepath. This module will try to make directory if given directory dose not exist. '
                        'Default is "<current_directry>/log/preprocess.log"')
    parser.add_argument('-k', '--keep_all_place_id', action='store_true',
                        help='If given, observations of the all of `PlaceID` in training set will be used for '
                        'feature engineering, even if that `PlaceID` dose not have observation of the all years, '
                        'which can be cause of missing value.')
    parser.add_argument('-sy', '--start_year', type=int, default=START_YEAR,
                        help=f'The year when to start collecting observations. Default = {START_YEAR}.')
    parser.add_argument('-ey', '--end_year',type=int, default=END_YEAR,
                        help=f'The year when to start collecting observations. Default = {END_YEAR}.')
    args = parser.parse_args()
    if not args.start_year < args.end_year:
        raise ValueError('Should be start_year < end_year but start_year = {}, end_year = {}' \
                         .format(args.start_year, args.end_year))
    for f in (args.input1, args.input2):
        if not os.path.isfile(f):
            raise FileNotFoundError(f'{f} is not a existing file.')
    return args


def load_dataset(train_filepath: str, test_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test data file.
    
    Parameters
    ----------
    train_filepath, test_filepath: str
        Filepath of train/test data.
    """
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)
    assert(train.duplicated(subset = ['PlaceID', 'Year']).sum() == 0)
    assert(test.duplicated(subset = ['PlaceID', 'Year']).sum() == 0)
    return train, test


def extract_ts_features(place_id: int, df: pd.DataFrame, feature_names: Tuple[str, ...]) -> pd.DataFrame:
    """Extract time series transition features.

    Parameters
    ----------
    place_id: int
        `PlaceID` to be processed.
    df: pd.DataFrame:
        Input dataframe.
    feature_names: Tuple of str
        Strings which represent feature names.
    """
    place_df = df.loc[df.PlaceID == place_id, :].copy()
    place_df.sort_values('Year', inplace=True)
    first_year = place_df.Year.min()

    for f in feature_names:
        base_value = place_df.loc[place_df.Year == first_year, f].values[0]
        # Index: almost same with row value of this year / that of first year, but +1 for avoiding zero division.
        place_df[f'{f}Index'] = (1 + place_df[f]) / (1 + base_value)
        # change rate from previous year, first year is = -0.
        place_df[f'{f}ChangeRate'] = (1 + place_df[f]).pct_change().fillna(0)

    return place_df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Simple faeature 
    df['SumByMean1pLight'] = df.apply(lambda x: x.SumLight / (1 + x.MeanLight), axis=1)

    # Time series transition
    feature_names = ['MeanLight', 'SumLight', 'SumByMean1pLight']
    if 'AverageLandPrice' in df.columns:
        # not a feature, but use for training phase.
        feature_names.append('AverageLandPrice')
    extract_func = partial(extract_ts_features, df=df, feature_names=feature_names)
    result_dfs = Parallel(n_jobs=-1)([delayed(extract_func)(place_id) for place_id in df.PlaceID.unique()])
    out_df = pd.concat(result_dfs)
    del result_dfs
    out_df['ChangeRateDiff'] = out_df['SumLightChangeRate'] - out_df['MeanLightChangeRate']
    out_df['IndexDiff'] = out_df['SumLightIndex'] - out_df['MeanLightIndex']
    # Statistic features
    measures = feature_names \
             + [f'{f}Index' for f in feature_names] \
             + [f'{f}ChangeRate' for f in feature_names]
    stat_by_place_id = out_df.groupby('PlaceID')[measures].describe()
    stat_by_place_id.columns = [f'{i[0]}_{i[1]}' for i in stat_by_place_id.columns]  # drop multi-index
    count_columns = [c for c in stat_by_place_id.columns if c.endswith('_count')]
    stat_by_place_id.drop(columns=count_columns, inplace=True)  # all values might be = 1, useless
    out_df = pd.merge(out_df, stat_by_place_id.reset_index(), on='PlaceID', how='left')
    # Coefficient of Variation
    for m in measures:
        cv = out_df[f'{m}_std'] / out_df[f'{m}_mean']
        cv.replace([np.inf, -np.inf], np.nan, inplace=True)
        out_df[f'{m}_cv'] = cv.values
        out_df[f'{m}_cv'].fillna(0, inplace=True)
    return out_df


def main() -> None:
    since = time.time()  # start time

    # get arguments
    args = _get_args()

    # prepare directories and logger
    log_dir = os.path.dirname(args.log_filepath)
    made_log_dir = False
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        made_log_dir = True
    logger = get_logger(name=__name__, filepath=args.log_filepath)
    if made_log_dir:
        logger.warning('{} dose not exist but made automatically'.format(log_dir))    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        logger.warning('{} dose not exist but made automatically'.format(args.out_dir))

    # load dataset
    with timer('Load dataset', logger):
        train, test = load_dataset(args.input1, args.input2)
    logger.debug('{} have {} records ({} unique `PlaceID`)' \
                 .format(args.input1, train.shape[0], train.PlaceID.nunique()))
    logger.debug('{} have {} records ({} unique `PlaceID`)' \
                 .format(args.input2, test.shape[0], test.PlaceID.nunique()))
    
    # drop observations if its year is out of range
    if args.start_year != train.Year.min() or args.end_year != train.Year.max():
        train = train.query(f'{args.start_year} <= Year <= {args.end_year}')
        logger.warning('Drop observations from from training set, of which `Year` is out of range '
                       '({} records / {} unique `PlaceID` remains).' \
                       .format(train.shape[0], train.PlaceID.nunique()))
    if args.start_year != test.Year.min() or args.end_year != test.Year.max():
        test = test.query(f'{args.start_year} <= Year <= {args.end_year}')
        logger.warning('Drop observations from from test set, of which `Year` is out of range '
                       '({} records / {} unique `PlaceID` remains).' \
                       .format(test.shape[0], test.PlaceID.nunique()))

    # drop unuseful observations if needed.
    if not args.keep_all_place_id:  # drop
        logger.debug('Observations of `PlaceID` will be dropped if that `PlaceID` '
                     'dose not have ones of the all years (start_year, end_year) '
                     '= ({}, {})'.format(args.start_year, args.end_year))
        nrow_expected = args.end_year - args.start_year + 1
        nrow_by_place_id = train.PlaceID.value_counts()
        keep_mask = (nrow_by_place_id == nrow_expected)
        place_id_to_keep = nrow_by_place_id[keep_mask].index
        train = train[train.PlaceID.isin(place_id_to_keep)]
        logger.debug('{} records ({} unique `PlaceID`) is remained in training set' \
                     .format(train.shape[0], train.PlaceID.nunique()))
    else:  # do not drop
        logger.warning('`keep_all_place_id` was set True. There can be observations of'
                       '`PlaceID` which dose not have ones of the all years.')
        
    # extract feature
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)
        with timer('Extract feature (training set)', logger):
            features_train = extract_features(train)
        with timer('Extract feature (test set)', logger):
            features_test = extract_features(test)

    # write preprocessed data
    with timer(f'Write output to {args.out_dir}', logger):
        features_train.to_csv(os.path.join(args.out_dir, args.output1), index=False)
        features_test.to_csv(os.path.join(args.out_dir, args.output2), index=False)
    logger.info('Complete ({:.5f} seconds passed).'.format(time.time() - since))
    return None


if __name__ == '__main__':
    main()
