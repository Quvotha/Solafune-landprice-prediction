import argparse
from dataclasses import dataclass, field
from functools import partial
import json
from logging import Logger, INFO, DEBUG
import os
import os.path
import pickle
import time
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from utils import get_logger, timer, join_prefix_suffix, ColumnSelector

rmse = partial(mean_squared_error, squared=False)
FEATURE_NAMES = ['Year',
                 'MeanLight',
                 'SumLight',
                 'SumByMean1pLight',
                 'MeanLightIndex',
                 'MeanLightChangeRate',
                 'SumLightIndex',
                 'SumLightChangeRate',
                 'SumByMean1pLightIndex',
                 'SumByMean1pLightChangeRate',
                 'ChangeRateDiff',
                 'IndexDiff',
                 'MeanLight_mean',
                 'MeanLight_std',
                 'MeanLight_min',
                 'MeanLight_25%',
                 'MeanLight_50%',
                 'MeanLight_75%',
                 'MeanLight_max',
                 'SumLight_mean',
                 'SumLight_std',
                 'SumLight_min',
                 'SumLight_25%',
                 'SumLight_50%',
                 'SumLight_75%',
                 'SumLight_max',
                 'SumByMean1pLight_mean',
                 'SumByMean1pLight_std',
                 'SumByMean1pLight_min',
                 'SumByMean1pLight_25%',
                 'SumByMean1pLight_50%',
                 'SumByMean1pLight_75%',
                 'SumByMean1pLight_max',
                 'MeanLightIndex_mean',
                 'MeanLightIndex_std',
                 'MeanLightIndex_min',
                 'MeanLightIndex_25%',
                 'MeanLightIndex_50%',
                 'MeanLightIndex_75%',
                 'MeanLightIndex_max',
                 'SumLightIndex_mean',
                 'SumLightIndex_std',
                 'SumLightIndex_min',
                 'SumLightIndex_25%',
                 'SumLightIndex_50%',
                 'SumLightIndex_75%',
                 'SumLightIndex_max',
                 'SumByMean1pLightIndex_mean',
                 'SumByMean1pLightIndex_std',
                 'SumByMean1pLightIndex_min',
                 'SumByMean1pLightIndex_25%',
                 'SumByMean1pLightIndex_50%',
                 'SumByMean1pLightIndex_75%',
                 'SumByMean1pLightIndex_max',
                 'MeanLightChangeRate_mean',
                 'MeanLightChangeRate_std',
                 'MeanLightChangeRate_min',
                 'MeanLightChangeRate_25%',
                 'MeanLightChangeRate_50%',
                 'MeanLightChangeRate_75%',
                 'MeanLightChangeRate_max',
                 'SumLightChangeRate_mean',
                 'SumLightChangeRate_std',
                 'SumLightChangeRate_min',
                 'SumLightChangeRate_25%',
                 'SumLightChangeRate_50%',
                 'SumLightChangeRate_75%',
                 'SumLightChangeRate_max',
                 'SumByMean1pLightChangeRate_mean',
                 'SumByMean1pLightChangeRate_std',
                 'SumByMean1pLightChangeRate_min',
                 'SumByMean1pLightChangeRate_25%',
                 'SumByMean1pLightChangeRate_50%',
                 'SumByMean1pLightChangeRate_75%',
                 'SumByMean1pLightChangeRate_max',
                 'MeanLight_cv',
                 'SumLight_cv',
                 'SumByMean1pLight_cv',
                 'MeanLightIndex_cv',
                 'SumLightIndex_cv',
                 'SumByMean1pLightIndex_cv',
                 'MeanLightChangeRate_cv',
                 'SumLightChangeRate_cv',
                 'SumByMean1pLightChangeRate_cv',
]


class PlaceIDCluasterer(BaseEstimator, TransformerMixin):
    '''Unsupervised classification for PlaceID.

    This class will make clusters using time series transition 
    of given features.
    
    Attributes
    ----------
    feature_names: List[str]
        List of feature names used for training.
    '''

    def __init__(self, feature_names: List[str], *,
                 kmeans_args: dict = {'random_state': 1, 'n_clusters': 5}):
        self.feature_names = feature_names
        self.kmeans_args = kmeans_args
        self.clustere = Pipeline(steps=[
            ('scaler', MinMaxScaler()),
            ('clusterer', KMeans(**kmeans_args))
        ])


    def pivot_table(self, X: pd.DataFrame) -> pd.DataFrame:
        '''Convert given dataframe so each row represents 1 `PlaceID`.


        Parameters
        ----------
        X: pd.DataFrame
            The structure is same with competition data, meaning that 
            each row represents one `PlaceID`'s one year.

        Return
        ------
        pivot_table: pd.DataFrame
            Each row represents time series transition of `self.feature_names` 
            of one `PlaceID`.
            - number of rows: Number of unique `PlaceID` in X
            - number of columns: (Number of unique `Year` in X) * len(self.feature_names)
        '''
        return pd.pivot_table(X, values=self.feature_names, index='PlaceID', columns='Year')


    def fit(self, X, y=None):
        X_pivot = self.pivot_table(X)
        self.clustere.fit(X_pivot)
        return self


    def predict(self, X) -> pd.DataFrame:
        X_pivot = self.pivot_table(X)
        cluster_id = self.clustere.predict(X_pivot)
        return pd.DataFrame(data=cluster_id, index=X_pivot.index, columns=['ClusterID'])


@dataclass
class CVResult:
    """Class for keeping track of an cross validation.
    
    Attributes
    -----------
    k: int
        Number of cross validation loop. Should be > 1 integer.
    pred_train, pred_valid: pd.DataFrame
        Prediction result of each cv loop. Always contains following columns;
        - Prediction: predicted value.
        - Actual: actual value.
        - Fold: Number of cv-iteration.
    models: list of models, length = k
        List of models obtained from cv-iterations.
    """
    k: int
    pred_train: List[pd.DataFrame] = field(default_factory=list)
    pred_valid: List[pd.DataFrame] = field(default_factory=list)
    losses_train: List[float] = field(default_factory=list)
    losses_valid: List[float] = field(default_factory=list)
    models: list = field(default_factory=list)


    def add_model(self, model) -> None:
        self.models.append(model)
        return None


    def add_prediction(self, pred: pd.DataFrame, metric: float,
                       validation: bool=False) -> None:
        if validation:
            self.pred_valid.append(pred)
            self.losses_valid.append(metric)
        else:
            self.pred_train.append(pred)
            self.losses_train.append(metric)
        return None


    def get_cv_metrics(self) -> Tuple[float, float, float, float]:
        return (np.mean(self.losses_train), np.std(self.losses_train),
                np.mean(self.losses_valid), np.std(self.losses_valid))


@dataclass
class ModelCV(BaseEstimator, TransformerMixin):
    target_col: Union[str, List[str]]
    feature_names: List[str]
    key_columns: List[str]
    model_factory: callable
    logger: Logger
    stratifier: Optional[Union[str, np.ndarray, pd.Series]] = None
    grouper: Optional[Union[str, np.ndarray, pd.Series]] = None
    k: int = 10
    random_state: int = 1
    model_name: str = 'anonymous'


    def fit(self, X: pd.DataFrame, y=None):
        assert(isinstance(self.k, int) and self.k > 1)
        assert(isinstance(self.random_state, int) and self.random_state >= 0)
        cv_result = run_cv(X, self.target_col, self.feature_names, self.key_columns,
                           stratifier=self.stratifier, grouper=self.grouper,
                           model_factory=self.model_factory, logger=self.logger,
                           k=self.k, random_state=self.random_state)
        self.cv_result_ = cv_result
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.vstack(
            [model.predict(X) for model in self.cv_result_.models]
        ).T
        return out


    def save_results(self, cv_dir: str, model_dir: str) -> None:
        model_name = self.model_name
        pred_train = pd.concat(self.cv_result_.pred_train)
        pred_train.to_csv(os.path.join(cv_dir, f'{model_name}_pred_train.csv'),
                          index=False)
        pred_valid = pd.concat(self.cv_result_.pred_valid)
        pred_valid.to_csv(os.path.join(cv_dir, f'{model_name}_pred_valid.csv'),
                          index=False)
        for i, model in enumerate(self.cv_result_.models):
            fold = i + 1
            basename = f'{model_name}_fold{fold}.model'
            with open(os.path.join(model_dir, basename), 'wb') as f:
                pickle.dump(model, f)
                f.close()


def get_lgbm(random_state: int) -> Pipeline:
    return Pipeline(steps=[
        ('regressor', LGBMRegressor(random_state=random_state,
                                    importance_type='gain'))
    ])


def get_xgb(random_state: int) -> Pipeline:
    return Pipeline(steps=[
        ('regressor', XGBRegressor(random_state=random_state,
                                   n_jobs=-1,
                                   importance_type='gain'))
    ])


def get_ridge(random_state: int) -> Pipeline:
    return Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', Ridge(random_state=random_state))
    ])


def get_knn(random_state: int) -> Pipeline:
    return Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('regressor', KNeighborsRegressor(n_jobs=-1))
    ])


def get_svm(random_state: int) -> Pipeline:
    return Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('regressor', SVR())
    ])


def extract_importances(cv_result: CVResult,
                        feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Extract coefficients & feature importances information if obtainable.
    '''
    coefficients = pd.DataFrame(index=feature_names)
    importances = pd.DataFrame(index=feature_names)
    for i, model in enumerate(cv_result.models):
        fold = i + 1
        final_estimator = model[-1][-1]
        if hasattr(final_estimator, 'coef_'):
            coefficients[f'fold{fold}'] = final_estimator.coef_
        if hasattr(final_estimator, 'feature_importances_'):
            importances[f'fold{fold}'] = final_estimator.feature_importances_
    return coefficients, importances


def run_cv(train: pd.DataFrame,
           target_col,
           feature_names: List[str],
           key_columns: List[str],
           *,
           stratifier: Optional[Union[str, np.ndarray, pd.Series]] = None,
           grouper: Optional[Union[str, np.ndarray, pd.Series]] = None,
           model_factory: callable,
           logger: Optional[Logger] = None,
           k: int = 10,
           random_state: int = 1) -> CVResult:

    # validate arguments
    if stratifier is not None and grouper is not None:
        raise ValueError('Only one of `stratifier` and `grouper` should be but both given.')
    elif stratifier is not None:
        y = train[stratifier] if isinstance(stratifier, str) else stratifier
        groups = None
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        if logger is not None:
            logger.debug('StratifiedKFold is seleceted for splitter')
    elif grouper is not None:
        y = None
        groups = train[grouper] if isinstance(grouper, str) else grouper
        splitter = GroupKFold(n_splits=k)
        if logger is not None:
            logger.debug('GroupKFold is seleceted for splitter')
    else:
        y = None
        groups = None
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        if logger is not None:
            logger.debug('KFold is seleceted for splitter')

    cv_result = CVResult(k=k)
    for i, (train_idx, valid_idx) in enumerate(splitter.split(train, y, groups)):
        fold = i + 1
        # separate dataset into train/validation set
        X_train, y_train, pred_train = (train.iloc[train_idx],
                                        train.iloc[train_idx][target_col],
                                        train.iloc[train_idx][key_columns])
        X_valid, y_valid, pred_valid = (train.iloc[valid_idx],
                                        train.iloc[valid_idx][target_col],
                                        train.iloc[valid_idx][key_columns])
        # training
        model = Pipeline(steps=[
            ('selector', ColumnSelector(feature_names)),
            ('predictor', model_factory(random_state))
        ])
        model.fit(X_train, y_train)
        # save result
        ## model
        cv_result.add_model(model)
        ## cv-result
        ### training set
        pred_train_ = model.predict(X_train)
        metric_train = rmse(y_train, pred_train_)
        pred_train['Prediction'] = pred_train_
        pred_train['Actual'] = y_train.values
        pred_train['Fold'] = fold
        cv_result.add_prediction(pred=pred_train, metric=metric_train)
        ### validation set
        pred_valid_ = model.predict(X_valid)
        metric_valid = rmse(y_valid, pred_valid_)
        pred_valid['Prediction'] = pred_valid_
        pred_valid['Actual'] = y_valid.values
        pred_valid['Fold'] = fold
        cv_result.add_prediction(pred=pred_valid, metric=metric_valid, validation=True)
        if logger is not None:
            logger.info('Fold {}: training metrics={:.5f} validation metrics={:.5f}' \
                       .format(fold, metric_train, metric_valid))
    
    metrics = cv_result.get_cv_metrics()
    if logger is not None:
        logger.info('Complete {}-fold cv. Training metris: mean={:.5f}, std={:.5f}, '
                    'validation metrics: mean={:.5f}, std={:.5f}' \
                    .format(k, metrics[0], metrics[1], metrics[2], metrics[3]))
    return cv_result

    
def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=
        'This module will train a regressor using given training data file, then output models (pickle format), '
        'cross validation results (meaning prediction data for training/validation set obtained in cross validation '
        'iterations and feature importance information), and log file.')
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(os.getcwd(), 'preprocessed', 'TrainDataSetPreprocessed.csv'),
                        help='Filepath of features data extracted from training set. Default is '
                        '"<current_directory>/preprocessed/TrainDataSetPreprocessed.csv"')
    parser.add_argument('-c', '--cv_dir', type=str, default=os.path.join(os.getcwd(), 'cv'),
                        help='Directory for which this module will write cross validation results. '
                        'This module will try to make directory if given directory dose not exist. '
                        'Default is "<current_directory>/cv/"')
    parser.add_argument('-m', '--model_dir', type=str, default=os.path.join(os.getcwd(), 'models'),
                        help='Directory for which this module will write models. This module will try to make '
                        'directory if given directory dose not exist. Default is "<current_directory>/models/"')
    parser.add_argument('-l', '--log_filepath', type=str, default=os.path.join(os.getcwd(), 'log', 'train.log'),
                        help='Log filepath. This module will try to make directory if given directory dose not '
                        'exist. Default is "<current_directry>/log/train.log"')
    parser.add_argument('-k', '--n_splits', type=int, default=5,
                        help='Number of cross validation iteration, should be >= 2 integer. Default is 5')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed, should be non-negative integer. Default is 1.')
    args = parser.parse_args()
    if args.n_splits < 2:
        raise ValueError('n_splits(k) should be >= 2 integer but {} was given.'.format(args.n_splits))
    if args.seed < 0:
        raise ValueError('seed(s) should be non-negative integer but {} was given.'.format(args.seed))
    return args


def main():
    since = time.time()  # start time

    # get arguments
    args = _get_args()

    # prepare directories and logger
    log_dir = os.path.dirname(args.log_filepath)
    made_log_dir = False
    if not os.path.isdir(log_dir):  # directory for logger
        os.makedirs(log_dir)
        made_log_dir = True
    logger = get_logger(name=__name__, filepath=args.log_filepath)
    if made_log_dir:
        logger.warning('"{}" dose not exist but made automatically'.format(log_dir))    

    if not os.path.isdir(args.model_dir):  # directory for save modesl
        os.makedirs(args.model_dir)
        logger.warning('"{}" dose not exist but made automatically'.format(args.model_dir))
    if not os.path.isdir(args.cv_dir):  # directory for save cross validation results
        os.makedirs(args.cv_dir)
        logger.warning('"{}" dose not exist but made automatically'.format(args.cv_dir))
    logger.info('Random seed is {}.'.format(args.seed))

    # load dataset
    with timer('Load data', logger, DEBUG):
        train = pd.read_csv(args.input)
    logger.info('"{}" have {} records ({} unique `PlaceID`)' \
                .format(args.input, train.shape[0], train.PlaceID.nunique()))
    # make target be more likely to normal distribution (probably there are few outliers)
    assert(train['AverageLandPrice'].min() > 0)
    train['AverageLandPrice'] = np.log(train['AverageLandPrice'])

    # Prepare for GroupKFold
    with timer('Prepare for GroupKFold', logger, DEBUG):
        # use for clustering
        list_of_feature_names = [
            'AverageLandPrice',
            'AverageLandPriceIndex',
            'AverageLandPriceChangeRate',
            'SumLight',
            'SumLightIndex',
            'SumLightChangeRate',
            'MeanLight',
            'MeanLightIndex',
            'MeanLightChangeRate'
        ]
        logger.info('Clustering PlaceID (number of patterns: {})'.format(len(list_of_feature_names)))
        cluster_df = train[['PlaceID', 'Year']].copy()  # Store grouper here for GroupKFold
        for i, feature_name in enumerate(list_of_feature_names):
            clusterer = PlaceIDCluasterer(feature_names=[feature_name],
                                          kmeans_args={'n_clusters': args.n_splits,
                                                       'random_state': args.seed})
            clusterer.fit(train.copy())
            cluster_by_place_id = clusterer.predict(train) \
                                 .reset_index() \
                                 .rename(columns={'ClusterID': f'{feature_name}Cluster'})
        cluster_df = pd.merge(cluster_df, cluster_by_place_id, how='left', on='PlaceID')
        cluster_df.to_csv(os.path.join(args.cv_dir, 'PlaceIDClusteringResult.csv'))
        assert(cluster_df.isnull().sum().sum() == 0)
    
    # Prepare for StratifiedKFold
    n_bins = 5
    labels = list(range(n_bins))
    target_bin = pd.cut(train['AverageLandPrice'].values, n_bins, labels=labels)
    
    # Training
    model_factories = {
        # Models to be trained
        'Ridge': get_ridge,
        'LightGBM': get_lgbm,
        'XGBoost': get_xgb,
        'KNN': get_knn,
        'SVM': get_xgb,
    }
    loss_threshold = 0.7  # determine whether save results or not
    metrics = {}  # store metrics here
    logger.info('Training {} models with {} patterns cross validation.' \
                .format(len(model_factories), cluster_df.shape[1] + 1))
    counter = 0
    for model_name_prefix, model_factory in model_factories.items():
        # GroupKFold
        for c in cluster_df.columns:
            model_name = f'{model_name_prefix}GroupBy{c}'
            counter += 1
            logger.info('{}: start train {}.'.format(counter, model_name))
            model_cv = ModelCV(target_col='AverageLandPrice',
                               feature_names=FEATURE_NAMES,
                               key_columns=['PlaceID', 'Year'],
                               model_factory=model_factory,
                               logger=logger,
                               grouper=cluster_df[c],
                               k=args.n_splits,
                               random_state=args.seed,
                               model_name=model_name)
            with timer(f'Train {model_name}', logger, DEBUG):
                model_cv.fit(train.copy())
            train_mean, train_std, valid_mean, valid_std = model_cv.cv_result_.get_cv_metrics()
            metrics[model_name] = [train_mean, train_std, valid_mean, valid_std]
            # save results if performance is good
            if valid_mean > loss_threshold:
                logger.info('Too bad performance (loss = {:.5f} > {:.5f}), do not save it.' \
                            .format(valid_mean, loss_threshold))
                continue  # next group k-fold
            model_cv.save_results(args.cv_dir, args.model_dir)
            coefficients, importances = extract_importances(model_cv.cv_result_, FEATURE_NAMES)
            if coefficients.shape[1] > 0:
                coefficients.to_csv(os.path.join(args.cv_dir, f'{model_name}Coefficients.csv'))
            if importances.shape[1] > 0:
                importances.to_csv(os.path.join(args.cv_dir, f'{model_name}FeatureImportances.csv'))

        # Stratified k-fold
        model_name = f'{model_name_prefix}Stratified'
        counter += 1
        logger.info('{}: start train {}.'.format(counter, model_name))
        model_cv = ModelCV(target_col='AverageLandPrice',
                           feature_names=FEATURE_NAMES,
                           key_columns=['PlaceID', 'Year'],
                           model_factory=model_factory,
                           logger=logger,
                           stratifier=target_bin,
                           k=args.n_splits,
                           random_state=args.seed,
                           model_name=model_name)
        with timer(f'Train {model_name}', logger, DEBUG):
            model_cv.fit(train)
        train_mean, train_std, valid_mean, valid_std = model_cv.cv_result_.get_cv_metrics()
        metrics[model_name] = [train_mean, train_std, valid_mean, valid_std]
        if valid_mean > loss_threshold:
            logger.info('Too bad performance (loss = {:.5f} > {:.5f}), do not save it.' \
                        .format(valid_mean, loss_threshold))
            continue  # next model
        model_cv.save_results(args.cv_dir, args.model_dir)
        coefficients, importances = extract_importances(model_cv.cv_result_, FEATURE_NAMES)
        if coefficients.shape[1] > 0:
            coefficients.to_csv(os.path.join(args.cv_dir, f'{model_name}Coefficients.csv'))
        if importances.shape[1] > 0:
            importances.to_csv(os.path.join(args.cv_dir, f'{model_name}FeatureImportances.csv'))
    logger.info('Training is complete.')
    # Save metrics
    with open(os.path.join(args.cv_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        f.close()
    logger.info('Complete ({:.5f} seconds passed).'.format(time.time() - since))
    return None


if __name__ == '__main__':
    main()
