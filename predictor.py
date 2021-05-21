import argparse
from collections import defaultdict
import glob
import logging
import os
import os.path
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import get_logger, timer, ColumnSelector

MAX_PRICE = 120000.
MIN_PRICE = 5.

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=
        'This module will output prediction result of given feature files and models.')
    parser.add_argument('-i', '--input_files', type=str, nargs='+',
                        default=[os.path.join(os.getcwd(), 'preprocessed', 'EvaluationDataPreprocessed.csv')],
                        help='Filepath of dataset to be processed. By default, this module processes only '
                        '"<current_directory>/preprocessed/EvaluationDataPreprocessed.csv".')
    parser.add_argument('-m', '--model_dir', type=str, default=os.path.join(os.getcwd(), 'models'),
                        help='Directory in which prediction models are stored. Default is '
                        '"<current_directry>/models".')
    parser.add_argument('-p', '--pred_dir', type=str, default=os.path.join(os.getcwd(), 'prediction'),
                        help='Directory for which this module write the prediction result. This module will '
                        'try to make directory if given directory dose not exist. Default is '
                        '"<current_directry>/prediction".')
    parser.add_argument('-l', '--log_filepath', type=str, default=os.path.join(os.getcwd(), 'log', 'prediction.log'),
                        help='Log filepath. This module will try to make directory if given directory dose not exist. '
                        'Default is "<current_directry>/log/prediction.log"')
    parser.add_argument('-t', '--submission_template', type=str,
                        default=os.path.join(os.getcwd(), 'read_only', 'UploadFileTemplate.csv'),
                        help='Filepath of upload file template. Prediction results will be sorted following '
                        'to the order of this template. Template file should be camma separated UTF-8 file, '
                        'have a header line at the top row, and contain columns named "PlaceID" and "Year".')
    parser.add_argument('-n', '--no_template', action='store_true',
                        help='If given, prediction results dose not sorted. They will be output in original order.')
    parser.add_argument('-a', '--write_all', action='store_true',
                        help='If given, not only prediction results but intermediate prediction data will be output'
                       ' to pred_dir.')

    args = parser.parse_args()
    return args
    

def main():
    since = time.time()  # start time

    # Get arguments
    args = _get_args()

    # Prepare directories and logger
    log_dir = os.path.dirname(args.log_filepath)
    made_log_dir = False
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        made_log_dir = True
    logger = get_logger(name=__name__, filepath=args.log_filepath)
    if made_log_dir:
        logger.warning('{} dose not exist but made automatically'.format(log_dir))
    if not os.path.isdir(args.pred_dir):
        os.makedirs(args.pred_dir)
        logger.warning('{} dose not exist but made automatically'.format(args.pred_dir))

    # Do given files really exist ?
    not_existing_files = [f for f in args.input_files if not os.path.isfile(f)]
    if not_existing_files:  # partialy yes.
        logger.warning('They are not existing files: {}'.format(", ".join(not_existing_files)))
    existing_files = [f for f in args.input_files if f not in not_existing_files]
    if not existing_files:  # no!
        raise FileNotFoundError('All of the given input files do not exist.')

    # Is there any model files in given directory?
    if not os.path.isdir(args.model_dir):  # no!
        raise NotADirectoryError('{} is not a existing directory.'.format(args.model_dir))
    model_filepaths = glob.glob(os.path.join(args.model_dir, '*.model'))
    if not model_filepaths:  # no!
        raise FileNotFoundError('There are no model file in {}.'.format(args.model_dir))
    else:  # yes!
        logger.info('Found {} model files in {}'.format(len(model_filepaths), args.model_dir))

    if not args.no_template:
        template = pd.read_csv(args.submission_template, usecols=['PlaceID', 'Year'])
        assert(template.duplicated(subset = ['PlaceID', 'Year']).sum() == 0)
        assert(template.isnull().sum().sum() == 0)
    else:
        logger.info('Template file is not used.')

    # Load model files.
    logger.info('Load models')
    models = {}  # store model object by file
    columns_by_model = defaultdict(list)  # store dataframe's column names
    ## Load one by one.
    for filepath in model_filepaths:
        with timer(f'Load {filepath}', logger, logging.DEBUG):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                f.close()
        filename = os.path.basename(filepath)
        models[filename] = model
        model_name = filename.split('_fold')[0]
        columns_by_model[model_name].append(filename)
    # Make prediction for each input file.
    with timer('Prediction loop', logger, logging.DEBUG):
        logger.info('Process {} files.'.format(len(existing_files)))
        for i, input_file in enumerate(existing_files):
            logger.info('{}: Start prediction process of  {}.'.format(i + 1, input_file))
            feature = pd.read_csv(input_file)
            # Obtain prediction results of the all model files.
            pred_by_model_file = pd.DataFrame()
            for filename, model in models.items():
                with timer(f'Predict by {filename}', logger, logging.INFO):
                    pred_by_model_file[filename] = model.predict(feature.copy())
            # Average prediction results by model.
            pred_by_model = pd.DataFrame()
            for model_name, columns in columns_by_model.items():
                pred_by_model[model_name] = pred_by_model_file[columns].mean(axis=1).values
            # Averege each model's prediction result.
            prediction = feature[['PlaceID', 'Year']].copy()
            prediction['LandPrice'] = pred_by_model.mean(axis=1).values
            # Make output file.
            out_filepath = os.path.join(args.pred_dir, f'PredictionFor{os.path.basename(input_file)}')
            if not args.no_template:
                # Sort prediction result following to template file
                prediction = pd.merge(template.copy(), prediction, on=['PlaceID', 'Year'], how='left')
                assert(prediction.duplicated(subset = ['PlaceID', 'Year']).sum() == 0)
            # transform into original scale then clipping
            prediction['LandPrice'] = np.clip(np.exp(prediction['LandPrice']), a_min=MIN_PRICE, a_max=MAX_PRICE)
            prediction.to_csv(out_filepath, index=False)
            if args.write_all:
                pred_by_model_file = np.exp(pred_by_model_file).clip(lower=MIN_PRICE, upper=MAX_PRICE)
                pred_by_model = np.exp(pred_by_model).clip(lower=MIN_PRICE, upper=MAX_PRICE)
                pred_by_model_file.to_csv(
                    os.path.join(args.pred_dir, f'PredByModelFileFor{os.path.basename(input_file)}'),
                    index=False)
                pred_by_model.to_csv(
                    os.path.join(args.pred_dir, f'PredByModelFor{os.path.basename(input_file)}'),
                    index=False)
    logger.info('Complete ({:.5f} seconds passed).'.format(time.time() - since))

if __name__ == '__main__':
    main()
