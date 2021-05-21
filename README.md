# Solafune-landprice-prediction
Solution of [夜間光データから土地価格を予測](https://solafune.com/#/competitions/f03f39cc-597b-4819-b1a5-41479d4b73d6) competition held by Solafune.

# My rank

|LB|RMSE|Rank|
|:---:|---:|---:|
|Public|0.524415|51|
|Private|0.553294|56|

# Solution

## Data cleansing
If a `PlaceID` in training set dosen't have all years' observation (meaning 1992s' observation, ..., 2013s' observation), the all observation of that `PlaceID` is removed. Original training set contains 1018 `PlaceID` but 900 survives after cleansing.

## Feature engineering
Create approximately 110 features.

- `SumLight` / (1 + `MeanLight`)
- Calculate each feature's time series transition by `PlaceID`
  - Ratio to first year (1992s)'s observation
  - Year-to-year change rate
 - Aggregate features by `PlaceID` (sum, mean, median, std, min, max)

## Cross validation (cv)
GroupKFold for `Year` and `PlaceID`, StratifiedKFold for `AverageLandPrice` (5-fold).

## Algorithms
Ensemble of LightGBM, KNN, SVM, XGBoost, and Ridge.

## Hyperparameters
I didn't do any hyperparameter tuning because of lack of time and energy.

# How to use
## Store dataset and modules
Following tree shows expected folder structure.

├── read_only  
│   ├── TrainDataSet.csv  
│   ├── EvaluationData.csv  
│   └── UploadFileTemplate.csv  
├── preprocessor.py  
├── trainer.py  
├── predictor.py  

Competition dataset is stored [here](https://solafune.com/#/competitions/f03f39cc-597b-4819-b1a5-41479d4b73d6).

## Execute modules
Activate virtual environment, then execute 3 modules in correct order. 

```
>>> conda create -n [env_name] --file env_name.txt
>>> python preprocessor.py
>>> python trainer.py
>>> python predictor.py
```

I created virtual environment by miniconda. "env_name.txt" shows required packages. 

With my laptop, `preprocessor.py` takes 36 seconds, `trainer.py` 233 seconds, and `predictor.py` 52 seconds (here is my laptop's spec).

- OS: Microsoft Windows 10 Home
- RAM: 16GB
- CPU: Intel(R) Core(TM) i5-10300HCPU @ 2.5GHz (4-cores)
- No GPU

### preprocessor.py
This module applies data cleansing and feature extraction for "read_only/TrainDataSet.csv" and "read_only/EvaluationData.csv".

The results are stored in "preprocessed/TrainDataPreprocessed.csv" and "preprocessed/EvaluationDataPreprocessed.csv".

### trainer.py
This module performs machine learning, then output model files in "models" folder, cv results in "cv" folder.

Models are stored in pickle format. File name follows the following naming rule.

  <em>Algorithm + "_" + CV method + "_fold" + number of fold + ".model.</em>

  For example, "KNNGroupByYear_fold1.model" indicates that this model uses KNN as a regressor, cv is done by applying GroupKFold for `Year`, and is made in 1st fold. Additionaly, "SVMGroupStratified_fold2.model" indicates that SVM is used as a regressor, cv is done by applying StratifiedKFold, and is made in 3rd fold.

  Be careful that `AverageLandPrice` is log-scaled before training.

CV results are used just for inspecting training result, not for prediction.

### predictor.py
This module predicts `LandPrice` and writes result to "prediction/PredictionForEvaluationDataPreprocessed.csv".
