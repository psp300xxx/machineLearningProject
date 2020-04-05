from IsViewsGood import ViewsResult
import datetime
import pandas as pd
import numpy as np
from imblearn import over_sampling
import utility
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from enum import Enum
# I transform the ratio into the class I will predict in my model
class FeatureSelectionMethod(Enum):
    SELECT_K_BEST = 0
    TREE_BASED = 1
def video_views_res(ratio_arr):
    res = []
    for i in ratio_arr:
        if i < 0.1:
            res.append(ViewsResult.POOR)
        elif i >= 0.1 and i<1:
            res.append(ViewsResult.MEDIUM)
        else:
            res.append(ViewsResult.GOOD)
    return res
def days_between(d1, d2):
    return abs((d2 - d1).days)
# in order to have simpler use of the publish_date, I transform it into the days from that date into today
# I also add a new feature into the dataset (Is week end).
def transform_dates_to_day_since_today(dataset,OUTPUT_CSV_FILE = "temp.csv"):
    days = []
    is_published_in_weekend = []
    publish_times = dataset["publish_time"]
    date_format = "%Y-%m-%d"
    for i in publish_times:
        date = i[0:10]
        date = datetime.datetime.strptime(date, date_format)
        is_published_in_weekend.append( date.weekday()>=5 )
        days_passed = days_between(datetime.datetime.now(), date)
        days.append(days_passed)
    np_days_since_publish_dates = np.array(days)
    np_is_weekend = np.array(is_published_in_weekend)
    new_dataset = dataset.drop(["publish_time"], axis=1)
    index = 2
    new_dataset.insert(index, "days_since_publish_date", np_days_since_publish_dates)
    new_dataset.insert( index, "is_published_in_week_end", np_is_weekend )
    new_dataset.to_csv(OUTPUT_CSV_FILE)
    return new_dataset
#    returns an array of boolean, the boolean indicates if the instance has NaN values inside
def has_NaN_mask(dataset):
    return dataset["channel_1_entropy"].notnull()
def scale_features(scaler, dataset, features):
    encoded = pd.get_dummies(data=dataset, columns=["is_good_result", "category_id"])
    return pd.DataFrame(scaler.fit_transform(encoded), columns=encoded.columns)
def rebalance_dataset(dataset):
    balancer = over_sampling.SMOTE(random_state=40)
    is_good_index = 0
    x_train = dataset.iloc[:,is_good_index+1::]
    y_train = dataset.iloc[:, is_good_index:is_good_index+1]
    dataset_res, y_dataset_res = balancer.fit_resample(x_train, np.array(y_train).ravel())
    dict = {}
    for i in y_dataset_res:
        value = ViewsResult.init(i)
        if value in dict:
            dict[value] +=1
        else:
            dict[value] = 1
    dataset_res.insert(0, "is_good_result", y_dataset_res)
    return dataset_res
def remove_nan_rows(dataset):
    mask = has_NaN_mask(dataset)
    temp_csv = dataset[mask]
    return temp_csv
def perform_feature_selection(dataset, number_features = 25, feature_selection_method = FeatureSelectionMethod.SELECT_K_BEST):
    target_indexes = utility.get_target_index(dataset)
    input, output = utility.train_target_separation(dataset, target_indexes)
    if feature_selection_method == FeatureSelectionMethod.SELECT_K_BEST:
        selector = SelectKBest(chi2, k = number_features)
        selector.fit(input, output)
        cols = selector.get_support(indices=True)
        x_new = input.iloc[:, cols]
        new_dataset = x_new.join(output)
        return new_dataset
    elif feature_selection_method == FeatureSelectionMethod.TREE_BASED:
        temp_classifier = ExtraTreesClassifier(n_estimators=200)
        temp_classifier.fit(input, output)
        model = SelectFromModel(temp_classifier, prefit=True)
        cols = model.get_support(indices=True)
        x_new = input.iloc[:, cols]
        new_dataset = x_new.join(output)
        return new_dataset