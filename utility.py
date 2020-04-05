from IsViewsGood import ViewsResult
import pandas as pd
import numpy as np
# this value is returned into a dictionary
def target_instances_number(csv):
    result = csv["is_good_result"]
    dict = {}
    for i in result:
        value = ViewsResult.init( str(i) )
        if value in dict:
            dict[value] = dict[value] + 1
        else:
            dict[value] = 1
    return dict
def get_target_index(dataset, target_name_selector = "is_good_result"):
    indexes = []
    count = 0
    for i in dataset:
        col_string = str(i)
        if target_name_selector in col_string:
            indexes.append(count)
        count += 1
    return indexes

def train_target_separation(dataset : pd.DataFrame, target_indexes):
    y_train = dataset.iloc[:,target_indexes]
    non_target_indexes = []
    for i in range( len(dataset.iloc[0,:]) ):
        if not i in target_indexes:
            non_target_indexes.append(i)
    x_train = dataset.iloc[:,non_target_indexes]
    return x_train, y_train
def svm_target_converter(y_test):
    y = []
    for i in y_test.iterrows():
        if i[1][0]>0.0:
            y.append(ViewsResult.GOOD.value)
        elif i[1][1]>0.0:
            y.append(ViewsResult.MEDIUM.value)
        else:
            y.append(ViewsResult.POOR.value)
    return y