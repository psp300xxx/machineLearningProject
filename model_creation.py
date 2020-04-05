# returns an already fit classifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
from sklearn import svm
from sklearn import neural_network
from IsViewsGood import ViewsResult
import numpy as np

import utility


def create_random_forest(dataset : pd.DataFrame):
    target_indexes = utility.get_target_index(dataset)
    x_train, y_train = utility.train_target_separation(dataset=dataset, target_indexes=target_indexes)
    classifier = RandomForestClassifier(bootstrap=False, max_depth=10, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=400)
    # classifier = RandomForestClassifier()

    # classifier.fit(x_train, y_train)
    return classifier

def create_support_vector_machine(dataset: pd.DataFrame):
    target_indexes = utility.get_target_index(dataset)
    x_train, y_train = utility.train_target_separation(dataset=dataset, target_indexes=target_indexes)
    y = utility.svm_target_converter(y_test= y_train)
    classifier = svm.LinearSVC(dual=True,loss='squared_hinge', penalty='l2',tol=1e-4)
    # classifier.fit(x_train, y)
    return classifier
def create_neural_network(dataset: pd.DataFrame):
    target_indexes = utility.get_target_index(dataset)
    x_train, y_train = utility.train_target_separation(dataset=dataset, target_indexes=target_indexes)
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100, 150, 250, 300, 200, 100, 50), activation='tanh', alpha=0.001, random_state=40, solver='adam' )
    # classifier.fit(x_train, y_train)
    return classifier
