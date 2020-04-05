import utility
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
import json
import numpy as np
import pandas as pd

def classification_report_to_file(test_dataset, classifier, output_file, is_svm = False):
    target_indexes = utility.get_target_index(test_dataset)
    x_test, y_test = utility.train_target_separation(test_dataset, target_indexes)
    if is_svm:
    #     convert to svm possible result
        y_test = utility.svm_target_converter(y_test=y_test)
    y_predict = classifier.predict(x_test)
    report = classification_report(y_test, y_predict)
    file = open("report/{}".format(output_file), 'w')
    file.write(report)
    file.close()
def full_report(test_dataset : pd.DataFrame, classifier, output_file, is_svm=False):
    file = open("report/k_fold_{}".format(output_file), 'w')
    target_indexes = utility.get_target_index(test_dataset)
    input, output = utility.train_target_separation(test_dataset, target_indexes)
    dict = {}
    dict['accuracies'] = []
    dict['f1s'] = []
    dict['precisions'] = []
    dict['recalls'] = []
    dict['auc'] = []
    if is_svm:
        output = utility.svm_target_converter(output)
        output = pd.DataFrame(output)
    k_folder = KFold(n_splits=10)
    count = 1
    for train, test in k_folder.split(test_dataset):
        print('training {}'.format(count))
        count+=1
        x_train, x_test = input.iloc[train], input.iloc[test]
        y_train, y_test = output.iloc[train], output.iloc[test]
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
        dict['accuracies'].append(accuracy)
        dict['f1s'].append( report['weighted avg']['f1-score'] )
        dict['precisions'].append( report['weighted avg']['precision'] )
        dict['recalls'].append( report['weighted avg']['recall'] )
        try:
            dict['auc'].append( roc_auc_score(y_test, y_predicted) )
        except:
            dict['auc'].append( np.nan )
    dict['mean_accuracy'] = np.array( dict['accuracies'] ).mean()
    dict['mean_f1_score'] = np.array( dict['f1s'] ).mean()
    dict['mean_precision'] = np.array( dict["precisions"] ).mean()
    dict['mean_recall'] = np.array( dict['recalls'] ).mean()
    dict['mean_auc'] = np.nanmean( np.array( dict['auc'] ) )
    file.write( json.dumps(dict) )
    file.close()
