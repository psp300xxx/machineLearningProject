import matplotlib.pyplot as plt
from missingno import missingno
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from IsViewsGood import ViewsResult
from utility import target_instances_number, svm_target_converter
def bar_plot_target_features(csv, image_name):
    dict = target_instances_number(csv)
    plt.figure()
    labels = ["POOR", "MEDIUM", "GOOD"]
    y_pos = np.arange(len(labels))
    values = [dict[ViewsResult.POOR], dict[ViewsResult.MEDIUM], dict[ViewsResult.GOOD]]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Values')
    plt.title('Instances for target features')
    plt.savefig("balancing/{}".format(image_name))
    plt.close()
def plot_features_values(strings, dataset : pd.DataFrame, imageName):
    plt.figure()
    n_bins = 100
    count = 0
    fig, axs = plt.subplots(1, len(strings), sharey=True, tight_layout=True)
    for i in strings:
        subseries = dataset[i]
        sorted = subseries.sort_values()
        # We can set the number of bins with the `bins` kwarg
        axs[count].hist(sorted, bins=n_bins)
        axs[count].set_title(i)
        count = (count+1)
    plt.savefig("feature_distributions/{}".format(imageName))
    plt.close()

def plot_all(dataset, directory_name):
    features = ["total_entropy", "total_mean", "total_standard_deviation"]
    plot_features_values(features, dataset, "{}/features_total.png".format(directory_name))
    for index in range(3):
        features_names = ["channel_{}_entropy".format(index + 1), "channel_{}_mean".format(index + 1),
                          "channel_{}_standard_deviations".format(index + 1)]
        plot_features_values(features_names, dataset, "{}/features{}.png".format(directory_name,index + 1))
        features = features + features_names
    title_features = ["title_length", "title_longest_word_length", "days_since_publish_date"]
    plot_features_values(title_features, dataset, "{}/title_len.png".format(directory_name))
    features = features + title_features
    return features
def plot_missing_value_data(dataset, image_name):
    res = missingno.bar(dataset)
    plt.savefig("missings/{}".format(image_name))
    plt.close()
def plot_roc(tpr,fpr):
    plt.figure()

    plt.close()
def get_feature_importance(data):
    classifier = ExtraTreesClassifier(n_estimators=100)
    is_good_index = 27
    y = data.iloc[:, is_good_index:is_good_index+3]
    x = data.iloc[:,0:is_good_index]
    x = x.join( data.iloc[:,is_good_index+3::] )
    classifier.fit(x,y)
    importances = classifier.feature_importances_
    std = np.std( [  tree.feature_importances_ for tree in classifier.estimators_ ], axis=0 )
    indices = np.argsort(importances)[::-1]
    plot_features_importances(indices=indices, std=std, image_name="feature_distributions/feature_importance.png",x=x)
def plot_features_importances(indices,std, image_name,x):
    fig, ax = plt.subplots(figsize=(20, 8))
    std.sort(axis=0)
    ax.bar([str(i) for i in indices], std)
    coloumns_ordered_named = []
    for i in indices:
        coloumns_ordered_named.append( x.columns[i] )
    ax.set_xticklabels(coloumns_ordered_named, rotation=65)
    ax.set_ylabel("importance")
    plt.savefig(image_name, bbox_inches='tight')
    plt.close()