import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import preprocessing
import plot
import model_creation
import utility
import evaluation
import hyperparameter_tuning


title = pd.read_csv('with_title.csv')
plot.plot_missing_value_data(title,'partial.png')
TEST_TRAIN_RATIO = 0.2
SEMIFINAL_CSV_FILE = "semifinal.csv"
FINAL_CSV_FILE = "final.csv"
TRANSFORMED_TIME_CSV = "time_trans.csv"
final_csv = pd.read_csv(SEMIFINAL_CSV_FILE, index_col=0)
final_csv = final_csv.drop(['channel_subscribers','views'], axis=1)
temp_csv = preprocessing.remove_nan_rows(final_csv)
plot.plot_missing_value_data(final_csv, "missing_before.png")
plot.plot_missing_value_data(temp_csv, "missing_after.png")
temp_csv = preprocessing.transform_dates_to_day_since_today(temp_csv, OUTPUT_CSV_FILE=TRANSFORMED_TIME_CSV)
# plot.bar_plot_target_features(temp_csv,"before_smote.png")
features = plot.plot_all(temp_csv, "original")
temp_csv = preprocessing.rebalance_dataset(temp_csv)
# plot.bar_plot_target_features(temp_csv, "after_smote.png")
scaled_ds = preprocessing.scale_features(MaxAbsScaler(), temp_csv, features)
features = plot.plot_all(scaled_ds, "scaled")
k_best_ds = preprocessing.perform_feature_selection(scaled_ds, number_features=25, feature_selection_method=preprocessing.FeatureSelectionMethod.SELECT_K_BEST)
tree_based_ds = preprocessing.perform_feature_selection(scaled_ds, number_features=25, feature_selection_method=preprocessing.FeatureSelectionMethod.TREE_BASED)


train_k_best_dataset, test_k_best_dataset = train_test_split(k_best_ds, test_size=TEST_TRAIN_RATIO)
train_tree_best_dataset, test_tree_best_dataset = train_test_split(tree_based_ds, test_size=TEST_TRAIN_RATIO)
print('learning random forest 1')
random_forest_k_best_classifier = model_creation.create_random_forest(train_k_best_dataset)
print('learning random forest 2')
random_forest_tree_best_classifier = model_creation.create_random_forest(train_tree_best_dataset)

print('learning svm 1')
support_vector_machine_k_best_classifier = model_creation.create_support_vector_machine(train_k_best_dataset)
print('learning svm 2')
support_vector_machine_tree_best_classifier = model_creation.create_support_vector_machine(train_tree_best_dataset)


print('learning NN 1')
neural_network_k_best_classifier = model_creation.create_neural_network(train_k_best_dataset)
print('learning NN 2')
neural_network_tree_best_classifier = model_creation.create_neural_network(train_tree_best_dataset)

TUNE_HYPERPARAMETER = False
if TUNE_HYPERPARAMETER:
    print('tuning')
    svm_parameters = {}
    svm_parameters['penalty'] = ['l1','l2']
    svm_parameters['loss'] = ['hinge', 'squared_hinge']
    svm_parameters['dual'] = [True, False]
    svm_parameters['tol'] = [1e-4, 1e-2, 1e-3]
    hyperparameter_tuning.learn_best_hyperparams_grid_search(support_vector_machine_tree_best_classifier, svm_parameters, test_k_best_dataset[0:200], "svm.txt", is_svm=True)
    nn_parameters = {}
    nn_parameters['hidden_layer_sizes'] = [ (200, 300, 500,600, 400, 200, 100), (20,30,50,60,40,20,10), (100,150,250,300,200,100,50)]
    nn_parameters['activation'] = ['relu','logistic','tanh']
    nn_parameters['solver'] =['sgd','adam']
    nn_parameters['alpha'] = [0.0001, 0.001]
    nn_parameters['random_state'] = [10,40]
    nn_parameters['max_iter'] = [1000]

    hyperparameter_tuning.learn_best_hyperparams_grid_search(neural_network_tree_best_classifier,nn_parameters,test_tree_best_dataset[0:100],"mlpclassifier.txt")
    random_forest_params = {}
    random_forest_params['bootstrap'] = [True,False]
    random_forest_params['max_depth'] = [10, 20, 30, None]
    random_forest_params['max_features'] = ['auto', 'sqrt']
    random_forest_params['min_samples_leaf'] = [1,2,4]
    random_forest_params['min_samples_split'] = [2,5,10]
    random_forest_params['n_estimators'] = [400, 600, 800, 1000]
    hyperparameter_tuning.learn_best_hyperparams_grid_search(random_forest_k_best_classifier, random_forest_params, test_k_best_dataset[0:200], 'random_forest.txt')

# print('eval random forest')
# evaluation.full_report(k_best_ds, random_forest_k_best_classifier,"random_forest_k_best.txt")
# evaluation.full_report(tree_based_ds, random_forest_tree_best_classifier,"random_forest_tree_best.txt")
#
# print('eval svm')
# evaluation.full_report(k_best_ds, support_vector_machine_k_best_classifier,"svm_k_best.txt", is_svm=True)
# evaluation.full_report(tree_based_ds, support_vector_machine_tree_best_classifier,"svm_tree_best.txt", is_svm=True)

# print('eval neural network')
evaluation.full_report(k_best_ds, neural_network_k_best_classifier, "mlpclassifier_k_best.txt")
# evaluation.full_report(tree_based_ds, neural_network_tree_best_classifier, "mlpclassifier_tree_best.txt")