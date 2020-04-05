from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import utility
from enum import Enum
class HyperParameterTuningMethod(Enum):
    GRID_SEARCH = 0
    RANDOMIZED = 1
def learn_best_hyperparams_grid_search(original_classifier, params, x_test, output_filename, is_svm = False, tuningMethod = HyperParameterTuningMethod.GRID_SEARCH):
    target_indexes = utility.get_target_index(x_test)
    x_test, y_train = utility.train_target_separation(x_test, target_indexes=target_indexes)
    if tuningMethod == HyperParameterTuningMethod.GRID_SEARCH:
        param_classifier = GridSearchCV(original_classifier, params, n_jobs=-1, cv=3, scoring='accuracy', verbose=10)
    else:
        param_classifier = RandomizedSearchCV(original_classifier, params, n_jobs=1, cv=3, scoring='accuracy', verbose=10)
    if is_svm:
        y_train = utility.svm_target_converter(y_train)
    param_classifier.fit(x_test, y_train)
    file = open("hyperparams/{}".format(output_filename), "w")
    file.write( str(param_classifier.best_estimator_) )
    file.write("\n")
    file.write( str( param_classifier.best_params_) )
    file.close()
    return param_classifier.best_estimator_