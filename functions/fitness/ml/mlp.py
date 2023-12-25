#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import arange, mean, flatnonzero
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

class MLP:        
        def __init__(self, X_train, X_test, y_train, y_test):
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test

                self.kfold = 5
                self.grid_search = False
                self.cross_validation = False
                
        def _fit__(self, solution=None, minmax=0):
                """
                    GridSearchCV works by training our model multiple times on a range of sepecified parameters.
                    That way, we can test our model with each parameter and figure out the optimal values to get the best accuracy results.
                """
                cols = flatnonzero(solution)

                hidden_layer_sizes = (1000,500,100)
                alpha = 0.001
                max_iter = 1000
                random_state = 42
                if self.grid_search:
                        rf_gs = GaussianProcessClassifier()                             #create a dictionary of all values we want to test for c, kernel, degree, gamma
                        param_grid = {"kernel":["linear","poly","rbf","sigmoid","precomputed"]}        
                        rf_gscv = GridSearchCV(rf_gs, param_grid, cv=self.kfold)        #use gridsearch to test all values for n_estimators, criterion, max_depth
                        rf_gscv.fit(feat, label)                                        #fit model to data
                        kernel = rf_gscv.best_params_["kernel"]                         #check top performing n_estimators value
                
                clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter, random_state=random_state)

                train_data = self.X_train[:,cols]
                test_data = self.X_test[:,cols]
                
                clf.fit(train_data, self.y_train)
                
                if self.cross_validation:
                        cv_scores = cross_val_score(clf, self.feat[:,cols], label, cv=self.kfold)
                        err = [1 - x for x in cv_scores]
                else:
                        score = clf.score(test_data, self.y_test)
                        err = 1 - score
                
                return mean(err)
