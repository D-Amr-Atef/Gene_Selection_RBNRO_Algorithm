#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import arange, mean, shape, flatnonzero
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

class RandomForest:        
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

                n_estimators = 10
                criterion = "gini"
                max_depth = 5
                max_features = 1
                random_state=42
                if self.grid_search:
                        rf_gs = RandomForestClassifier()                             #create a dictionary of all values we want to test for c, kernel, degree, gamma
                        param_grid = {"n_estimators":arange(10,100,10),
                                      "criterion":["gini","entropy"],
                                      "max_depth":arange(1,10),
                                      "max_features": arange(1,shape(self.X_train)[1]),
                                      "random_state": [0,1]}        
                        rf_gscv = GridSearchCV(rf_gs, param_grid, cv=self.kfold)        #use gridsearch to test all values for n_estimators, criterion, max_depth
                        rf_gscv.fit(feat, label)                                        #fit model to data
                        n_estimators = rf_gscv.best_params_["n_estimators"]             #check top performing n_estimators value
                        criterion = rf_gscv.best_params_["criterion"]                   #check top performing criterion
                        max_depth = rf_gscv.best_params_["max_depth"]                   #check top performing max_depth value
                        max_features = rf_gscv.best_params_["max_features"]             #check top performing max_features value
                        random_state = rf_gscv.best_params_["random_state"]             #check top performing random_state value
                
                clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features, random_state=random_state)

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
