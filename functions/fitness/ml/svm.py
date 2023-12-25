#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import arange, mean, flatnonzero
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

class SVM:        
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
                
                C = 1
                kernel = "poly"
                degree = 2
                gamma = "scale"
                random_state = 42
                
                if self.grid_search:
                        svm_gs = SVC()                                                  #create a dictionary of all values we want to test for c, kernel, degree, gamma
                        param_grid = {"C":arange(1,10),"kernel":["linear","poly","rbf","sigmoid","precomputed"],
                                      "degree":arange(1,5),"gamma":["scale","auto"]}        
                        svm_gscv = GridSearchCV(svm_gs, param_grid, cv=self.kfold)      #use gridsearch to test all values for c, kernel, degree, gamma
                        svm_gscv.fit(feat, label)                                       #fit model to data
                        C = svm_gscv.best_params_["c"]                  #check top performing c value
                        kernel = svm_gscv.best_params_["kernel"]        #check top performing kernel value
                        degree = svm_gscv.best_params_["degree"]        #check top performing degree value
                        gamma = svm_gscv.best_params_["gamma"]          #check top performing gamma value
                
                clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=random_state)

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
