#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import arange, mean, flatnonzero
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

class KNN:        
        def __init__(self, X_train, X_test, y_train, y_test):
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test

                self.fold = 5
                self.grid_search = False
                self.cross_validation = False
                
        def _fit__(self, solution=None, minmax=0):
                """
                    GridSearchCV works by training our model multiple times on a range of sepecified parameters.
                    That way, we can test our model with each parameter and figure out the optimal values to get the best accuracy results.
                """
                cols = flatnonzero(solution)

                k = 5
                if self.grid_search:
                        knn_gs = KNeighborsClassifier()                                 
                        param_grid = {"n_neighbors": arange(1, 10)}             #create a dictionary of all values we want to test for n_neighbors
                        knn_gscv = GridSearchCV(knn_gs, param_grid, cv=self.kfold)   #use gridsearch to test all values for n_neighbors
                        knn_gscv.fit(feat, label)                               #fit model to data
                        k = knn_gscv.best_params_["n_neighbors"]                #check top performing n_neighbors value
                
                clf = KNeighborsClassifier(n_neighbors=k)

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
