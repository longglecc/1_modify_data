
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from model.model_lib import Model_lib


class Model_knn(Model_lib):
    def __init__(self,dataset,p):
        super(Model_knn,self).__init__(dataset,p)

    def select_knn_model(self, case):
        """
        :param case:
        :return:
        """
        fun_name = "knn_" + str(case)
        method = getattr(self, fun_name,self.knn_other)
        return method

    def knn_KNeighborsClassifier(self, param='no'):
        """
        :param param:
        :return:
        """
        print("knn model initing...")
        knf0 = KNeighborsClassifier(30)
        print("knn model fitting...")
        knf0.fit(self.train_x, self.train_y)
        y_pred0 = knf0.predict(self.test_x)
        self._set_test_pred(y_pred0)

        print("ACC Score (Train): %f" % knf0.score(self.train_x, self.train_y))
        print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred0))
        #print(knf0.feature_importances_)

        if param is 'cv':

            random_state = None

            cv_params = {'max_depth': range(1, 50, 1)}
            gbm = GridSearchCV(estimator=KNeighborsClassifier(
                random_state=random_state
            ),
                param_grid=cv_params,
                scoring="accuracy",
                cv=5,
                verbose=True
            )

            gbm.fit(self.train_x, self.train_y)
            print("Best parameters %s" % gbm.best_params_)
            print("Best score %s" % gbm.best_score_)
            max_depth = gbm.best_params_['max_depth']

            knf1 = KNeighborsClassifier(
                max_depth=max_depth,
                random_state=random_state
            )
            knf1.fit(self.train_x, self.train_y)
            #print(rf1.score(self.train_x, self.train_y))
            y_pred1 = knf1.predict(self.test_x)
            self._set_test_pred(y_pred1)
            print("ACC Score (Train): %f" % knf1.score(self.train_x, self.train_y))
            print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred1))


    def knn_XGB(self,param='no'):
        print(param)

    def knn_other(self,param='no'):
        print(param)

    def exec(self,fun_name="default_fun",param="no"):
        self._train_test_split()
        self.select_knn_model(fun_name)(param)
        return self._get_test_pred()



