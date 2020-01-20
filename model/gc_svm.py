
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from model.model_lib import Model_lib

class Model_svm(Model_lib):

    def __init__(self, dataset, p):
        super(Model_svm,self).__init__(dataset, p)

    def select_svm_model(self, case):
        """
        :param case:
        :return:
        """
        fun_name = "svm_" + str(case)
        print(fun_name)
        method = getattr(self, fun_name, self.svm_other)
        return method

    def svm_sklearn_SVC(self, param='no'):
        """
        :param param:
        :return:
        """
        '''
        svc = SVC(C=1.0,kernel='rbf',degree=3,gamma='scale',coef0=0.0, shrinking=True,
            probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
            decision_function_shape='ovr', break_ties=False, random_state=None)
        '''

        print("svm model initing...")
        svc0 = SVC(degree=2, gamma=1.0, C=100, kernel='rbf')#C=1.0, kernel='linear', gamma='auto',decision_function_shape='ovr', random_state=0
        print("svm model fiting...")
        svc0.fit(self.train_x,self.train_y)
        y_pred0 = svc0.predict(self.test_x)
        self._set_test_pred(y_pred0)

        print("ACC Score (Train): %f" % svc0.score(self.train_x, self.train_y))
        print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred0))
        #print('支持向量：', svc.support_vectors_)
        #print('支持向量索引：', svc.support_)
        #print('支持向量数量：', svc.n_support_)

        if param is 'cv':

            decision_function_shape = 'ovr'
            random_state = 0

            cv_params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            gbm = GridSearchCV(estimator=SVC(
                decision_function_shape=decision_function_shape,
                random_state = random_state
            ),
                param_grid=cv_params,
                scoring="accuracy",
                cv=5,
                verbose=True
            )
            gbm.fit(self.train_x, self.train_y)
            print("Best parameters %s" % gbm.best_params_)
            print("Best score %s" % gbm.best_score_)

            print()
            print("Grid scores on development set:")
            print()

            means = gbm.cv_results_['mean_test_score']
            stds = gbm.cv_results_['std_test_score']

            for mean, std, params in zip(means, stds, gbm.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            C=gbm.best_params_['C']
            kernel=gbm.best_params_['kernel']
            gamma = gbm.best_params_['gamma']

            svc1 = SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape=decision_function_shape, random_state=random_state)
            svc1.fit(self.train_x, self.train_y, sample_weight=None)
            y_pred1 = svc1.predict(self.test_x)
            self._set_test_pred(y_pred1)
            print("ACC Score (Train): %f" % svc1.score(self.train_x, self.train_y))
            print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred1))



    def svm_xgboost_SVC(self, param='no'):
        print(param)

    def svm_other(self, param='no'):
        print(param)

    def exec(self, fun_name="default_fun", param="no"):

        self._train_test_split()
        self.select_svm_model(fun_name)(param)
        return self._get_test_pred()