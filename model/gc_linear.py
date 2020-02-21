
from sklearn.linear_model import LogisticRegression
from model.model_lib import Model_lib
from sklearn import metrics
import time


class Model_linear(Model_lib):

    def __init__(self, dataset, p):
        super(Model_linear, self).__init__(dataset, p)


    def select_linear_model(self, case):
        """
        :param case:
        :return:
        """
        fun_name = "linear_" + str(case)
        method = getattr(self, fun_name, self.linear_other)
        return method

    def linear_LogisticRegression(self, param='no'):
        """
        :param param:
        :return:
        """
        print("linear model initing...")
        lr0 = LogisticRegression(penalty = 'l2', tol = 0.0001, C = 1.0,solver = 'liblinear', max_iter = 100, multi_class = 'ovr',
                                 verbose = 0, warm_start = False, n_jobs = 1
                                 )
        print("linear model fitting...")
        fit_start_time = time.time()
        lr0.fit(self.train_x, self.train_y)
        fit_end_time = time.time()
        fit_time = round(fit_end_time - fit_start_time,2)
        print("model fit cost time:{} s".format(fit_time))

        pred_start_time = time.time()
        y_pred1 = lr0.predict(self.test_x)
        pred_end_time = time.time()
        pred_time = round(pred_end_time - pred_start_time, 2)
        print('model pred cost time:{} s'.format(pred_time))
        print('model cost time {} s'.format(round(pred_end_time - fit_start_time, 2)))

        self._set_test_pred(y_pred1)
        #print(lr0.score(self.test_x, self.test_y))
        print("ACC Score (Train): %f" % lr0.score(self.train_x, self.train_y))
        print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred1))
        print(lr0.coef_)

        if param is 'cv':
            pass
            # TODO(gaolongcc):


    def linear_xgboost(self, param='no'):
        print(param)
        # TODO(gaolongcc):

    def linear_other(self, param='no'):
        print(param)
        # TODO(gaolongcc):

    def exec(self,fun_name="default_fun", param="no"):

        self._train_test_split()
        self.select_linear_model(fun_name)(param)
        return  self._get_test_pred()