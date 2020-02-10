
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from model.model_lib import Model_lib
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd
import matplotlib.pylab as plt


class Model_tree(Model_lib):
    def __init__(self, dataset, p):
        super(Model_tree,self).__init__(dataset,p)

    def select_tree_model(self, case):
        """
        :param case:
        :return:
        """
        fun_name = "tree_" + str(case)
        method = getattr(self, fun_name,self.tree_other)
        return method

    def tree_DecisionTreeClassifier(self, param='no'):
        """
        :param param:
        :return:
        """
        clf0 = DecisionTreeClassifier(max_depth=8, random_state=0)
        clf0.fit(self.train_x, self.train_y)
        y_pred0 = clf0.predict(self.test_x)
        self._set_test_pred(y_pred0)

        print("ACC Score (Train): %f" % clf0.score(self.train_x, self.train_y))
        print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred0))
        print(clf0.feature_importances_)

        if param is 'cv':

            random_state = None

            cv_params = {'max_depth': range(1, 50, 1)}
            gbm = GridSearchCV(estimator=DecisionTreeClassifier(
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

            rf1 = RandomForestClassifier(
                max_depth=max_depth,
                random_state=random_state
            )
            rf1.fit(self.train_x, self.train_y)
            #print(rf1.score(self.train_x, self.train_y))
            y_pred1 = rf1.predict(self.test_x)
            self._set_test_pred(y_pred1)
            print("ACC Score (Train): %f" % clf0.score(self.train_x, self.train_y))
            print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred1))




    def tree_RandomForestClassifier(self, param='no'):
        """
        :param param:
        :return:
        """
        rf0 = RandomForestClassifier(oob_score=False, random_state=0)
        rf0.fit(self.train_x, self.train_y)
        # print(rf0.score(self.train_x, self.train_y))
        y_pred0 = rf0.predict(self.test_x)
        self._set_test_pred(y_pred0)
        print("ACC Score (Train): %f" % rf0.score(self.train_x, self.train_y))
        print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred0))

        if param is 'cv':
            n_estimators = 100
            learning_rate = 0.1
            criterion = 'gini'
            splitter = 'best'
            max_depth = 8
            min_samples_split = 100
            min_samples_leaf = 20
            max_features = 'sqrt'
            random_state = 0
            oob_score = False

            parameters = {}
            parameters['n_estimators'] = n_estimators
            parameters['criterion'] = criterion
            parameters['splitter'] = splitter
            parameters['learning_rate'] = learning_rate
            parameters['max_depth'] = max_depth
            parameters['min_samples_split'] = min_samples_split
            parameters['min_samples_leaf'] = min_samples_leaf
            parameters['random_state'] = random_state
            parameters['max_features'] = max_features
            parameters['oob_score'] = oob_score

            scores = []
            cv_params = {'n_estimators': range(1, 100, 10)}

            gbm = GridSearchCV(estimator=RandomForestClassifier(
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state,
                oob_score=oob_score
            ),
                param_grid=cv_params,
                scoring="accuracy",  # neg_mean_squared_error
                cv=5,
                verbose=True
            )

            gbm.fit(self.train_x, self.train_y)
            print("Best parameters %s" % gbm.best_params_)
            print("Best score %s" % gbm.best_score_)

            parameters['n_estimators'] = gbm.best_params_['n_estimators']
            n_estimators = gbm.best_params_['n_estimators']

            cv_params = {'max_depth': range(2, 20, 2), 'min_samples_split': range(50, 200, 20)}

            gbm = GridSearchCV(estimator=RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                oob_score=oob_score
            ),
                param_grid=cv_params,
                scoring="accuracy",  # neg_mean_squared_error
                cv=5,
                verbose=True
            )

            gbm.fit(self.train_x, self.train_y)
            print("Best parameters %s" % gbm.best_params_)
            print("Best score %s" % gbm.best_score_)

            parameters['max_depth'] = gbm.best_params_['max_depth']
            parameters['min_samples_split'] = gbm.best_params_['min_samples_split']

            rf1 = RandomForestClassifier(
                n_estimators=parameters['n_estimators'],
                max_depth=parameters['max_depth'],
                min_samples_split=min_samples_split,
                min_samples_leaf=parameters['min_samples_leaf'],
                max_features=parameters['max_features'],
                random_state=parameters['random_state'],
                oob_score=parameters['oob_score']
            )
            rf1.fit(self.train_x, self.train_y)
            print(rf1.score(self.train_x, self.train_y))
            y_pred1 = rf1.predict(self.test_x)
            self._set_test_pred(y_pred1)
            print("ACC Score (Train): %f" % rf1.score(self.train_x, self.train_y))
            print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred1))


    def tree_xgboost(self, param='no'):
        if param is 'no':
            xgb_model = XGBClassifier(
                learning_rate=0.1,
                n_estimators=177,
                max_depth=7,
                min_child_weight=1,
                gamma=0,
                subsample=0.9,
                colsample_bytree=0.6,
                reg_alph=0.001,
                objective='binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27)

            cv_folds = 5
            early_stopping_rounds = 50

            xgb_param = xgb_model.get_xgb_params()
            xgb_data = xgb.DMatrix(self.train_x, self.train_y)
            cv_result = xgb.cv(xgb_param, xgb_data, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=cv_folds,
                                      metrics='auc', early_stopping_rounds=early_stopping_rounds)
            xgb_model.set_params(n_estimators=cv_result.shape[0])

            # Fit the algorithm on the data
            xgb_model.fit(self.train_x, self.train_y, eval_metric='auc')

            # Predict training set:
            train_predictions = xgb_model.predict(self.train_x)
            train_predprob = xgb_model.predict_proba(self.train_x)[:, 1]

            test_predictions = xgb_model.predict(self.test_x)
            test_predprob = xgb_model.predict_proba(self.test_x)[:, 1]
            self._set_test_pred(test_predictions)

            # self._set_test_pred(y_pred1)
            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(self.train_y, train_predictions))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(self.test_y, test_predictions))
            print("AUC Score (Train): %f" % metrics.roc_auc_score(self.train_y, train_predprob))
            print("AUC Score (Test): %f" % metrics.roc_auc_score(self.test_y, test_predprob))

            # print(xgb_model.get_booster().get_fscore())#xgb_model.booster().get_fscore()
            # feat_imp = pd.Series(xgb_model.get_booster().get_fscore()).sort_values(ascending=False)
            # feat_imp.plot(kind='bar', title='Feature Importances')
            # plt.ylabel('Feature Importance Score')
            # plt.show()

        if param is "cv":
            #cv_params = {'max_depth': range(3,15,2), 'min_child_wight': range(1,7,2)}
            #cv_params = {'gamma':[i/10.0 for i in range(0,5)]}
            # cv_params={
            #     'subsample': [i / 10.0 for i in range(6, 10)],
            #     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
            # }
            cv_params = {'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]}
                        #'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}

            gbm = GridSearchCV(estimator=XGBClassifier(
                learning_rate=0.1,
                n_estimators=177,
                max_depth=7,
                min_child_weight=1,
                gamma=0,
                subsample=0.9,
                colsample_bytree=0.6,
                objective='binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27
            ),
                param_grid=cv_params,
                scoring='roc_auc',
                n_jobs=4,
                cv=5
            )
            gbm.fit(self.train_x, self.train_y)
            print("Repoet:")
            print(gbm.cv_results_)
            print("Best parameters %s" % gbm.best_params_)
            print("Best score %s" % gbm.best_score_)



    def tree_other(self, param='no'):
        print(param)

    def exec(self, fun_name="default_fun", param="no"):

        self._train_test_split()
        self.select_tree_model(fun_name)(param)
        return self._get_test_pred()



