
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from model.model_lib import Model_lib


class Model_tree(Model_lib):
    def __init__(self,dataset,p):
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
        clf0 = DecisionTreeClassifier(max_depth=8,random_state=0)
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


    def tree_XGB(self,param='no'):
        print(param)

    def tree_other(self,param='no'):
        print(param)

    def exec(self,fun_name="default_fun",param="no"):

        self._train_test_split()
        self.select_tree_model(fun_name)(param)
        return self._get_test_pred()



