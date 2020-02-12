
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

class Model_lib(object):

    def __init__(self, dataset, p):
        self.data = dataset
        self.p = p
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.pred = []

    def _train_test_split(self):
        """
        :return:
        """
        if self.data.empty:
            print("dataset is empty")
            return 0

        values = self.data.values
        values_x, values_y = values[:, :-1], values[:, -1]

        #pca 降维
        #pca = PCA(whiten=True)
        #pca.fit(values_x)
        #ix = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0][0]
        #pca = PCA(n_components=ix, whiten=True)
        #values_x = pca.fit_transform(values_x)
        #print(values_x.shape)

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # values_x = scaler.fit_transform(values_x)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(values_x, values_y, test_size=self.p,
                                                                                random_state=42)
        print(self.train_x.shape, self.train_y.shape)
        print(self.test_x.shape, self.test_y.shape)

        # split_p = int(self.p * len(values) - 1)
        # self.train_x, self.test_x = values_x[:split_p, :], values_x[split_p:, :]
        # self.train_y, self.test_y = values_y[:split_p], values_y[split_p:]
        # # train,text = values[:p,:],values[p:,:]
        # # train_x,train_y = train[1:,:],train[:-1,-1]
        # print(self.train_x.shape, self.train_y.shape)
        # print(self.test_x.shape, self.test_y.shape)

    def _set_test_pred(self, y_pred):
        self.pred = y_pred

    def _get_test_pred(self):
        return self.test_y,self.pred

    def get_test_data(self):
        return np.concatenate((self.test_x, np.array([self.test_y]).T,np.array([self.pred]).T), axis=1)


    def exec(self, fun_name="default_fun", param="no"):
        print(fun_name+'('+param+')')
