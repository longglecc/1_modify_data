
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

class Model_cluster(object):

    def __init__(self,data):
        self.data = data

    def select_cluster_model(self, case):
        """
        :param case:
        :return:
        """
        fun_name = "cluster_" + str(case)
        method = getattr(self, fun_name,self.cluster_other)
        return method

    def cluster_Kmeans(self,n=2,param='no'):
        """
        :param param:
        :return:
        """
        km0 = KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,tol=0.0001,precompute_distances='auto',
                               verbose=0,random_state=None,copy_x=True,n_jobs=1,algorithm='auto'
                               )

        km0.fit(self.data)


        # print("ACC Score (Train): %f" % clf0.score(self.train_x, self.train_y))
        # print("ACC Score (Test): %f" % metrics.accuracy_score(self.test_y, y_pred0))
        # print(clf0.feature_importances_)

        #手肘法
        # SSE=[]
        # for k in range(1, n):
        #     km1 = KMeans(n_clusters=k)  #构造聚类器
        #     km1.fit(np.array(self.data))
        #     print(km1.inertia_)
        #     SSE.append(km1.inertia_)
        # return SSE

        #轮廓系数
        # Scores=[]
        # for k in range(2, 9):
        #     km1 = KMeans(n_clusters=k)  #构造聚类器
        #     km1.fit(np.array(self.data))
        #     print(silhouette_score(np.array(self.data), km1.labels_, metric='euclidean'))
        #     Scores.append(silhouette_score(np.array(self.data), km1.labels_, metric='euclidean'))
        #
        # return Scores


    def cluster_XGB(self,param='no'):
        print(param)

    def cluster_other(self,param='no'):
        print(param)




