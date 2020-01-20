
# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


class Model_plot(object):

    def __init__(self):
        self.labels = ['0', '1']

    '''
    具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
    去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
    是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
    个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
    数字）。
    同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
    你训练好的网络预测出来的预测label。
    这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
    label和预测label分别保存到y_true和y_pred这两个变量中即可。
    '''

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.get_cmap('gray_r')):#Blues #gray_r
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(self.labels)))
        plt.xticks(xlocations, self.labels)#rotation=90
        plt.yticks(xlocations, self.labels)
        plt.ylabel('Test label')
        plt.xlabel('Predicted label')

    def plot_show(self,y_true,y_pred):
        tick_marks = np.array(range(len(self.labels))) + 0.5
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.copy().astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        # cm = np.flipud(cm)
        # cm_normalized = np.flipud(cm_normalized)
        print(cm)
        print(cm_normalized)
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(self.labels))
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c=cm[y_val][x_val]
            c_float = cm_normalized[y_val][x_val]
            plt.text(x_val, y_val, "sum: %d, acc: %0.02f" % (c, c_float), color='red', fontsize=10, va='center', ha='center')

        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        self.plot_confusion_matrix(cm, title='Normalized confusion matrix')
        # show confusion matrix
        #TODO(gaolongcc)
        plt.savefig('./fig/test_confusion_matrix.jpg', format='png')
        plt.show()


    '''
    计算比例
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print cm_normalized
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

    '''

    def plot_cluster(self, sse, n=2):
        X = range(1, n)
        # plt.xlabel('k')
        # plt.ylabel('SSE')
        # plt.plot(X, sse, 'o-')
        plt.xlabel('k')
        plt.ylabel('轮廓系数')
        plt.plot(X, sse, 'o-')

        #plt.savefig('./fig/test_hand_kmeans.jpg', format='png')
        plt.savefig('./fig/test_s_kmeans.jpg', format='png')
        plt.show()

    # def plot_confusion_matrix_1(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.get_cmap('Blues')):
    #     """
    #     This function prints and plots the confusion matrix.
    #     Normalization can be applied by setting `normalize=True`.
    #     Input
    #     - cm : 计算出的混淆矩阵的值
    #     - classes : 混淆矩阵中每一行每一列对应的列
    #     - normalize : True:显示百分比, False:显示个数
    #     """
    #     if normalize:
    #         cm_normalized = cm.copy().astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    #     else:
    #         print('Confusion matrix, without normalization')
    #
    #     print(cm)
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes)#rotation=45
    #     plt.yticks(tick_marks, classes)
    #     fmt = '.2f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')

