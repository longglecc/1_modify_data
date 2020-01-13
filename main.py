
from model.gc_knn import Model_knn
from model.gc_tree import Model_tree
from plot import  Model_plot
from log import  Moddel_log
from model.gc_cluster import Model_cluster
from data_proces.gc_process import read_data_csv,init_df_feature_muti,init_df_cluster
from sys import stdin,stdout


if __name__ == "__main__":

    stdout = Moddel_log("./log/run.log", stdout)

    #file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/Uintdist_2020-1-3.csv"
    file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/UnitedDist_AIOrderDuplic.csv"

    #df = read_data(file_path)
    df = read_data_csv(file_path)

    '''
    print(df.head(5))
    df_inited = init_df_feature(22,df)
    print(df_inited.head(5))
    print(df_inited.shape)
    '''

    df_inited = init_df_cluster(df)
    print(df_inited.head(5))
    print(df_inited.shape)

    model_cluster =  Model_cluster(df_inited)
    sse = model_cluster.select_cluster_model("Kmeans")(9,"no")

    model_plot = Model_plot()
    model_plot.plot_cluster(sse,9)

    #print(df.head(5))
    # for i in range(7,8,1):
    #
    #     df_1 = df.copy()
    #     df_inited = init_df_feature_muti(i,df_1)
    #     print(df_inited.head(5))
    #     print(df_inited.shape)
    #
    #     #df_balance = make_sample_balance(df_inited)
    #     #print(df_balance.head(5))
    #     #print(df_balance.shape)
    #
    #     '''
    #     model_knn = Model_knn(df_inited,0.33)
    #     y_test,y_pred = model_tree.exec("KNeighborsClassifier","no")
    #     '''
    #
    #
    #     model_tree = Model_tree(df_inited,0.33)
    #     y_test,y_pred = model_tree.exec("DecisionTreeClassifier","no")
    #
    #
    #     '''
    #     model_tree = Model_tree(df_balance,0.33)
    #     y_test,y_pred = model_tree.exec("RandomForestClassifier","no")
    #     '''
    #
    #     '''
    #     model_svm = Model_svm(df_inited,0.33)
    #     y_test, y_pred = model_svm.exec("sklearn_SVC")("no")
    #     '''
    #
    #     '''
    #     model_linear = Model_linear(df_balance,0.33)
    #     y_test,y_pred = model_linear.exec("LogisticRegression")('no')
    #     '''
    #
    #     model_plot = Model_plot()
    #     model_plot.plot_show(y_test,y_pred)
















