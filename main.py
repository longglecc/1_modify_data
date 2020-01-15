
from model.gc_knn import Model_knn
from model.gc_tree import Model_tree
from model.gc_svm import Model_svm
from model.gc_linear import Model_linear
from plot import  Model_plot
from log import  Moddel_log
from model.gc_cluster import Model_cluster
from data_proces.gc_process import read_data_csv,init_df_feature_muti,init_df_cluster,init_df_on_premise,init_df_type_code
import sys



def make_type_code_model(df):

    df_type_dict, df_type_index = init_df_type_code(df)
    for x in df_type_index:
        print(x)
        df_type_code = df_type_dict[x]
        # df_type_code.to_csv("./inter_data/1.csv")
        print(df_type_code.head(5))
        print(df_type_code.shape)
        make_model(df_type_code)


def make_model(df):

    if df.empty:
        print("Warinning: dataset is empty!")
        return 0

    MIN_FEATURE = 7
    MAX_FEATURE = 22
    for i in range(MIN_FEATURE, MAX_FEATURE + 1, 1):
        df_1 = df.copy()
        df_inited = init_df_feature_muti(i, df_1)
        print(df_inited.head(5))
        print(df_inited.shape)


        # df_balance = make_sample_balance(df_inited)
        # print(df_balance.head(5))
        # print(df_balance.shape)

        '''
        model_knn = Model_knn(df_inited,0.33)
        y_test,y_pred = model_tree.exec("KNeighborsClassifier","no")
        '''

        # model_tree = Model_tree(df_inited,0.33)
        # y_test,y_pred = model_tree.exec("DecisionTreeClassifier","no")

        # model_tree = Model_tree(df_inited, 0.33)
        # y_test, y_pred = model_tree.exec("RandomForestClassifier", "no")

        # model_svm = Model_svm(df_inited,0.33)
        # y_test, y_pred = model_svm.exec("sklearn_SVC","no")
        # print(y_pred)
        if df_inited.empty:
            print("Warinning: dataset is empty!")
            return 0

        model_linear = Model_linear(df_inited,0.3)
        y_test,y_pred = model_linear.exec("LogisticRegression",'no')

        model_plot = Model_plot()
        model_plot.plot_show(y_test, y_pred)



if __name__ == "__main__":


    sys.stdout = Moddel_log("log/type_code_run.log", sys.stdout)

    #file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/Uintdist_2020-1-3.csv"
    file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/UnitedDist_AIOrderDuplic.csv"

    #df = read_data(file_path)
    df = read_data_csv(file_path)

    make_type_code_model(df)

    '''
    print(df.head(5))
    df_inited = init_df_feature(22,df)
    print(df_inited.head(5))
    print(df_inited.shape)
    '''
    # df_on = init_df_on_premise(df)
    # df_on_premise = init_df_on_premise(df,"on")



        # make_model(df_type_code)
        # model_linear = Model_linear(df_inited, 0.3)
        # y_test, y_pred = model_linear.exec("LogisticRegression", 'no')
        #
        # model_plot = Model_plot()
        # model_plot.plot_show(y_test, y_pred)


    #2019/1/14
    # df.drop(df.columns[-1], axis=1, inplace=True)
    # df.drop(df.columns[0], axis=1, inplace=True)
    # print(df.columns)
    #
    # label_columns = df.columns
    # type_code_count = df[label_columns].groupby(df[label_columns[1]]).count()
    # type_code_index = type_code_count.index
    # print(type_code_index)
    # type_code_df = df.loc[df[label_columns[1]] != 'Other']




    #type_code_df.to_csv("./inter_data/NotOther_data.csv")
    #if len(type_code_df.valuse)>100:


    # type_code_df = type_code_df.iloc[:, -22:]
    #
    # print(type_code_df.head(5))
    # print(type_code_df.shape)
    #make_model(type_code_df)






    #df_on = init_df_on_premise(df)
    # df_on_premise = init_df_on_premise(df,"on")
    #
    # MIN_FEATURE = 7
    # MAX_FEATURE = 22
    # for i in range(MIN_FEATURE, MAX_FEATURE, 1):
    #     df_1 = df_on_premise.copy()
    #     df_inited = init_df_feature_muti(i, df_1)
    #     print(df_inited.head(5))
    #     print(df_inited.shape)
    #
    #     # df_balance = make_sample_balance(df_inited)
    #     # print(df_balance.head(5))
    #     # print(df_balance.shape)
    #
    #     '''
    #     model_knn = Model_knn(df_inited,0.33)
    #     y_test,y_pred = model_tree.exec("KNeighborsClassifier","no")
    #     '''
    #
    #     # model_tree = Model_tree(df_inited,0.33)
    #     # y_test,y_pred = model_tree.exec("DecisionTreeClassifier","no")
    #
    #     # model_tree = Model_tree(df_inited, 0.33)
    #     # y_test, y_pred = model_tree.exec("RandomForestClassifier", "no")
    #
    #     # model_svm = Model_svm(df_inited,0.33)
    #     # y_test, y_pred = model_svm.exec("sklearn_SVC","no")
    #     # print(y_pred)
    #
    #     model_linear = Model_linear(df_inited, 0.33)
    #     y_test, y_pred = model_linear.exec("LogisticRegression", 'no')
    #
    #     model_plot = Model_plot()
    #     model_plot.plot_show(y_test, y_pred)


    #df_inited = init_df_cluster(df)
    #print(df_inited.head(5))
    #print(df_inited.shape)

    # model_cluster =  Model_cluster(df_on)
    # model_cluster.select_cluster_model("Kmeans")(3,"no")

    #model_cluster = Model_cluster(df_inited)
    #model_cluster.select_cluster_model("MiniBatchKMeans")(2, "no")

    # model_plot = Model_plot()
    # model_plot.plot_cluster_center(df_inited,km0,2)

    #print(df.head(5))

















