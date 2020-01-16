
from model.gc_knn import Model_knn
from model.gc_tree import Model_tree
from model.gc_svm import Model_svm
from model.gc_linear import Model_linear
from plot import  Model_plot
from log import  Moddel_log
from model.gc_cluster import Model_cluster
from data_proces.gc_process import read_data_csv,init_df_feature_muti,init_df_type_code,get_label_count
import sys
import numpy as np
import pandas as dp



def make_lag_no_balance_dec(df_lag_no,lag_len):

    if df_lag_no.empty:
       print("Warinning: dataset is empty!")
       return 0

    lag_no_len = df_lag_no.shape[0]
    #print(lag_no_len)
    #print(df_lag_no.index.to_list()[-1])

    drop_indices = np.random.choice(df_lag_no.index[-1], lag_no_len - lag_len, replace=False)
    df_drop = df_lag_no.drop(drop_indices)
    # print(df_drop.head(6))
    # print(df_drop.shape)

    df_drop.reset_index(drop=True, inplace=True)
    #print(df_drop.head(5))
    #print(df_drop.shape)
    make_model(df_drop)

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

    label_columns = df.columns.to_list()
    if label_columns[-1] != "label":
        df.drop(df.columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df.drop(df.columns[0], axis=1, inplace=True)

    MIN_FEATURE = 7
    MAX_FEATURE = 22
    for i in range(MIN_FEATURE, MAX_FEATURE + 1, 1):
        df_1 = df.copy()
        df_inited = init_df_feature_muti(i, df_1)
        print(df_inited.head(5))
        print(df_inited.shape)

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


    sys.stdout = Moddel_log("log/run.log", sys.stdout)

    #file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/Uintdist_2020-1-3.csv"
    #完整数据的路径
    #file_path = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/UnitedDist_AIOrderDuplic.csv"

    #NoOther data path
    lag_no_file_path = "./inter_data/NotOther_data.csv"
    lag_file_path = "./inter_data/Other_data.csv"
    #df = read_data(file_path)

    df_lag_no = read_data_csv(lag_no_file_path)
    #df_lag = read_data_csv(lag_file_path)

    #lag_len = df_lag.shape[0]
    lag_no_len =df_lag_no.shape[0]

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    # if label_columns[0] != "feature1":
    #     df_lag_no = df_lag_no.iloc[:, -22:]

    print(df_lag_no.head(5))
    print(df_lag_no.shape)

    #df_del_zero =  df_lag_no.loc[(df_lag_no==0).all(axis=1),:]  #删除
    #print(df_del_zero.head(5))
    #print(df_del_zero.shape)

    #去掉重复的有7万条数据，并且训练数据的得分并不高0.67
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    print(label_columns[-22:-1])
    print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    #df_del_zero = df_del_same.loc[df_del_same[label_columns[-22:-1]]]
    print(df_del_zero.head(5))
    print(df_del_zero.shape)


    #make_model(df_del_same)
    #扩展纬度
    df_data = df_del_zero.iloc[:, -22:]
    df_data_buff = df_data.copy()
    print(df_data.head(5))
    print(df_data.shape)

    df_tex  =df_del_zero.iloc[:,:-22]
    df_tex_buff = df_tex.copy()
    print(df_tex.head(5))
    print(df_tex.shape)

    for i in range(1,10, 1):
        df_shift = df_data.shift(periods=i, axis=1)
        df_data_buff = dp.concat([df_data_buff, df_shift])
        df_tex_buff = dp.concat([df_tex_buff,df_tex])

    df_data_buff.dropna(axis=1, how='any', inplace=True)
    print(df_data_buff.head(5))
    print(df_data_buff.shape)
    df_data_buff = df_data_buff.astype("int64")
    print(df_data_buff.head(5))
    print(df_data_buff.shape)
    print(df_tex_buff.shape)

    data_balance = dp.concat([df_tex_buff,df_data_buff],axis=1)
    print(data_balance.head(5))
    print(data_balance.shape)
    label_columns = data_balance.columns.to_list()
    print(label_columns[-13:])

    data_balance_same = data_balance.drop_duplicates(subset=label_columns[-13:-1],keep='first',inplace=False)
    print(data_balance_same.shape)

    data_balance_zero = data_balance_same.loc[~(data_balance_same[label_columns[-13:-1]] == 0).all(axis=1), :]
    print(data_balance_zero.shape)

    #data_balance_zero.to_csv("./inter_data/data_balance_del_zero.csv")
    label_count = get_label_count(data_balance_zero.copy())
    print(label_count)

    df_inited = init_df_feature_muti(13, data_balance_zero.copy())

    if df_inited.empty:
        print("Warinning: dataset is empty!")
    else:

        print(df_inited.head(5))
        print(df_inited.shape)

        model_linear = Model_linear(df_inited, 0.3)
        y_test, y_pred = model_linear.exec("LogisticRegression", 'no')

        model_plot = Model_plot()
        model_plot.plot_show(y_test, y_pred)

    # model_knn = Model_knn(df_inited,0.33)
    # y_test,y_pred = model_tree.exec("KNeighborsClassifier","no")

    # model_tree = Model_tree(df_inited,0.33)
    # y_test,y_pred = model_tree.exec("DecisionTreeClassifier","no")

    # model_tree = Model_tree(df_inited, 0.33)
    # y_test, y_pred = model_tree.exec("RandomForestClassifier", "no")

    # model_svm = Model_svm(df_inited,0.33)
    # y_test, y_pred = model_svm.exec("sklearn_SVC","no")
    # print(y_pred)
    # df_balance.dropna(axis=1, how='any', inplace=True)

















