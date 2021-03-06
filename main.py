
from model.gc_knn import Model_knn
from model.gc_tree import Model_tree
from model.gc_svm import Model_svm
from model.gc_linear import Model_linear
from plot import  Model_plot
from log import  Moddel_log
from model.gc_cluster import Model_cluster
from data_proces.gc_process import read_data_csv,init_df_type_code,get_label_count,trans_label_to_b,split_str,init_df_feature_muti
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import difflib



def make_lag_no_balance_inc(df_lag_no):
    lag_no_len = df_lag_no.shape[0]

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

    # df_del_zero =  df_lag_no.loc[(df_lag_no==0).all(axis=1),:]  #删除
    # print(df_del_zero.head(5))
    # print(df_del_zero.shape)

    # 去掉重复的有7万条数据，并且训练数据的得分并不高0.67
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    print(label_columns[-22:-1])
    print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    # df_del_zero = df_del_same.loc[df_del_same[label_columns[-22:-1]]]
    print(df_del_zero.head(5))
    print(df_del_zero.shape)

    # make_model(df_del_same)
    # 扩展纬度
    df_data = df_del_zero.iloc[:, -22:]
    df_data_buff = df_data.copy()
    print(df_data.head(5))
    print(df_data.shape)

    df_tex = df_del_zero.iloc[:, :-22]
    df_tex_buff = df_tex.copy()
    print(df_tex.head(5))
    print(df_tex.shape)

    for i in range(1, 10, 1):
        df_shift = df_data.shift(periods=i, axis=1)
        df_data_buff = pd.concat([df_data_buff, df_shift])
        df_tex_buff = pd.concat([df_tex_buff, df_tex])

    df_data_buff.dropna(axis=1, how='any', inplace=True)
    print(df_data_buff.head(5))
    print(df_data_buff.shape)
    df_data_buff = df_data_buff.astype("int64")
    print(df_data_buff.head(5))
    print(df_data_buff.shape)
    print(df_tex_buff.shape)

    data_balance = pd.concat([df_tex_buff, df_data_buff], axis=1)
    print(data_balance.head(5))
    print(data_balance.shape)
    label_columns = data_balance.columns.to_list()
    print(label_columns[-13:])

    data_balance_same = data_balance.drop_duplicates(subset=label_columns[-13:-1], keep='first', inplace=False)
    print(data_balance_same.shape)

    data_balance_zero = data_balance_same.loc[~(data_balance_same[label_columns[-13:-1]] == 0).all(axis=1), :]
    print(data_balance_zero.shape)

    # data_balance_zero.to_csv("./inter_data/data_balance_del_zero.csv")
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

def make_model(df_,param='no'):

    if df_.empty:
        print("Warinning: dataset is empty!")
        return 0

    columns_name = df_.columns.to_list()
    if columns_name[-1] != "label":
        df_.drop(df_.columns[-1], axis=1, inplace=True)
    if columns_name[0] == "Unnamed: 0":
        df_.drop(df_.columns[0], axis=1, inplace=True)

    label_ = columns_name[-1]
    label_count = df_.groupby([label_]).count().iloc[:, 0]
    print(label_count)

    # model build
    # model_tree = Model_tree(df_, 0.3)
    # y_test, y_pred = model_tree.exec("xgboost", param)

    # model_tree = Model_tree(df_, 0.3)
    # y_test, y_pred = model_tree.exec("RandomForestClassifier", param)

    # model_tree = Model_tree(df_, 0.3)
    # y_test, y_pred = model_tree.exec("AdaBoostClassifier", param)

    model_linear = Model_linear(df_,0.3)
    y_test, y_pred = model_linear.exec("LogisticRegression", param)


    # df_test = pd.DataFrame(ary_test, columns=new_columns)
    # df_test.to_csv('./data/test_result/test_feature_{}.csv'.format(feature),index=False)

    if param is 'no':
        model_plot = Model_plot()
        model_plot.plot_show(y_test, y_pred)


def split_ai_array(df_,array_col = -2):
    #split ai_array on many features
    AI_ORDER_COL_NUM = array_col
    df_str_data = df_.iloc[:, AI_ORDER_COL_NUM]
    df_str_tex = df_.iloc[:, :AI_ORDER_COL_NUM]
    # get feature in orderarray
    df_data = pd.DataFrame([split_str(x) for x in df_str_data], index=df_str_data.index).astype('int32')
    # get columns in df_data
    columns_count = df_data.shape[1]
    columns_name = []
    for col in range(columns_count):
        if col < columns_count - 1:
            name = 'feature_{}'.format(col + 1)
        else:
            name = 'label'
        columns_name.append(name)
    # rename columns
    df_data.columns = columns_name
    # concat df with str and data
    df_formated = pd.concat([df_str_tex, df_data], axis=1)
    columns_labels = df_formated.columns.to_list()
    print(df_formated.shape)
    # df_formated.to_csv("./data/df_format.csv",index=False)
    return df_formated

def drop_duplicates(df_,feature_col = 21):
    # drop duplicates by feature
    columns_name = df_.columns.to_list()
    df_sort = df_.sort_values(by=columns_name[-1], ascending=False)
    # drop duplicates by feature
    df_del_same = df_sort.drop_duplicates(subset=columns_name[-feature_col:-1], keep='first', inplace=False)
    # print(df_del_same.shape)
    # drop features are all zero
    df_del_zero = df_del_same.loc[~(df_del_same[columns_name[-feature_col:-1]] == 0).all(axis=1), :]
    # print(df_del_zero.shape)
    # print(df_del_zero.head(5))
    df_duplicates = df_del_zero.sort_index()
    # print(df_del_zero.head(5))
    df_duplicates.reset_index(drop=True, inplace=True)
    # print(df_sort_index.head(5))
    # print(df_del_zero.shape)
    #df_sort_index.to_csv("./data/df_format_sort.csv")
    return df_duplicates

def split_features(df_,feature_col = 0):
    #split feature with feature_col
    columns_name = df_.columns.to_list()
    if columns_name[-1] != 'label':
        print("data format is not in groupby label !")
        return 0
    feature_name_list = [name for name in columns_name if name.find('feature_') != -1]
    if feature_col == 0:
        feature_col = len(feature_name_list)

    df_tex = df_.iloc[:,:-22]
    df_label = df_.iloc[:,-1]
    df_data = df_.loc[:, feature_name_list[-feature_col:]]
    df_split_features = pd.concat([df_tex, df_data,df_label],axis=1)
    #df_split_features.to_csv("./inter_data/df_split_feature_{}.csv".format(feature_col),index=False)
    return df_split_features

def trans_label(df_):
    #trans label to 1 or 0
    df_temp = df_.copy()
    label_column = df_temp.columns.to_list()[-1]

    if label_column != "label":
        print("last columns is not label!")
        return 0

    df_label = df_temp.loc[:, [label_column]]
    df_label.loc[df_label[df_label.columns.to_list()[0]] > 0] = 1
    df_temp[label_column] = df_label.values

    return df_temp

def groupby_label(df_,feature_col = 0):
    #groupby label with sum and count
    columns_name = df_.columns.to_list()
    if columns_name[-1] != 'label':
        print("data format is not in groupby label !")
        return 0

    df_temp = df_.copy()
    feature_name_list = [name for name in columns_name if name.find('feature_') != -1]
    if feature_col == 0:
        feature_col = len(feature_name_list)

    df_agg = df_temp.groupby(feature_name_list[-feature_col:]).agg(label_sum=('label', 'sum'), label_count=("label", "count"))
    df_agg = df_agg.reset_index()
    return df_agg

def shfit_features(df_,feature_col = 0):
    #reduced length of array by sliding window in order to extend data records
    columns_name = df_.columns.to_list()
    feature_name_list = [name for name in columns_name if name.find('feature_') != -1]

    df_tex = df_.iloc[:,:-22]
    df_data = df_.iloc[:, -22:]

    n_shift = len(feature_name_list) - feature_col + 1
    df_data_buff = df_data.copy()
    df_tex_buff = df_tex.copy()

    for i in range(1, n_shift):
        df_shift = df_data.shift(i, axis=1)
        df_data_buff = pd.concat([df_data_buff, df_shift])
        df_tex_buff = pd.concat([df_tex_buff,df_tex])
    df_data_buff.dropna(axis=1, how='any', inplace=True)
    df_data_buff = df_data_buff.astype("int64")

    # df_tex = pd.DataFrame(df_tex.values.tolist() * n_shift, columns=df_tex.columns)
    df_shift = pd.concat([df_tex_buff, df_data_buff], axis=1)
    # df_shift.to_csv("./inter_data/df_shift_{}.csv".format(feature_col),index=False)
    return df_shift

def groupby_rate(df_,feature_col = 0):
    #groupby rate with label_num.label_count
    columns_name = df_.columns.to_list()
    if columns_name[-1] != 'label_count':
        print("data format is not in groupby rate!")
        return 0

    df_temp = df_.copy()
    feature_name_list = [name for name in columns_name if name.find('feature_') != -1]
    if feature_col == 0:
        feature_col = len(feature_name_list)

    df_group = df_temp.groupby(feature_name_list[-feature_col:])['label_sum', 'label_count'].sum()
    df_group = df_group.reset_index()
    df_group['rate'] = round(df_group['label_sum'] / df_group['label_count'], 2)
    df_group['label'] = 0
    df_group.loc[df_group['rate'] >= 0.4, 'label'] = 1
    # print(df_group.head(100))
    # print(df_group.shape)
    df_group.drop(['label_sum', 'label_count', 'rate'], axis=1, inplace=True)
    # df_group.to_csv("./inter_data/df_split_feature_{}.csv".format(13),index=False)
    return df_group

def cut_on_premise(df_):
    columns_name = df_.columns.to_list()
    on_premise_count = df_[columns_name[-1]].groupby(df_[columns_name[0]]).count()
    on_premise_index = on_premise_count.index.to_list()
    on_premise_dict = {}
    for x in on_premise_index:
        print(x)
        on_premise_dict[x] = pd.DataFrame()
        df_premise = df_.loc[df_[columns_name[0]] == x]
        on_premise_dict[x].append(df_premise)
    return on_premise_dict,on_premise_index

def cut_type_code(df_):
    columns_name = df_.columns.to_list()
    # df_.drop(columns_name[2:4], axis=1, inplace=True)
    type_code_count = df_[columns_name[-1]].groupby(df_[columns_name[1]]).count()
    type_code_index = type_code_count.index.to_list()
    df_other = df_.loc[df_[columns_name[1]] == 'Other']
    df_no_other = df_.loc[df_[columns_name[1]] != 'Other']

    df_other.to_csv('./inter_data/df_other_1.csv', index=False)
    # df_no_other.to_csv('./inter_data/df_no_other.csv', index=False)

def smooth_verify(df_no_other):
    # df_shift_feature = shfit_features(df_other,15)
    df_shift_feature = shfit_features(df_no_other, 15)
    df_temp = trans_label(df_shift_feature)
    df_group = groupby_label(df_temp)
    # # print(df_group.head(5))
    df_rate = groupby_rate(df_group)
    # make_model(df_rate, 'no')

    columns_name = df_rate.columns.to_list()
    label_ = columns_name[-1]
    label_count = df_rate.groupby([label_]).count().iloc[:, 0]
    print(label_count)

    values = df_rate.values
    values_x, values_y = values[:, :-1], values[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(values_x, values_y, test_size=0.3, random_state=42)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    print('交叉验证')
    y_pred1 = lr.predict(train_x)
    print("ACC Score (Test): %f" % metrics.accuracy_score(train_y, y_pred1))
    model_plot = Model_plot()
    model_plot.plot_show(train_y, y_pred1)

def split_verify(df_no_other):

    # df_split_feature = split_features(df_other,15)
    df_split_feature = split_features(df_no_other, 15)
    df_temp = trans_label(df_split_feature)
    df_group = groupby_label(df_temp)
    # print(df_group.head(5))
    df_rate = groupby_rate(df_group)
    # make_model(df_rate, 'no')

    columns_name = df_rate.columns.to_list()
    label_ = columns_name[-1]
    label_count = df_rate.groupby([label_]).count().iloc[:, 0]
    print(label_count)

    values = df_rate.values
    values_x, values_y = values[:, :-1], values[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(values_x, values_y, test_size=0.3, random_state=42)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    print('交叉验证')
    y_pred1 = lr.predict(test_x)
    print("ACC Score (Test): %f" % metrics.accuracy_score(test_y, y_pred1))
    model_plot = Model_plot()
    model_plot.plot_show(test_y, y_pred1)

def model_load():

    # model_no_other_shift_15_path = './inter_data/lr0_no_other_shift_15.model'
    # lr = joblib.load(model_no_other_shift_15_path)

    # model_no_other_split_15_path = './inter_data/lr0_no_other_split_15.model'
    # lr = joblib.load(model_no_other_split_15_path)

    # model_other_split_15_path = './inter_data/lr0_other_shift_15.model'
    # lr = joblib.load(model_other_split_15_path)
    pass

# def shift_features(df, feature_num=13):
#     # reduced length of array by sliding window in order to extend data records
#     df_attr = df.iloc[:, :2]
#     df_data = df.iloc[:, 2:-4]
#     df_label = df.iloc[:, -4:-2]
#
#     N_ALL = 22
#     n_shift = N_ALL - feature_num
#     df_data_temp = df_data.copy()
#     for i in range(1, n_shift):
#         df_shift = df_data_temp.shift(i, axis=1)
#         df_data = pd.concat([df_data, df_shift])
#     df_data.dropna(axis=1, how='any', inplace=True)
#
#     df_attr = pd.DataFrame(df_attr.values.tolist() * n_shift, columns=df_attr.columns)
#     df_label = pd.DataFrame(df_label.values.tolist() * n_shift, columns=df_label.columns)
#
#     df_smooth = pd.concat([df_attr, df_data, df_label], axis=1)
#
#     return df_smooth

def label_encode(df_):
    #endcoe by LabelEncoder in df_
    str_col = df_.columns.to_list()
    df_encode = df_.copy()
    encoder_dict = {}
    for col in str_col:
        encoder_dict[col] = LabelEncoder()
        integer_encoded = encoder_dict[col].fit_transform(df_encode[col].values)
        df_encode[col] = pd.DataFrame(integer_encoded).values.astype('int64')

    return df_encode,encoder_dict

def label_decode(label_encoder, df_, col_name=''):
    inverted = label_encoder.inverse_transform(df_[col_name].values)
    df_[col_name].values = inverted
    return 0



if __name__ == "__main__":

    #重定向日志
    sys.stdout = Moddel_log("./log/run.log", sys.stdout)
    # orign data format
    # path_ori = "../3_data/UnitedDist_AIOrderAnalyze.csv"

    # df_dir = "/Users/longgle/Documents/0_work/0_projects/ai/ai_order/2_data/"
    # df_file_name = "df_format.csv"
    # df_ = read_data_csv(df_dir + df_file_name)
    #
    # print(df_.shape)
    # cut_type_code(df_)

    df_other_path = "./inter_data/df_other_1.csv"
    df_other = read_data_csv(df_other_path)

    # df_no_other_path = "./inter_data/df_no_other.csv"
    # df_no_other = read_data_csv(df_no_other_path)

    #
    columns_name = df_other.columns.to_list()
    # # print(columns_name)
    # if columns_name[-1] != 'label':
    #     print("data format is not in groupby label !")
    #
    # feature_name_list = [name for name in columns_name if name.find('feature_') != -1]
    #
    # df_temp = df_.groupby(columns_name[:-1]).agg(label_sum=('label', 'sum'), label_count=("label", "count"))
    # df_temp = df_temp.reset_index()
    # print(df_temp.head(5))
    # print(df_temp.shape)
    # df_temp.sort_values(by='label_count',ascending=False)
    # print(df_temp.head(100))


    df_str_feature = df_other.loc[:,columns_name[0:4]]
    df_label = df_other.iloc[:,-22:]

    # df_feature = df_.loc[:,columns_name[5:]]

    str_feature_name = df_str_feature.columns.to_list()
    print(str_feature_name)
    # print(str_feature_name)
    # df_encode, *_ = label_encode(df_str_feature)
    #
    # df_temp = pd.concat([df_encode, df_feature], axis=1)
    # df_init = trans_label(df_temp)

    # On Premise
    print(df_str_feature['On Premise'].value_counts())
    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(df_str_feature[str_feature_name[0]].values)
    df_str_feature[str_feature_name[0]] = pd.DataFrame(integer_encoded).values.astype('int64')
    # print(df_str_feature.head(5))

    # Type Code
    # df_type_code = df_str_feature.iloc[:,1]
    # print(df_type_code.head(5))
    print(df_str_feature['Type Code'].value_counts())
    df_str_feature.drop(['Type Code'], axis=1, inplace=True)

    # df_str_feature['Type Code'] = df_str_feature['Type Code'].map(lambda x: 1 if x == "Other" else 0).astype('int64')
    # print(df_str_feature.head(5))

    # print(df_str_feature['Type Code'].value_counts())

    # type_dict = {
    #     "Wine": "Wine",
    #     "Repeal Beer": "Beer",
    #     "Liquor": "Liquor",
    #     "Soda/Water": "Soda/Water",
    #     "Other": "Other",
    #     "NA Beer": "Beer",
    #     "3.2% Beer": "Beer"
    # }
    # df_str_feature['Type Code'] = df_str_feature['Type Code'].map(type_dict)
    # print(df_str_feature['Type Code'].value_counts())

    # df_type = pd.get_dummies(df_str_feature['Type Code'])
    # df_str_feature = pd.concat([df_str_feature,df_type],axis=1)
    # df_str_feature.drop('Type Code',axis=1,inplace=True)

    # Customer Num Company
    print(df_str_feature['Customer Num Company'].value_counts())

    df_str_feature['Market'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('MARKET') != -1 else 0)
    df_str_feature['Shop'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('SHOP') != -1 else 0)
    df_str_feature['Store'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('STORE') != -1 else 0)
    df_str_feature['Mart'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('MART') != -1 else 0)
    df_str_feature['publix'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('PUBLIX') != -1 else 0)
    df_str_feature['outlet'] = df_str_feature['Customer Num Company'].map(lambda s: 1 if s.find('OUTLET') != -1 else 0)

    print(df_str_feature['Market'].value_counts())
    print(df_str_feature['Shop'].value_counts())
    print(df_str_feature['Store'].value_counts())
    print(df_str_feature['Mart'].value_counts())
    print(df_str_feature['publix'].value_counts())
    print(df_str_feature['outlet'].value_counts())

    df_str_feature.drop(['Customer Num Company','Product Num Name'], axis=1, inplace=True)
    print(df_str_feature.head(5))

    df_feature = pd.concat([df_str_feature,df_label],axis=1)
    # df_temp = trans_label(df_feature)
    # print(df_feature.shape)

    # columns_name = df_.columns.to_list()
    # feature_name_list = [name for name in columns_name if name.find('feature_') != -1]

    df_tex = df_feature.iloc[:, :-22]
    df_data = df_feature.iloc[:, -22:]

    n_shift = 21 - 15 + 1
    df_data_buff = df_data.copy()
    df_tex_buff = df_tex.copy()

    for i in range(1, n_shift):
        df_shift = df_data.shift(i, axis=1)
        df_data_buff = pd.concat([df_data_buff, df_shift])
        df_tex_buff = pd.concat([df_tex_buff, df_tex])
    df_data_buff.dropna(axis=1, how='any', inplace=True)
    df_data_buff = df_data_buff.astype("int64")

    # df_tex = pd.DataFrame(df_tex.values.tolist() * n_shift, columns=df_tex.columns)
    df_shift = pd.concat([df_tex_buff, df_data_buff], axis=1)
    df_temp = trans_label(df_shift)

    columns = df_temp.columns.to_list()
    df_agg = df_temp.groupby(columns[:-1]).agg(label_sum=('label', 'sum'),label_count=("label", "count"))
    df_agg = df_agg.reset_index()

    df_group = df_agg.groupby(columns[:-1])['label_sum', 'label_count'].sum()
    df_group = df_group.reset_index()
    df_group['rate'] = round(df_group['label_sum'] / df_group['label_count'], 2)
    df_group['label'] = 0
    df_group.loc[df_group['rate'] >= 0.4, 'label'] = 1
    # print(df_group.head(100))
    # print(df_group.shape)
    df_group.drop(['label_sum', 'label_count', 'rate'], axis=1, inplace=True)

    print(df_group.head(5))
    print(df_group.shape)

    corrDf = df_group.corr()
    # print(corrDf)
    print(corrDf['label'].sort_values(ascending=False))


    label_ = df_group.columns.to_list()[1]
    # print(label_)
    label_count = df_group.groupby([label_]).count().iloc[:, 0]
    # print(label_count)
    label_max = max(label_count.values)
    label_min = min(label_count.values)

    df_label_max = df_group[df_group[label_] == label_count.idxmax()]
    # print(df_label_max.head(5))
    df_label_min = df_group[df_group[label_] == label_count.idxmin()]
    # print(df_label_min.head(5))
    df_label_max.reset_index(drop=True, inplace=True)

    drop_indices = np.random.choice(df_label_max.index, label_max - label_min,replace=False)
    df_drop = df_label_max.drop(drop_indices)
    df_balance = pd.concat([df_label_min, df_drop])
    df_balance.reset_index(drop=True, inplace=True)

    make_model(df_balance, 'no')















    # df_str_feature['label'] = 1
    # df_agg = df_str_feature.groupby(str_feature_name[2])['label'].count()
    # df_agg = df_agg.reset_index()
    # df_agg_q = df_agg.sort_values(by='label',ascending=False)
    # # df_agg_q.to_csv('./inter_data/agg.csv',index=False)
    # print(df_agg_q.head(5))








    # print(df_temp.head(5))
    # make_model(df_init, 'no')



    # print(df_str_feature.head(5)

    # print(integer_encoded)























































