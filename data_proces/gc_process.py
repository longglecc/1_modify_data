
import pandas as pd
import numpy as np

__all__ = ['read_data_csv','init_df_type_code','get_label_count','trans_label_to_b']

def read_data_csv(file_path):

    df = pd.read_csv(file_path)
    #df = df[df.columns.tolist()[-1]].astype(str)
    return df

def save_data_csv(data,save_path):

    data.to_csv(save_path)
    print("data is saved :"+ save_path)


def mkdir_file():
    pass
    #TODO(gaolongcc):

def split_str(str):
    """
    :param str:字符串
    :return:拆分后的字符串
    """
    str_list=[]
    for i in range(len(str)):
        str_list.append(int(str[i]))
    return str_list

def init_df_feature(n,df):
    """
    :param n:特征个数
    :param df:数据集合
    :return:处理好的数据集合
    """
    df_inited = pd.DataFrame([split_str(n,x) for x in df], index=df.index).astype('int32')
    label_columns = df_inited.columns.to_list()[n-1]
    #print(label_columns)
    df_label = df_inited.loc[:,[label_columns]]
    #print(df_label.head(5))
    df_label.loc[df_label[df_label.columns.to_list()[0]]>0]=1
    #print(df_label.head(5))
    df_inited[label_columns] = df_label.values
    #print(df_inited)
    return df_inited

def init_df_feature_muti(n,df):
    pass


def init_df_cluster(df):
    df.drop(df.columns[-1], axis=1, inplace=True)
    df_feature = df.iloc[:, -22:]
    return df_feature

def init_df_on_premise(df,param="on"):

    label_columns = df.columns
    if label_columns[-1] != "label":

        df.drop(df.columns[-1], axis=1, inplace=True)
        df.drop(df.columns[0],axis=1,inplace=True)
    label_columns = df.columns

    if param is "on":
        df_on = df.loc[df[label_columns[0]]=="On Premise"]#df_label.loc[df_label[df_label.columns.to_list()[0]] > 0]
        print(df_on.head(5))
        print(df_on.shape)
        df_on = df_on.iloc[:, -22:]
        return df_on

    elif param is "off":

        df_off = df.loc[df[label_columns[0]]=="Off Premise"]
        print(df_off.head(5))
        print(df_off.shape)
        df_off = df_off.iloc[:, -22:]
        return df_off

    else:
        df_new = df.copy().iloc[:, -22:]
        return df_new


def init_df_type_code(df):

    label_columns = df.columns
    if label_columns[-1] != "label":
        df.drop(df.columns[-1], axis=1, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)

    label_columns = df.columns
    type_code_dict = {}
    type_code_count = df[label_columns].groupby(df[label_columns[1]]).count()
    type_code_index = type_code_count.index.to_list()
    for x in type_code_index[:]:
        #print(x)
        type_code_dict[x] = pd.DataFrame()#columns = label_columns[-22:]
        type_code_df = df.loc[df[label_columns[1]]==x]
        type_code_df = type_code_df.iloc[:, -22:]
        if len(type_code_df.values)>100:
            type_code_dict[x] = type_code_dict[x].append(type_code_df, ignore_index=True)
        else:
            type_code_dict.pop(x)
            type_code_index.remove(x)
            #print(type_code_df.head(5))
            #print(type_code_df.shape)

    return type_code_dict,type_code_index

def get_label_count(df):

    if df.empty:
        print("Warinning: dataset is empty!")
        return 0
    label = df.columns[-1]
    print(label)

    label_count = df[df.columns[0]].groupby(df[label]).count()
    print(label_count)

def trans_label_to_b(df):
    label_columns = df.columns.to_list()[-1]
    if label_columns == "label":
        print(label_columns)
        df_label = df.loc[:, [label_columns]]
        # print(df_label.head(5))
        df_label.loc[df_label[df_label.columns.to_list()[0]] > 0] = 1
        # print(df_label.head(5))
        df[label_columns] = df_label.values
        # print(df.head(5))
        # print(df.shape)
    else:
        print("last columns is not label!")

    return df

def make_df_smooth_balance(df_lag_no):

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    print(label_columns)

    '''
    去重复和去零
    '''
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    # print(label_columns[-22:-1])
    # print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    # print(df_del_zero.head(5))
    print(df_del_zero.shape)

    on_premise_count = df_del_zero[label_columns[-1]].groupby(df_del_zero[label_columns[0]]).count()
    on_premise_index = on_premise_count.index.to_list()

    '''
    数据扩展：
    原理：利用滑窗，减少特征数量，增大数据条数
    '''
    df_data = df_del_zero.iloc[:, -22:]
    df_data_buff = df_data.copy()
    print(df_data.shape)

    df_tex = df_del_zero.iloc[:, :-22]
    df_tex_buff = df_tex.copy()
    print(df_tex.shape)

    for i in range(1, 10, 1):
        df_shift = df_data.shift(periods=i, axis=1)
        df_data_buff = pd.concat([df_data_buff, df_shift])
        df_tex_buff = pd.concat([df_tex_buff, df_tex])

    df_data_buff.dropna(axis=1, how='any', inplace=True)
    # print(df_data_buff.head(5))
    print(df_data_buff.shape)
    df_data_buff = df_data_buff.astype("int64")
    # print(df_data_buff.head(5))
    print(df_data_buff.shape)
    print(df_tex_buff.shape)

    data_smooth = pd.concat([df_tex_buff, df_data_buff], axis=1)
    print(data_smooth.head(5))
    print(data_smooth.shape)
    label_columns = data_smooth.columns.to_list()
    print(label_columns[-13:])

    '''
    去重复和去零
    '''
    data_smooth_same = data_smooth.drop_duplicates(subset=label_columns[-13:-1], keep='first', inplace=False)
    print(data_smooth_same.shape)

    data_smooth_zero = data_smooth_same.loc[~(data_smooth_same[label_columns[-13:-1]] == 0).all(axis=1), :]
    print(data_smooth_zero.shape)

    '''
    将标签转换为0/1
    '''
    df_trans_label = trans_label_to_b(data_smooth_zero.copy())

    '''
    统计标签数量
    '''
    label_ = df_trans_label.columns.to_list()[-1]
    # print(label_)
    label_count = df_trans_label.groupby([label_]).count().iloc[:, 0]
    print(label_count)
    label_max = max(label_count.values)
    label_min = min(label_count.values)
    print(label_max, label_min)

    label_count_0 = label_count[label_count.keys() == 0].values[0]
    label_count_1 = label_count[label_count.keys() == 1].values[0]

    BALANCE_P = .8
    if label_max / label_min < BALANCE_P:
        print("sample is balance")
        pass

    '''
    获取标签数据集
    '''
    df_label_max = df_trans_label[df_trans_label[label_] == label_count.idxmax()]
    print(df_label_max.head(5))
    df_label_min = df_trans_label[df_trans_label[label_] == label_count.idxmin()]
    print(df_label_min.head(5))
    df_label_max.reset_index(drop=True, inplace=True)

    # df_label_1 = df_trans_label[df_trans_label[label_] == 1]
    # df_label_0 = df_trans_label[df_trans_label[label_] == 0]
    # drop_indices = np.random.choice(df_label_0.index, label_count_0 - int(label_count_1 * .8), replace=False)
    # df_drop = df_label_0.drop(drop_indices)
    if label_count.idxmax() == 1:
        print(label_min / label_max)
        if label_min / label_max > BALANCE_P:
            drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * 1.), replace=False)
        else:
            drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min / BALANCE_P), replace=False)
    else:
        drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * BALANCE_P), replace=False)

    df_drop = df_label_max.drop(drop_indices)

    print(df_drop.head(5))
    print(df_drop.shape)

    '''
    获取随机平衡数据集
    '''
    df_balance = pd.concat([df_label_min, df_drop])
    df_balance.reset_index(drop=True, inplace=True)
    #
    # # df_balance.to_csv("./inter_data/Nother_on_rand_balance.csv")
    #
    # df_balance = df_balance.iloc[:, -22:]
    # get_label_count(df_balance)

    # df_inited = init_df_feature_muti(22, df_premise.copy())
    # if df_inited.empty:
    #     print("Warinning: dataset is empty!")
    # else:

    print(df_balance.head(5))
    print(df_balance.shape)
    df_balance = df_balance.iloc[:, -13:]

    label_ = df_balance.columns.to_list()[-1]
    label_count = df_balance.groupby([label_]).count().iloc[:, 0]
    print(label_count)

    return df_balance

def make_df_on_smooth_balance(df_lag_no):

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    print(label_columns)

    '''
    去重复和去零
    '''
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    # print(label_columns[-22:-1])
    # print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    # print(df_del_zero.head(5))
    print(df_del_zero.shape)

    on_premise_count = df_del_zero[label_columns[-1]].groupby(df_del_zero[label_columns[0]]).count()
    on_premise_index = on_premise_count.index.to_list()

    for x in on_premise_index:
        print("---------start---------")
        print(x)
        df_premise = df_del_zero.loc[
            df_del_zero[label_columns[0]] == x]  # df_label.loc[df_label[df_label.columns.to_list()[0]] > 0]
        print(df_premise.head(5))
        print(df_premise.shape)

        '''
        数据扩展：
        原理：利用滑窗，减少特征数量，增大数据条数
        '''
        df_data = df_premise.iloc[:, -22:]
        df_data_buff = df_data.copy()
        print(df_data.shape)

        df_tex = df_premise.iloc[:, :-22]
        df_tex_buff = df_tex.copy()
        print(df_tex.shape)

        for i in range(1, 10, 1):
            df_shift = df_data.shift(periods=i, axis=1)
            df_data_buff = pd.concat([df_data_buff, df_shift])
            df_tex_buff = pd.concat([df_tex_buff, df_tex])

        df_data_buff.dropna(axis=1, how='any', inplace=True)
        # print(df_data_buff.head(5))
        print(df_data_buff.shape)
        df_data_buff = df_data_buff.astype("int64")
        # print(df_data_buff.head(5))
        print(df_data_buff.shape)
        print(df_tex_buff.shape)

        data_smooth = pd.concat([df_tex_buff, df_data_buff], axis=1)
        print(data_smooth.head(5))
        print(data_smooth.shape)
        label_columns = data_smooth.columns.to_list()
        print(label_columns[-13:])

        '''
        去重复和去零
        '''
        data_smooth_same = data_smooth.drop_duplicates(subset=label_columns[-13:-1], keep='first', inplace=False)
        print(data_smooth_same.shape)

        data_smooth_zero = data_smooth_same.loc[~(data_smooth_same[label_columns[-13:-1]] == 0).all(axis=1), :]
        print(data_smooth_zero.shape)

        '''
        将标签转换为0/1
        '''
        df_trans_label = trans_label_to_b(data_smooth_zero.copy())

        '''
        统计标签数量
        '''
        label_ = df_trans_label.columns.to_list()[-1]
        # print(label_)
        label_count = df_trans_label.groupby([label_]).count().iloc[:, 0]
        print(label_count)
        label_max = max(label_count.values)
        label_min = min(label_count.values)
        print(label_max, label_min)

        # label_count_0 = label_count[label_count.keys() == 0].values[0]
        # label_count_1 = label_count[label_count.keys() == 1].values[0]
        #
        # if label_count_0 / label_count_1 < .8:
        #     print("sample is balance")
        #     continue
        BALANCE_P = .8
        if label_max / label_min < BALANCE_P:
            print("sample is balance")
            continue

        '''
        获取标签数据集
        '''
        df_label_max = df_trans_label[df_trans_label[label_] == label_count.idxmax()]
        print(df_label_max.head(5))
        df_label_min = df_trans_label[df_trans_label[label_] == label_count.idxmin()]
        print(df_label_min.head(5))
        df_label_max.reset_index(drop=True, inplace=True)

        # df_label_1 = df_trans_label[df_trans_label[label_] == 1]
        # df_label_0 = df_trans_label[df_trans_label[label_] == 0]
        # drop_indices = np.random.choice(df_label_0.index, label_count_0 - int(label_count_1 * .8), replace=False)
        # df_drop = df_label_0.drop(drop_indices)
        if label_count.idxmax() == 1:
            print(label_min / label_max)
            if label_min / label_max > BALANCE_P:
                drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * 1.), replace=False)
            else:
                drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min / BALANCE_P),
                                                replace=False)
        else:
            drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * BALANCE_P), replace=False)

        df_drop = df_label_max.drop(drop_indices)

        print(df_drop.head(5))
        print(df_drop.shape)

        '''
        获取随机平衡数据集
        '''
        df_balance = pd.concat([df_label_min, df_drop])
        df_balance.reset_index(drop=True, inplace=True)
        #
        # # df_balance.to_csv("./inter_data/Nother_on_rand_balance.csv")
        #
        # df_balance = df_balance.iloc[:, -22:]
        # get_label_count(df_balance)

        # df_inited = init_df_feature_muti(22, df_premise.copy())
        # if df_inited.empty:
        #     print("Warinning: dataset is empty!")
        # else:

        print(df_balance.head(5))
        print(df_balance.shape)
        df_balance = df_balance.iloc[:, -13:]

        label_ = df_balance.columns.to_list()[-1]
        label_count = df_balance.groupby([label_]).count().iloc[:, 0]
        print(label_count)

        return df_balance

def make_df_balance(df_lag_no):

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    print(label_columns)

    '''
    去重复和去零
    '''
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    # print(label_columns[-22:-1])
    print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    print(df_del_zero.head(5))
    print(df_del_zero.shape)

    # 将标签转换为0，1
    df_trans_label = trans_label_to_b(df_del_zero.copy())

    '''
    统计标签数量
    '''
    label_ = df_trans_label.columns.to_list()[-1]
    if label_ != "label":
        print("last column is not label.")
        pass
    # print(label_)
    label_count = df_trans_label.groupby([label_]).count().iloc[:, 0]
    print(label_count)
    # label_max = max(label_count.values)
    # label_min = min(label_count.values)
    # print(label_max, label_min)

    label_count_0 = label_count[label_count.keys() == 0].values[0]
    label_count_1 = label_count[label_count.keys() == 1].values[0]

    if label_count_0 / label_count_1 < .8:
        print("sample is balance")
        pass

    '''
    获取标签数据集
    '''
    df_label_1 = df_trans_label[df_trans_label[label_] == 1]
    df_label_0 = df_trans_label[df_trans_label[label_] == 0]
    drop_indices = np.random.choice(df_label_0.index, label_count_0 - int(label_count_1 * .8), replace=False)
    df_drop = df_label_0.drop(drop_indices)
    print(df_drop.head(5))
    print(df_drop.shape)
    '''
    获取随机平衡数据集
    '''
    df_balance = pd.concat([df_label_1, df_drop])
    df_balance.reset_index(drop=True, inplace=True)

    # df_balance.to_csv("./inter_data/Nother_on_rand_balance.csv")

    df_balance = df_balance.iloc[:, -22:]
    get_label_count(df_balance)

    # df_inited = init_df_feature_muti(22, df_premise.copy())
    # if df_inited.empty:
    #     print("Warinning: dataset is empty!")
    # else:

    print(df_balance.head(5))
    print(df_balance.shape)

    # df_balance = df_trans_label.iloc[:, -22:]
    return df_balance

def make_df_on_balance(df_lag_no):
    """
    1、将df_lag_no数据集按照On_premise字段进行数据分组，然后分别进行数据均衡处理
    2、均衡处理的方式为：减少标签对应数据集较多的那组，随机删除一部分数据集，使得标签数据比例在0.8:1左右
    :param df_lag_no:
    :return:
    """

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    print(label_columns)

    '''
    去重复和去零
    '''
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-22:-1], keep='first', inplace=False)
    # print(label_columns[-22:-1])
    print(df_del_same.head(5))
    print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-22:-1]] == 0).all(axis=1), :]
    print(df_del_zero.head(5))
    print(df_del_zero.shape)

    on_premise_count = df_del_zero[label_columns[-1]].groupby(df_del_zero[label_columns[0]]).count()
    on_premise_index = on_premise_count.index.to_list()

    for x in on_premise_index:
        print("---------start---------")
        print(x)
        df_premise = df_del_zero.loc[df_del_zero[label_columns[0]] == x]  # df_label.loc[df_label[df_label.columns.to_list()[0]] > 0]
        print(df_premise.head(5))
        print(df_premise.shape)

        # 将标签转换为0，1
        df_trans_label = trans_label_to_b(df_premise.copy())

        '''
        统计标签数量
        '''
        label_ = df_trans_label.columns.to_list()[-1]
        # print(label_)
        label_count = df_trans_label.groupby([label_]).count().iloc[:, 0]
        print(label_count)
        # label_max = max(label_count.values)
        # label_min = min(label_count.values)
        # print(label_max, label_min)

        label_count_0 = label_count[label_count.keys() == 0].values[0]
        label_count_1 = label_count[label_count.keys() == 1].values[0]

        if label_count_0 / label_count_1 < .8:
            print("sample is balance")
            continue
        '''
        获取标签数据集
        '''
        # df_label_max = df_trans_label[df_trans_label[label_] == label_count.idxmax()]
        # print(df_label_max.head(5))
        # df_label_min = df_trans_label[df_trans_label[label_] == label_count.idxmin()]
        # print(df_label_min.head(5))
        # df_label_max.reset_index(drop=True, inplace=True)

        df_label_1 = df_trans_label[df_trans_label[label_] == 1]
        df_label_0 = df_trans_label[df_trans_label[label_] == 0]
        drop_indices = np.random.choice(df_label_0.index, label_count_0 - int(label_count_1 * .8), replace=False)
        df_drop = df_label_0.drop(drop_indices)
        print(df_drop.head(5))
        print(df_drop.shape)
        '''
        获取随机平衡数据集
        '''
        df_balance = pd.concat([df_label_1, df_drop])
        df_balance.reset_index(drop=True, inplace=True)

        # df_balance.to_csv("./inter_data/Nother_on_rand_balance.csv")

        df_balance = df_balance.iloc[:, -22:]
        get_label_count(df_balance)

        # df_inited = init_df_feature_muti(22, df_premise.copy())
        # if df_inited.empty:
        #     print("Warinning: dataset is empty!")
        # else:

        print(df_balance.head(5))
        print(df_balance.shape)

        return df_balance

def make_df_diff_feature(df_lag_no):

    FEATURE_MAX = 22
    FEATURE_MIN = 7

    label_columns = df_lag_no.columns.to_list()
    if label_columns[-1] != "label":
        df_lag_no.drop(label_columns[-1], axis=1, inplace=True)
    if label_columns[0] == "Unnamed: 0":
        df_lag_no.drop(label_columns[0], axis=1, inplace=True)

    label_columns = df_lag_no.columns.to_list()
    # print(label_columns)

    '''
    去重复和去零
    '''
    df_del_same = df_lag_no.drop_duplicates(subset=label_columns[-FEATURE_MAX:-1], keep='first', inplace=False)
    # print(label_columns[-22:-1])
    # print(df_del_same.head(5))
    # print(df_del_same.shape)

    df_del_zero = df_del_same.loc[~(df_del_same[label_columns[-FEATURE_MAX:-1]] == 0).all(axis=1), :]
    # print(df_del_zero.head(5))
    # print(df_del_zero.shape)

    on_premise_count = df_del_zero[label_columns[-1]].groupby(df_del_zero[label_columns[0]]).count()
    on_premise_index = on_premise_count.index.to_list()

    for x in on_premise_index:
        print("---------start---------")
        print(x)
        df_premise = df_del_zero.loc[
            df_del_zero[label_columns[0]] == x]  # df_label.loc[df_label[df_label.columns.to_list()[0]] > 0]
        # print(df_premise.head(5))
        # print(df_premise.shape)

        '''
        数据扩展：
        原理：利用滑窗，减少特征数量，增大数据条数
        '''
        df_data = df_premise.iloc[:, -FEATURE_MAX:]
        # print(df_data.shape)
        df_tex = df_premise.iloc[:, :-FEATURE_MAX]
        # print(df_tex.shape)

        for column in range(FEATURE_MIN, FEATURE_MAX + 1, 1):
            print("current feature is {}".format(column - 1))
            df_tex_buff = df_tex.copy()
            df_data_buff = df_data.copy()

            for i in range(1, FEATURE_MAX - column + 1, 1):
                df_shift = df_data.shift(periods=i, axis=1)
                df_data_buff = pd.concat([df_data_buff, df_shift])
                df_tex_buff = pd.concat([df_tex_buff, df_tex])

            df_data_buff.dropna(axis=1, how='any', inplace=True)
            # print(df_data_buff.head(5))
            # print(df_data_buff.shape)
            df_data_buff = df_data_buff.astype("int64")
            # print(df_data_buff.head(5))
            # print(df_data_buff.shape)
            # print(df_tex_buff.shape)

            data_smooth = pd.concat([df_tex_buff, df_data_buff], axis=1)
            # print(data_smooth.head(5))
            # print(data_smooth.shape)

            label_columns = data_smooth.columns.to_list()
            # print(label_columns[-column:])

            '''
            去重复和去零
            '''
            data_smooth_same = data_smooth.drop_duplicates(subset=label_columns[-column:-1], keep='first',
                                                           inplace=False)
            # print(data_smooth_same.shape)

            data_smooth_zero = data_smooth_same.loc[~(data_smooth_same[label_columns[-column:-1]] == 0).all(axis=1), :]
            # print(data_smooth_zero.shape)

            '''
            将标签转换为0/1
            '''
            df_trans_label = trans_label_to_b(data_smooth_zero.copy())

            '''
            统计标签数量
            '''
            label_ = df_trans_label.columns.to_list()[-1]
            # print(label_)
            label_count = df_trans_label.groupby([label_]).count().iloc[:, 0]
            print(label_count)
            label_max = max(label_count.values)
            label_min = min(label_count.values)
            # print(label_max, label_min)

            # label_count_0 = label_count[label_count.keys() == 0].values[0]
            # label_count_1 = label_count[label_count.keys() == 1].values[0]
            #
            # if label_count_0 / label_count_1 < .8:
            #     print("sample is balance")
            #     continue
            BALANCE_P = .8
            if label_max / label_min < BALANCE_P:
                print("sample is balance")
                continue

            '''
            获取标签数据集
            '''
            df_label_max = df_trans_label[df_trans_label[label_] == label_count.idxmax()]
            # print(df_label_max.head(5))
            df_label_min = df_trans_label[df_trans_label[label_] == label_count.idxmin()]
            # print(df_label_min.head(5))
            df_label_max.reset_index(drop=True, inplace=True)

            # df_label_1 = df_trans_label[df_trans_label[label_] == 1]
            # df_label_0 = df_trans_label[df_trans_label[label_] == 0]
            # drop_indices = np.random.choice(df_label_0.index, label_count_0 - int(label_count_1 * .8), replace=False)
            # df_drop = df_label_0.drop(drop_indices)
            if label_count.idxmax() == 1:
                print(label_min / label_max)
                if label_min / label_max > BALANCE_P:
                    drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * 1.), replace=False)
                else:
                    drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min / BALANCE_P),
                                                    replace=False)
            else:
                drop_indices = np.random.choice(df_label_max.index, label_max - int(label_min * BALANCE_P),
                                                replace=False)

            df_drop = df_label_max.drop(drop_indices)

            # print(df_drop.head(5))
            # print(df_drop.shape)

            '''
            获取随机平衡数据集
            '''
            df_balance = pd.concat([df_label_min, df_drop])
            df_balance.reset_index(drop=True, inplace=True)
            #
            # # df_balance.to_csv("./inter_data/Nother_on_rand_balance.csv")
            #
            # df_balance = df_balance.iloc[:, -22:]
            # get_label_count(df_balance)

            # df_inited = init_df_feature_muti(22, df_premise.copy())
            # if df_inited.empty:
            #     print("Warinning: dataset is empty!")
            # else:

            # print(df_balance.head(5))
            # print(df_balance.shape)
            df_balance = df_balance.iloc[:, -column:]

            label_ = df_balance.columns.to_list()[-1]
            label_count = df_balance.groupby([label_]).count().iloc[:, 0]
            # print(label_count)




