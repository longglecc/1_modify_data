
import pandas as pd
import numpy as np

__all__ = ['read_data_csv', 'init_df_feature_muti','init_df_cluster']

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

def split_str(n,str):
    """
    :param str:字符串
    :return:拆分后的字符串
    """
    str_list=[]
    if n > len(str):
        print("Warnning:n is too large than str len!")
        n = len(str)
    if n < 0:
        print("Warnning:n is too short than less 0!")
        n = 0

    for i in range(n,len(str)):
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

def init_df_cluster(df):
    df.drop(df.columns[-1], axis=1, inplace=True)
    df_feature = df.iloc[:, -22:]
    return df_feature


def init_df_feature_muti(n,df):
    """
    :param n:特征个数
    :param df:数据集合
    :return:处理好的数据集合
    """

    df.drop(df.columns[-1],axis = 1, inplace = True)
    df_feature = df.iloc[:,-22:]
    #print(df_feature)
    label_columns = df_feature.columns.to_list()[-1]
    print(label_columns)
    df_label = df_feature.loc[:, [label_columns]]
    #print(df_label.head(5))
    df_label.loc[df_label[df_label.columns.to_list()[0]] > 0] = 1
    #print(df_label.head(5))
    df_feature[label_columns] = df_label.values

    new_df_featurev = df_feature.iloc[:,-n:]
    #df_inited = pd.DataFrame([split_str(n,x) for x in feature_df], index=df.index).astype('int32')
    #label_columns = df_inited.columns.to_list()[n-1]
    #print(label_columns)
    #df_label = df_inited.loc[:,[label_columns]]
    #print(df_label.head(5)
    #df_label.loc[df_label[df_label.columns.to_list()[0]]>0]=1
    #print(df_label.head(5))
    #df_inited[label_columns] = df_label.values
    #print(df_inited)
    return new_df_featurev

def check_sample_balance(label_count):

    label_max = max(label_count.values)
    label_min = min(label_count.values)

    if int(label_max/label_min)>5:
        return True
    else:
        return False

def make_sample_balance(df):
    """
    :param df: format data
    :return: 0:error,df_balance:balance is completed
    """

    if df.empty:
        print("Warinning: dataset is empty!")
        return 0

    label_columns = df.columns.to_list()[-1]
    #print(label_columns)

    label_count = df.groupby([label_columns]).count().iloc[:,0]
    if not check_sample_balance(label_count):
        print("Sample sum is balance")
        return 0

    '''
    两种模式：
    1）减少样本多的数量
    '''

    df_label_max = df[df[label_columns] == label_count.idxmax()]
    #print(df_label_max.head(5))
    df_label_min = df[df[label_columns] == label_count.idxmin()]
    #print(df_label_min.head(5))

    df_label_max.reset_index(drop=True, inplace=True)

    drop_indices = np.random.choice(df_label_max.index, max(label_count.values)-min(label_count.values), replace=False)
    df_drop = df_label_max.drop(drop_indices)
    #print(df_drop.head(6))
    #print(df_drop.shape)

    df_balance = pd.concat([df_label_min,df_drop])
    df_balance.reset_index(drop=True, inplace=True)
    #print(df_balance.shape)

    return df_balance

def make_sample_balance_(df):

    """
    :param df:
    :return:
    """

    if df.empty:
        print("Warinning: dataset is empty!")
        return 0

    label_columns = df.columns.to_list()[-1]
    #print(label_columns)

    label_count = df.groupby([label_columns]).count().iloc[:,0]
    if not check_sample_balance(label_count):
        print("Sample sum is balance")
        return 0

    '''
    第二种模式：
    2）增加样本少的数量
    '''



