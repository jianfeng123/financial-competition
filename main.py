# Load in our libraries
import pandas as pd
import re
# import plotly.offline as py
import copy
import random
# py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.offline as py
import xgboost as xgb
import datetime
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import Day, QuarterEnd
import matplotlib.ticker as mticker


def evaluate_models(models, X, y):
    guessed_sales = np.array([model.guess(X) for model in models])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = np.absolute((y - mean_sales) / (y + 0.00000001))
    result = np.sum(relative_err) / len(y)
    return result
def prediction_revenue(models, X):
    prediction_sales = np.array([model.guess(X) for model in models])
    mean_sales = prediction_sales.mean(axis=0)
    return mean_sales
class Model(object):
    def evaluate(self, X_val, y_val):
        # assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = np.absolute((y_val - guessed_sales) / (y_val + 0.00001))
        index_min_g = np.argmin(relative_err, axis=0)
        index_max_g = np.argmax(relative_err, axis=0)
        max_g = relative_err.max()
        min_g = relative_err.min()
        sum_g = np.sum(relative_err)
        result = np.sum(relative_err) / len(y_val)
        return result


class XGBoost(Model):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.max_y = max(y_train.max(), y_val.max())
        self.min_y = min(y_train.min(), y_val.min())
        self.max_log_y = np.log(self.max_y - self.min_y)

        y_label = np.log1p(y_train-self.min_y)
        dtrain = xgb.DMatrix(X_train, label=y_label)
        evallist = [(dtrain, 'train')]
        param = {
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = 1000
        self.bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=True)
        # self.bst = xgb.train(param, dtrain, num_round)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return (np.expm1(self.bst.predict(dtest)) + self.min_y)

################上下两行相减函数，同时去除孤零零的行或者起始行#################
#########上下两行相减加上滤出异常行，正好可以保证做季度差。同时让日期同步变化##########
def self_subtract_x(train_x):
    Id_and_date = train_x.loc[:,'TICKER_SYMBOL':'END_DATE']                                         # 将TICKER_SYMBOL和DATE提取出来
    Id_and_date.rename(columns={'TICKER_SYMBOL':'ID', 'END_DATE':'DATE'}, inplace = True)           # 将这两个列重命名，以便并入错位相减后的dateFrame中
    Id_and_date = Id_and_date.shift(-1)                                                             # 将这个dateFrame向上移一行
    new_df = train_x.shift(-1) - train_x                                                            # 错位相减，用下一行减去上一行
    new_df = pd.concat([new_df, Id_and_date], axis=1)                                           # 合并
    # remove_index代表的是train_x中，孤零零的一行或者起始行，通过做日期判断或者当前的ID进行判断，可以识别出来
    remove_index = new_df[(pd.Timedelta('90 days')>new_df['END_DATE']) | (pd.Timedelta('92 days')<new_df['END_DATE']) | (new_df['TICKER_SYMBOL'] != 0)].index
    new_df = new_df.drop(remove_index, axis=0)                                                      # 将这些行去除，以找出相差一个季度的值
    new_df['TICKER_SYMBOL'] = new_df['ID']
    new_df['END_DATE'] = new_df['DATE']
    new_df = new_df.drop(['ID', 'DATE'], axis=1)                                                    # 覆盖相减之后的值
    return new_df

###############营收的上下两行相减和train_x的略有区别，它主要是需要排除第一个季度的减值
def self_subtract_y(train_y):
    train_y['MONTH'] = train_y['END_DATE'].map(lambda t: t.month)                                   # 将月份提取出来，为后面做准备
    Id_and_date = train_y.loc[:,('TICKER_SYMBOL','END_DATE')]                                       # train_x一样
    Id_and_date.rename(columns={'TICKER_SYMBOL':'ID', 'END_DATE':'DATE'}, inplace = True)
    Id_and_date = Id_and_date.shift(-1)
    new_df = train_y.shift(-1) - train_y                                                            # 错位相减
    subs_df = train_y.shift(-1)[train_y.shift(-1)['MONTH']==3]                                      # 找出第一季度对应的收入值，采用shift的原因主要是为了使index匹配上，以便后面的替换操作
    new_df = pd.concat([new_df, Id_and_date], axis=1)                                           # 合并
    remove_index = new_df[(pd.Timedelta('90 days')>new_df['END_DATE']) | (pd.Timedelta('92 days')<new_df['END_DATE']) | (new_df['TICKER_SYMBOL'] != 0)].index
    new_df = new_df.drop(remove_index, axis=0)
    new_df['TICKER_SYMBOL'] = new_df['ID']
    new_df['END_DATE'] = new_df['DATE']
    new_df.drop(['ID','DATE','MONTH'], axis=1, inplace=True)                                        # 删除
    ##×××××重点××××××##
    ## 即将subs_df中的值与new_df中的值进行替换，只替换第一个季度。
    new_df['REVENUE'][(new_df['TICKER_SYMBOL'].isin(subs_df['TICKER_SYMBOL'])) & (new_df['END_DATE'].isin(subs_df['END_DATE']))] = subs_df['REVENUE'][(subs_df['TICKER_SYMBOL'].isin(new_df['TICKER_SYMBOL'])) & (subs_df['END_DATE'].isin(new_df['END_DATE']))]
    return new_df

###############营收的上下两行相减和train_x的略有区别，它主要是需要排除第一个季度的减值
def self_subtract(train_y):
    train_y['MONTH'] = train_y['END_DATE'].map(lambda t: t.month)                                   # 将月份提取出来，为后面做准备
    Id_and_date = train_y.loc[:,('TICKER_SYMBOL','END_DATE')]                                       # train_x一样
    Id_and_date.rename(columns={'TICKER_SYMBOL':'ID', 'END_DATE':'DATE'}, inplace = True)
    Id_and_date = Id_and_date.shift(-1)
    new_df = train_y.shift(-1) - train_y                                                            # 错位相减
    subs_df = train_y.shift(-1)[train_y.shift(-1)['MONTH']==3]                                      # 找出第一季度对应的收入值，采用shift的原因主要是为了使index匹配上，以便后面的替换操作
    new_df = pd.concat([new_df, Id_and_date], axis=1)                                           # 合并
    remove_index = new_df[(pd.Timedelta('90 days')>new_df['END_DATE']) | (pd.Timedelta('92 days')<new_df['END_DATE']) | (new_df['TICKER_SYMBOL'] != 0)].index
    new_df = new_df.drop(remove_index, axis=0)
    new_df['TICKER_SYMBOL'] = new_df['ID']
    new_df['END_DATE'] = new_df['DATE']
    new_df.drop(['ID','DATE','MONTH'], axis=1, inplace=True)                                        # 删除
    ##×××××重点××××××##
    ## 即将subs_df中的值与new_df中的值进行替换，只替换第一个季度。
    new_df[(new_df['TICKER_SYMBOL'].isin(subs_df['TICKER_SYMBOL'])) & (new_df['END_DATE'].isin(subs_df['END_DATE']))] = subs_df[(subs_df['TICKER_SYMBOL'].isin(new_df['TICKER_SYMBOL'])) & (subs_df['END_DATE'].isin(new_df['END_DATE']))]
    return new_df

###########################################
############# 一般企业数据提取 ###############
###########################################
def data_extract(balance1, cash1, income1, switch=0):
    ## 无关系数删除
    balance_na = (balance1.isnull().sum() / len(balance1)) * 100                          # 计算income的general Business的数据缺失比率
    balance1 = balance1.drop(balance_na[balance_na>98].index, axis=1)
    balance1 = balance1.drop('FISCAL_PERIOD', axis=1)
    balance1 = balance1.drop('REPORT_TYPE', axis=1)
    balance1 = balance1.fillna(0)
    cash_na = (cash1.isnull().sum() / len(cash1)) * 100
    cash1 = cash1.drop(cash_na[cash_na>98].index, axis=1)
    cash1 = cash1.drop('FISCAL_PERIOD', axis=1)
    cash1 = cash1.drop('REPORT_TYPE', axis=1)
    cash1 = cash1.fillna(0)
    income_na = (income1.isnull().sum() / len(income1)) * 100
    income1 = income1.drop(income_na[income_na>98].index, axis=1)
    income1 = income1.drop('FISCAL_PERIOD', axis=1)
    income1 = income1.drop('REPORT_TYPE', axis=1)
    income1 = income1.fillna(0)
    balance = copy.deepcopy(balance1)
    cash = copy.deepcopy(cash1)
    income = copy.deepcopy(income1)
    ## %% 训练数据和标签提取
    balance = self_subtract_x(balance)
    cash = self_subtract(cash)
    income = self_subtract(income)
    new_bc = pd.merge(balance, cash, how='inner', on=['TICKER_SYMBOL','END_DATE'])
    train_x = pd.merge(new_bc, income, how='inner', on=['TICKER_SYMBOL','END_DATE'])
    train_y = income.loc[:, ['TICKER_SYMBOL','END_DATE', 'REVENUE']]
    train_y['END_DATE'] = train_y['END_DATE'].map(lambda t: t-QuarterEnd())
    lbl = LabelEncoder()
    lbl.fit(list(train_x['TICKER_SYMBOL'].values))
    train_x['ID'] = lbl.transform(list(train_x['TICKER_SYMBOL'].values))
    train_x['MONTH'] = train_x['END_DATE'].map(lambda t: t.month)
    lbl.fit(list(train_x['MONTH'].values))
    train_x['MONTH'] = lbl.transform(list(train_x['MONTH'].values))
    remove_train_index = []
    for col in list(train_x.columns):
        if col == 'REVENUE':
            remove_train_index.append(col)
        if col == 'T_REVENUE':
            remove_train_index.append(col)
        if col == 'INT_INCOME':
            remove_train_index.append(col)
        if col == 'COMMIS_INCOME':
            remove_train_index.append(col)
        if col == 'PREM_EARNED':
            remove_train_index.append(col)
        if col == 'SPEC_TOR':
            remove_train_index.append(col)
        if col == 'N_INT_INCOME':
            remove_train_index.append(col)
    train_x = train_x.drop(remove_train_index, axis=1)
    test_x = train_x[train_x['END_DATE']==pd.to_datetime('2018-03-31')]
    train_x_y = pd.merge(train_x, train_y, how='inner', on=['TICKER_SYMBOL','END_DATE'])
    ## 计算相关系数
    correlation = train_x_y.corr()
    correlation = correlation.drop(['REVENUE', 'TICKER_SYMBOL', 'ID', 'MONTH'], axis=0)       # 删除无关行
    remove_index = correlation['REVENUE'][abs(correlation['REVENUE'])<0.02].index             # 相关行小的标签
    train_x_y = train_x_y.drop(remove_index, axis=1)
    test_x = test_x.drop(remove_index, axis=1)
    train_x_y.drop((train_x_y[train_x_y['REVENUE'] == 0]).index, axis=0, inplace=True)
    if switch==1:
        train_x_y.drop((train_x_y[train_x_y['REVENUE'] == train_x_y['REVENUE'].min()]).index, axis=0, inplace=True)
        train_x_y.drop((train_x_y[train_x_y['REVENUE'] == train_x_y['REVENUE'].min()]).index, axis=0, inplace=True)

    train_x_y['END_DATE'] = train_x_y['MONTH']
    train_x_y.drop(['MONTH'], axis=1, inplace=True)
    train_x_y.rename(columns={'END_DATE':'MONTH'}, inplace=True)
    test_x['END_DATE'] = test_x['MONTH']
    test_x.drop(['MONTH'], axis=1, inplace=True)
    test_x.rename(columns={'END_DATE':'MONTH'}, inplace=True)
    remove_test_index = []
    for col in list(test_x.columns):
        if col != 'MONTH' and col != 'END_DATE':
            if(len(test_x[test_x[col] == 0])/len(test_x))>=0.98:
                remove_test_index.append(col)
    train_x_y.drop(remove_test_index, axis=1, inplace=True)
    test_x.drop(remove_test_index, axis=1, inplace=True)
    return train_x_y, test_x

def sample(train_X):
    train_len_GB = int(len(train_X) * 1)
    arr_A = random.sample(range(len(train_X)), train_len_GB)
    train_index = arr_A[:int(train_len_GB*0.9)]
    val_index = arr_A[int(train_len_GB*0.9):]
    train_X_BU = train_X.loc[train_index, :]
    val_X_BU = train_X.loc[val_index, :]
    return train_X_BU, val_X_BU

def shuffle(train, val):
    train.index = range(0, len(train))
    val.index = range(0, len(val))
    arr_T = random.sample(range(len(train)), len(train))
    arr_V = random.sample(range(len(val)), len(val))
    train = train.loc[arr_T,:]
    val = val.loc[arr_V,:]

    train_y = train.loc[:, 'REVENUE']  # 标签
    train_y = train_y.values
    train_x = train.drop(['REVENUE', 'TICKER_SYMBOL'], axis=1)  # 训练数据
    train_x = train_x.values
    val_y = val.loc[:, 'REVENUE']
    val_y = val_y.values
    val_x = val.drop(['REVENUE', 'TICKER_SYMBOL'], axis=1)
    val_x = val_x.values
    return train_x, train_y, val_x, val_y

if __name__ == '__main__':
    path = '../data/'
    business = ['General Business', 'Bank', 'Insurance', 'Securities']
    # read data
    path_balance = 'Balance Sheet.xls'
    path_cash = 'Cashflow Statement.xls'
    path_income = 'Income Statement.xls'
    #%% reload the data
    balanceB = pd.read_excel(path+path_balance, sheetname=business)
    cashB = pd.read_excel(path+path_cash, sheetname=business)
    incomeB = pd.read_excel(path+path_income, sheetname=business)
    balance = copy.deepcopy(balanceB)
    cash = copy.deepcopy(cashB)
    income = copy.deepcopy(incomeB)
#### 三个报表数据的预处理 ######
    for key in business:
        balance[key]['END_DATE'] = pd.to_datetime(balance[key].END_DATE)                        # 将日期转变为可排序的类型
        income[key]['END_DATE'] = pd.to_datetime(income[key].END_DATE)                          #
        cash[key]['END_DATE'] = pd.to_datetime(cash[key].END_DATE)

        balance[key]['END_DATE_REP'] = pd.to_datetime(balance[key].END_DATE_REP)
        income[key]['END_DATE_REP'] = pd.to_datetime(income[key].END_DATE_REP)
        balance[key]['END_DATE_REP'] = pd.to_datetime(balance[key].END_DATE_REP)

        balance[key] = balance[key].sort_values(by=['TICKER_SYMBOL','END_DATE','END_DATE_REP'])  # 按照这三个属性进行排序，先排第一个，然后第二个
        cash[key] = cash[key].sort_values(by=['TICKER_SYMBOL','END_DATE','END_DATE_REP'])
        income[key] = income[key].sort_values(by=['TICKER_SYMBOL','END_DATE','END_DATE_REP'])
        balance[key] = balance[key].drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'], keep='last')    # 去掉响应的重复项，记住前面加上ID信息
        income[key] = income[key].drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'], keep='last')
        cash[key] = cash[key].drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'], keep='last')
        balance[key] = balance[key].drop(['MERGED_FLAG', 'PARTY_ID', 'EXCHANGE_CD', 'PUBLISH_DATE', 'END_DATE_REP'], axis=1)                                  # 去除merged flag
        cash[key] = cash[key].drop(['MERGED_FLAG', 'PARTY_ID', 'EXCHANGE_CD', 'PUBLISH_DATE', 'END_DATE_REP'], axis=1)
        income[key] = income[key].drop(['MERGED_FLAG', 'PARTY_ID', 'EXCHANGE_CD', 'PUBLISH_DATE', 'END_DATE_REP'], axis=1)
        balance[key].index = range(1, len(balance[key]) + 1)                                     # 将索引值重排序
        cash[key].index = range(1, len(cash[key]) + 1)
        income[key].index = range(1, len(income[key]) + 1)
## 数据预处理
## 一般工商业
    train_xy_GB, test_x_GB = data_extract(balance[business[0]], cash[business[0]], income[business[0]], switch=1)
## 银行
    train_xy_BA, test_x_BA = data_extract(balance[business[1]], cash[business[1]], income[business[1]])
## 保险
    train_xy_IN, test_x_IN = data_extract(balance[business[2]], cash[business[2]], income[business[2]])
## 证券
    train_xy_SE, test_x_SE = data_extract(balance[business[3]], cash[business[3]], income[business[3]])

## 标签重定义
    train_xy_GB.index = range(0, len(train_xy_GB))                                     # 将索引值重排序
    test_x_GB.index = range(0, len(test_x_GB))                                     # 将索引值重排序
    train_xy_BA.index = range(0, len(train_xy_BA))                                     # 将索引值重排序
    test_x_BA.index = range(0, len(test_x_BA))                                     # 将索引值重排序
    train_xy_IN.index = range(0, len(train_xy_IN))                                     # 将索引值重排序
    test_x_IN.index = range(0, len(test_x_IN))                                     # 将索引值重排序
    train_xy_SE.index = range(0, len(train_xy_SE))                                     # 将索引值重排序
    test_x_SE.index = range(0, len(test_x_SE))                                     # 将索引值重排序

# 生成训练集和测试集
    result_GB = pd.DataFrame({'TICKER_SYMBOL':test_x_GB['TICKER_SYMBOL']})
    test_x_GB = test_x_GB.drop('TICKER_SYMBOL', axis=1)  # 训练数据
    test_x_GB = test_x_GB.values

    result_BA = pd.DataFrame({'TICKER_SYMBOL':test_x_BA['TICKER_SYMBOL']})
    test_x_BA = test_x_BA.drop('TICKER_SYMBOL', axis=1)  # 训练数据
    test_x_BA = test_x_BA.values

    result_IN = pd.DataFrame({'TICKER_SYMBOL':test_x_IN['TICKER_SYMBOL']})
    test_x_IN = test_x_IN.drop('TICKER_SYMBOL', axis=1)  # 训练数据
    test_x_IN = test_x_IN.values

    result_SE = pd.DataFrame({'TICKER_SYMBOL':test_x_SE['TICKER_SYMBOL']})
    test_x_SE = test_x_SE.drop('TICKER_SYMBOL', axis=1)  # 训练数据
    test_x_SE = test_x_SE.values

    models_GB = []
    models_BA = []
    models_IN = []
    models_SE = []
    train_GB, val_GB = sample(train_xy_GB)
    train_BA, val_BA = sample(train_xy_BA)
    train_IN, val_IN = sample(train_xy_IN)
    train_SE, val_SE = sample(train_xy_SE)
    print("Fitting XGBOOST")
    for i in range(5):
        ## 一般工商业训练
        print("iteration {0} for GB".format(i))
        train_x_GB, train_y_GB, val_x_GB, val_y_GB = shuffle(train_GB, val_GB)
        models_GB.append(XGBoost(train_x_GB, train_y_GB, val_x_GB, val_y_GB))
        ## 银行业训练
        print("iteration {0} for BA".format(i))
        train_x_BA, train_y_BA, val_x_BA, val_y_BA = shuffle(train_BA, val_BA)
        models_BA.append(XGBoost(train_x_BA, train_y_BA, val_x_BA, val_y_BA))
        ## 保险业训练
        print("iteration {0} for IN".format(i))
        train_x_IN, train_y_IN, val_x_IN, val_y_IN = shuffle(train_IN, val_IN)
        models_IN.append(XGBoost(train_x_IN, train_y_IN, val_x_IN, val_y_IN))
        ## 证券业训练
        print("iteration {0} for SE".format(i))
        train_x_SE, train_y_SE, val_x_SE, val_y_SE = shuffle(train_SE, val_SE)
        models_SE.append(XGBoost(train_x_SE, train_y_SE, val_x_SE, val_y_SE))
    ### 一般工商业
    print("General business revenue prediction")
    result_GB['REVENUE'] = prediction_revenue(models_GB, test_x_GB)
    ### 银行业
    print("Bank business revenue prediction")
    result_BA['REVENUE'] = prediction_revenue(models_BA, test_x_BA)
    ### 保险业
    print("Insurance business revenue prediction")
    result_IN['REVENUE'] = prediction_revenue(models_IN, test_x_IN)
    ### 证券业
    print("Security business revenue prediction")
    result_SE['REVENUE'] = prediction_revenue(models_SE, test_x_SE)

    ## 生成结果
    ##%% 一般工商业
    result_GB_3 = income[business[0]]
    result_GB_3 = result_GB_3[result_GB_3['END_DATE'] == pd.to_datetime('2018-03-31')]
    result_GB_3 = result_GB_3.loc[:,('TICKER_SYMBOL', 'REVENUE')]
    result_GB['REVENUE'][result_GB['TICKER_SYMBOL'].isin(result_GB_3['TICKER_SYMBOL'])]=result_GB_3['REVENUE'][result_GB_3['TICKER_SYMBOL'].isin(result_GB['TICKER_SYMBOL'])].values + result_GB['REVENUE'][result_GB['TICKER_SYMBOL'].isin(result_GB_3['TICKER_SYMBOL'])].values
    # 对异常值进行处理, 这边可能你们给的数据有错误，股票代码563和627在一般工商业的资产负债表中存在，但是在营业收入表中却不存在，最后查到这两个数据在银行业的营业收入表中，因为缺失3月份的数据，所以这里采用两倍的夏季季度的营收来表示2018年半季度的营收。
    result_GB['REVENUE'][result_GB['TICKER_SYMBOL'] == 563] = result_GB['REVENUE'][result_GB['TICKER_SYMBOL'] == 563]*2
    result_GB['REVENUE'][result_GB['TICKER_SYMBOL'] == 627] = result_GB['REVENUE'][result_GB['TICKER_SYMBOL'] == 627]*2

    ##%% 银行业
    result_BA_3 = income[business[1]]
    result_BA_3 = result_BA_3[result_BA_3['END_DATE'] == pd.to_datetime('2018-03-31')]
    result_BA_3 = result_BA_3.loc[:,('TICKER_SYMBOL', 'REVENUE')]
    result_BA['REVENUE'][result_BA['TICKER_SYMBOL'].isin(result_BA_3['TICKER_SYMBOL'])] = result_BA_3['REVENUE'][result_BA_3['TICKER_SYMBOL'].isin(result_BA['TICKER_SYMBOL'])].values + result_BA['REVENUE'][result_BA['TICKER_SYMBOL'].isin(result_BA_3['TICKER_SYMBOL'])].values

    ##%% 保险业
    result_IN_3 = income[business[2]]
    result_IN_3 = result_IN_3[result_IN_3['END_DATE'] == pd.to_datetime('2018-03-31')]
    result_IN_3 = result_IN_3.loc[:,('TICKER_SYMBOL', 'REVENUE')]
    result_IN['REVENUE'][result_IN['TICKER_SYMBOL'].isin(result_IN_3['TICKER_SYMBOL'])] = result_IN_3['REVENUE'][result_IN_3['TICKER_SYMBOL'].isin(result_IN['TICKER_SYMBOL'])].values + result_IN['REVENUE'][result_IN['TICKER_SYMBOL'].isin(result_IN_3['TICKER_SYMBOL'])].values

    ##%% 证券业
    result_SE_3 = income[business[3]]
    result_SE_3 = result_SE_3[result_SE_3['END_DATE'] == pd.to_datetime('2018-03-31')]
    result_SE_3 = result_SE_3.loc[:,('TICKER_SYMBOL', 'REVENUE')]
    result_SE['REVENUE'][result_SE['TICKER_SYMBOL'].isin(result_SE_3['TICKER_SYMBOL'])] = result_SE_3['REVENUE'][result_SE_3['TICKER_SYMBOL'].isin(result_SE['TICKER_SYMBOL'])].values + result_SE['REVENUE'][result_SE['TICKER_SYMBOL'].isin(result_SE_3['TICKER_SYMBOL'])].values

    ## 结果生成部分
    result = pd.read_csv('../data/FDDC_financial_submit_20180524.csv', header=None)
    result.rename(columns={0:'TICKER_SYMBOL'},inplace=True)
    data = [int(re.sub("\D", "", str(value))) for value in result.values]
    result_df = pd.DataFrame(columns=['TICKER_SYMBOL', 'REVENUE'])
    result_df['TICKER_SYMBOL'] = data
    Save_ID = data
    result_df['REVENUE'][result_df['TICKER_SYMBOL'].isin(result_GB['TICKER_SYMBOL'])] = result_GB['REVENUE'][result_GB['TICKER_SYMBOL'].isin(result_df['TICKER_SYMBOL'])].values
    result_df['REVENUE'][result_df['TICKER_SYMBOL'].isin(result_BA['TICKER_SYMBOL'])] = result_BA['REVENUE'][result_BA['TICKER_SYMBOL'].isin(result_df['TICKER_SYMBOL'])].values
    result_df['REVENUE'][result_df['TICKER_SYMBOL'].isin(result_IN['TICKER_SYMBOL'])] = result_IN['REVENUE'][result_IN['TICKER_SYMBOL'].isin(result_df['TICKER_SYMBOL'])].values
    result_df['REVENUE'][result_df['TICKER_SYMBOL'].isin(result_SE['TICKER_SYMBOL'])] = result_SE['REVENUE'][result_SE['TICKER_SYMBOL'].isin(result_df['TICKER_SYMBOL'])].values
    result_df['TICKER_SYMBOL'] = result['TICKER_SYMBOL']
    value = result_df['REVENUE'].map(lambda x: float(x)/1000000)
    format = lambda x : ("%.2f" %float(x))
    value = value.apply(format)
    result_df['REVENUE'] = value
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
    result_file = 'submit_' + nowTime + '.csv'
    result_df.to_csv('../submit/'+result_file, index=False, header=False)
    for ID in Save_ID:
        if ID !=563 and ID != 627:
            picture_path = '../Picture_result/'
            if(len(income[business[0]][income[business[0]]['TICKER_SYMBOL']==ID])!=0):
                date = pd.to_datetime('2018-06-30')
                date = pd.Series(date, index=None)
                x = copy.deepcopy(income[business[0]][income[business[0]]['TICKER_SYMBOL']==ID]['END_DATE'])
                x = pd.concat([x, date], ignore_index=True)
                y = copy.deepcopy(income[business[0]][income[business[0]]['TICKER_SYMBOL']==ID].REVENUE)
                py = result_GB['REVENUE'][result_GB['TICKER_SYMBOL']==ID]
                y = pd.concat([y, py], ignore_index=True)
                f, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(x=x, y=y)
                ticklabels = x.dt.strftime('%Y-%m-%d')
                ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
                f.autofmt_xdate()
                plt.xticks(rotation='60')
                plt.ylabel('REVENUE', fontsize=15)
                plt.xlabel('END_DATE', fontsize=15)
                plt.title('REVENUE for ID is '+str(int(ID)), fontsize=15)
                plt.savefig(picture_path+'General Business'+str(int(ID))+'.jpg')
                plt.close()
            elif(len(income[business[1]][income[business[1]]['TICKER_SYMBOL']==ID])!=0):
                date = pd.to_datetime('2018-06-30')
                date = pd.Series(date, index=None)
                x = copy.deepcopy(income[business[1]][income[business[1]]['TICKER_SYMBOL']==ID]['END_DATE'])
                x = pd.concat([x, date], ignore_index=True)
                y = copy.deepcopy(income[business[1]][income[business[1]]['TICKER_SYMBOL']==ID].REVENUE)
                py = result_BA['REVENUE'][result_BA['TICKER_SYMBOL']==ID]
                y = pd.concat([y, py], ignore_index=True)

                f, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(x=x, y=y)
                ticklabels = x.dt.strftime('%Y-%m-%d')
                ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
                f.autofmt_xdate()
                plt.xticks(rotation='60')
                plt.ylabel('REVENUE', fontsize=15)
                plt.xlabel('END_DATE', fontsize=15)
                plt.title('REVENUE for ID is '+str(int(ID)), fontsize=15)
                plt.savefig(picture_path+'Bank Business'+str(int(ID))+'.jpg')
                plt.close()
            elif(len(income[business[2]][income[business[2]]['TICKER_SYMBOL']==ID])!=0):
                date = pd.to_datetime('2018-06-30')
                date = pd.Series(date, index=None)
                x = copy.deepcopy(income[business[2]][income[business[2]]['TICKER_SYMBOL']==ID]['END_DATE'])
                x = pd.concat([x, date], ignore_index=True)
                y = copy.deepcopy(income[business[2]][income[business[2]]['TICKER_SYMBOL']==ID].REVENUE)
                py = result_IN['REVENUE'][result_IN['TICKER_SYMBOL']==ID]
                y = pd.concat([y, py], ignore_index=True)
                f, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(x=x, y=y)
                ticklabels = x.dt.strftime('%Y-%m-%d')
                ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
                f.autofmt_xdate()
                plt.xticks(rotation='60')
                plt.ylabel('REVENUE', fontsize=15)
                plt.xlabel('END_DATE', fontsize=15)
                plt.title('REVENUE for ID is '+str(int(ID)), fontsize=15)
                plt.savefig(picture_path+'Insurace Business'+str(int(ID))+'.jpg')
                plt.close()
            else:
                date = pd.to_datetime('2018-06-30')
                date = pd.Series(date, index=None)
                x = copy.deepcopy(income[business[3]][income[business[3]]['TICKER_SYMBOL']==ID]['END_DATE'])
                x = pd.concat([x, date], ignore_index=True)
                y = copy.deepcopy(income[business[3]][income[business[3]]['TICKER_SYMBOL']==ID].REVENUE)
                py = result_SE['REVENUE'][result_SE['TICKER_SYMBOL']==ID]
                y = pd.concat([y, py], ignore_index=True)
                f, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(x=x, y=y)
                ticklabels = x.dt.strftime('%Y-%m-%d')
                ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
                f.autofmt_xdate()
                plt.xticks(rotation='60')
                plt.ylabel('REVENUE', fontsize=15)
                plt.xlabel('END_DATE', fontsize=15)
                plt.title('REVENUE for ID is '+str(int(ID)), fontsize=15)
                plt.savefig(picture_path + 'Security Business'+str(int(ID))+'.jpg')
                plt.close()
