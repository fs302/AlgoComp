#! usr/bin/python
# #coding=utf-8

#导入数值计算库
import numpy as np
#导入科学计算库
import pandas as pd
#导入数据预处理库
from sklearn import preprocessing
#导入交叉验证库
from sklearn import model_selection
#导入缺失值填充库
from sklearn.preprocessing import Imputer
import os


os.getcwd()
os.chdir('../Data/')
##1.读入数据
#训练
train = pd.read_csv("train.csv")
#用户
user = pd.read_csv("user.csv")
user_installedapps = pd.read_csv("user_installedapps.csv")
user_app_actions = pd.read_csv("user_app_actions.csv")
app_categories = pd.read_csv("app_categories.csv")
#广告维度
ad = pd.read_csv("ad.csv")
#上下文维度
position = pd.read_csv("position.csv")
#测试数据
test = pd.read_csv("test.csv")

#train -> user -> ad -> position -> app_categories

df = pd.merge(train, user, how='left', left_on='userID', right_on='userID')
#sum(pd.isnull(train_user.userID)) #0
df1 = pd.merge(df, ad, how='left', left_on='creativeID', right_on='creativeID')
#sum(pd.isnull(df1.creativeID)) 
df2 = pd.merge(df1, position, how='left', left_on='positionID', right_on='positionID')
#sum(pd.isnull(df2.positionID)) 
extendTrainingData = pd.merge(df2, app_categories, how='left', left_on='appID', right_on='appID')

train_0 = extendTrainingData[extendTrainingData.clickTime <= 300000]

train_1 = extendTrainingData[extendTrainingData.clickTime>300000]


train_0.to_csv('extend_train0.csv',index=False,header=True,sep=',')
train_1.to_csv('extend_train1.csv',index=False,header=True,sep=',')

sample_train0 = train_0.sample(n=10000)

sample_train0.to_csv('sample_train0.csv', index=False,header=True,sep=',')

df = pd.merge(test, user, how='left', left_on='userID', right_on='userID')
#sum(pd.isnull(train_user.userID)) #0
df1 = pd.merge(df, ad, how='left', left_on='creativeID', right_on='creativeID')
#sum(pd.isnull(df1.creativeID))
df2 = pd.merge(df1, position, how='left', left_on='positionID', right_on='positionID')
#sum(pd.isnull(df2.positionID))
extendTestData = pd.merge(df2, app_categories, how='left', left_on='appID', right_on='appID')


extendTestData.to_csv('extend_test.csv',index=False,header=True,sep=',')
