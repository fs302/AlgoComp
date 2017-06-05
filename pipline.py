#! usr/bin/python
# #coding=utf-8

import numpy as np
import pandas as pd
import os
import utils
import feature_generator

'''

1. 确定数据集
2. 建立特征库
3. 模型训练
4. 数据验证

'''

os.getcwd()
os.chdir('../Data/')

def model_training_pipline():

    # load_data
    #train = pd.read_csv("extend_train0.csv")
    train = pd.read_csv("sample_train0.csv")
    user = pd.read_csv("user.csv").sample(n=10000)
    user_installed_apps = pd.read_csv("user_installedapps.csv")
    user_app_actions = pd.read_csv("user_app_actions.csv")
    app_categories = pd.read_csv("app_categories.csv")
    ad = pd.read_csv("ad.csv")
    position = pd.read_csv("position.csv")

    # feature_generate

    user_feature = feature_generator.generate_user_feature(user, train, user_installed_apps, user_app_actions)

    ad_feature = feature_generator.generate_ad_feature(train, ad, app_categories, user_installed_apps, user_app_actions)

    u2i_feature = feature_generator.generate_user_item_feature(train, train)

    feature = pd.merge(u2i_feature, ad_feature.drop, how='left', left_on='creativeID', right_on='creativeID')
    feature = pd.merge(feature, user_feature, how='left', left_on='userID', right_on='userID')
    print feature.columns
    exclude_column = ['clickTime','conversionTime','creativeID','userID','positionID','connectionType','telecomsOperator','age','gender']
    feature1 = feature.drop(exclude_column)
    feature1.to_csv('training_feature.csv',index=False,header=True,sep=',')

    return feature1

def predict_pipline():


    pass

if __name__ == "__main__":
    model_training_pipline()
