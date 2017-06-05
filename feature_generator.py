#! usr/bin/python
# #coding=utf-8

#导入数值计算库
import numpy as np
#导入科学计算库
import pandas as pd

# ont-hot-feature

def one_hot_feature(data, columns):
    return pd.get_dummies(data, columns=columns)

# cal-sum by ids, norm or standardization?
# recentn, recent minites e.g. recentn = 10000 means oneday

def cal_count_byid(data, target_data, id, query="label==1", new_column = None):

    if new_column == None:
        new_column = id+"_count"

    count_df = pd.DataFrame(data.query(query).groupby(id).size(), columns=[new_column])
    return pd.merge(target_data,count_df,left_on=id,right_index=True,how='left').fillna(0)


def cal_count_byids(data, target_data, ids, query="label==1"):

    count_df = pd.DataFrame(data.query(query).groupby(ids).size(), columns=['2'.join(ids)+'_count'])
    return pd.merge(target_data,count_df,left_on=ids,right_index=True,how='left').fillna(0)

# cal-cvr by id2id

def cal_cvr_byids(data, target_data, ids, query="label==1"):


    conv = data.query(query).groupby(ids).size()
    imp = data.groupby(ids).size()
    conv_rate = (conv/(imp+1)).fillna(0)
    convRate = pd.DataFrame(conv_rate, columns=['2'.join(ids)+'_conv_rate'])
    return pd.merge(target_data,convRate,left_on=ids,right_index=True,how='left').fillna(0)

def generate_user_feature(user, action, user_installed_apps, user_app_actions):

    print "starting generate user features."
    # basic_features

    bins = [0,1,18,28,35,60,80]
    user['age_cut'] = pd.cut(user['age'], bins, labels=False).fillna(0)
    user_feature = one_hot_feature(user, ['age_cut','gender','education','marriageStatus','haveBaby','hometown','residence'])

    # stat_features

    user_feature = cal_count_byid(action, user_feature, 'userID', query="clickTime>0", new_column='tot_clk') # tot_clk
    user_feature = cal_count_byid(action, user_feature, 'userID', query="label==1", new_column='tot_cvt') # tot_cvt
    user_feature = cal_cvr_byids(action, user_feature, ['userID'],query="label==1") # cvr

    user_feature = pd.merge(user_feature, pd.DataFrame(user_installed_apps.groupby('userID').size(), columns=['install_apps']), \
    left_on='userID', right_index=True, how='left').fillna(0)

    # user_recent_install
    startTime = max(action.clickTime)
    install_app_1h = pd.DataFrame(user_app_actions[user_app_actions.installTime>=(startTime-60)].groupby('userID').size(), columns=['install_app_1h'])
    install_app_1d = pd.DataFrame(user_app_actions[user_app_actions.installTime>=(startTime-10000)].groupby('userID').size(), columns=['install_app_1d'])
    install_app_3d = pd.DataFrame(user_app_actions[user_app_actions.installTime>=(startTime-30000)].groupby('userID').size(), columns=['install_app_3d'])
    install_app_7d = pd.DataFrame(user_app_actions[user_app_actions.installTime>=(startTime-70000)].groupby('userID').size(), columns=['install_app_7d'])

    user_feature = pd.merge(user_feature, install_app_1h, left_on='userID', right_index=True, how='left').fillna(0)
    user_feature = pd.merge(user_feature, install_app_1d, left_on='userID', right_index=True, how='left').fillna(0)
    user_feature = pd.merge(user_feature, install_app_3d, left_on='userID', right_index=True, how='left').fillna(0)
    user_feature = pd.merge(user_feature, install_app_7d, left_on='userID', right_index=True, how='left').fillna(0)

    # lack user-category
    print "end generate user features."

    return user_feature

def generate_ad_feature(action, ad, app_categories, user_installed_apps, user_app_actions):

    # app -> creative -> ad -> campaign -> advertiser

    print "starting generate ad features."

    # app-onehot

    app_feature = one_hot_feature(app_categories, ['appCategory'])

    # app_install_num

    app_feature = cal_count_byid(user_installed_apps, app_feature, 'appID', query='appID>0',new_column='app_install_num')

    ad_related_column = ['creativeID','adID','camgaignID','advertiserID','appID']

    ad_feature = ad

    for col in ad_related_column:
        ad_feature = cal_cvr_byids(action, ad_feature, [col])
        ad_feature = cal_count_byid(action, ad_feature, col)

    # merge ad with app

    ad_feature = pd.merge(ad_feature, app_feature, how='left', left_on='appID', right_on='appID')

    print "end generate user features."

    return ad_feature

def generate_position_feature():
    pass

def generate_user_item_feature(action, target):#join target

    print "starting generate user_item features."
    user_columns = ['userID','gender','haveBaby','age','education','hometown','residence']
    item_columns = ['appCategory','appID','positionID','sitesetID','positionType','connectionType','telecomsOperator','appPlatform','camgaignID']


    for u in user_columns:
        for i in item_columns:
            col = [u,i]
            target = cal_cvr_byids(action, target, col)


    user_target_feature = target

    # user-blabla

    print "end generate user_item features."

    return user_target_feature

def generate_advertiser_target_feature():
    pass

def generate_user_fm_vector():
    pass

def generate_ad_fm_vector():
    pass