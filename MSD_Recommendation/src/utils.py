import random,time,math
import sys,os
import numpy as np
from scipy import sparse
import pandas as pd

def data_generation(raw_data, train_dir, song_user_limit=20, user_song_limit=20, dataset_prefix='msd'):
    
    user_song_cnt = raw_data.groupby(['user'])['count'].count()\
                                            .reset_index(name='song_cnt')\
                                            .sort_values(['song_cnt'], ascending=False)
    
    selected_user = user_song_cnt[user_song_cnt.song_cnt>user_song_limit]
    
    selected_user['userid'] = range(len(selected_user))
    
    filter_1 = pd.merge(raw_data, selected_user, on='user') 
    
    song_user_cnt = raw_data.groupby(['song'])['count'].count()\
                                            .reset_index(name='user_cnt')\
                                            .sort_values(['user_cnt'], ascending=False)
    selected_song = song_user_cnt[song_user_cnt.user_cnt>song_user_limit]
    
    selected_song['songid'] = range(len(selected_song))
    
    
    top_record = pd.merge(filter_1, selected_song, on='song')
    result = top_record.sort_values(by='userid')
    result['rate'] = 1
    result['timestamp'] = 0
    msk = np.random.rand(len(result)) < 0.8
    train = result[msk]
    test = result[~msk]
    train.to_csv(train_dir+dataset_prefix+'.train.rating',columns=['userid','songid','rate','timestamp'], index=False, sep='\t', header=False)
    test.to_csv(train_dir+dataset_prefix+'.test.rating',columns=['userid','songid','rate','timestamp'], index=False, sep='\t', header=False)
    test_neg_file = train_dir+dataset_prefix+'.test.negative'
    negative_candidate = selected_song[selected_song.songid.isin(result.songid.unique())]
    with open(test_neg_file,'w') as f:
        for index, row in test.iterrows():
            userid = row['userid']
            songid = row['songid']
            user_history = result[result.userid==userid]
            negative = negative_candidate[~negative_candidate.songid.isin(user_history.songid)].sample(99).songid
            f.write('('+str(userid)+','+str(songid)+')'+'\t'+'\t'.join([str(item) for item in negative])+'\n')


def load_train_data(csv_file):
    tp = pd.read_table(csv_file,header=None,names=['uid','sid','rate','timestamp'])
    n_users = tp['uid'].max() + 1
    n_items = tp['sid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return n_users, n_items, data

def load_test_data(csv_file, n_users, n_items):
    tp_tr = pd.read_table(csv_file,header=None,names=['uid','sid','rate','timestamp'])

    rows_tr, cols_tr = tp_tr['uid'], tp_tr['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(n_users, n_items))

    return data_tr

def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList
    
def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def song_to_count(if_str):
    stc=dict()
    with open(if_str,"r") as f:
        for line in f:
            _,song,_=line.strip().split('\t')
            if song in stc:
                stc[song]+=1
            else:
                stc[song]=1
    return stc

def user_to_count(if_str):
    utc=dict()
    with open(if_str,"r") as f:
        for line in f:
            user,_,_=line.strip().split('\t')
            if user in utc:
                utc[user]+=1
            else:
                utc[user]=1
    return utc

def sort_dict_dec(d):
    return sorted(d.keys(),key=lambda s:d[s],reverse=True)

def song_to_users(if_str, user2id, song2id,set_users=None, ratio=1.0):
    stu=dict()
    with open(if_str,"r") as f:
        for line in f:
            if random.random()<ratio:
                user,song,_=line.strip().split('\t')
                userid = user2id[user]
                songid = song2id[song]
                if not set_users or userid in set_users:
                    if songid in stu:
                        stu[songid].add(userid)
                    else:
                        stu[songid]=set([userid])
    return stu

def user_to_songs(if_str, user2id, song2id):
    uts=dict()
    with open(if_str,"r") as f:
        for line in f:
            user,song,_=line.strip().split('\t')
            userid = user2id[user]
            songid = song2id[song]
            if userid in uts:
                uts[userid].add(songid)
            else:
                uts[userid]=set([songid])
    return uts

def load_unique_tracks(if_str):
    ut=[]
    with open(if_str,"r") as f:
        for line in f:
            a_id,s_id,a,s=line.strip().split('<SEP>')
            ut.append((a_id,s_id,a,s))
    return ut

def load_users(if_str):
    with open(if_str,"r") as f:
        u=map(lambda line: line.strip(),f.readlines())
    return u

def song_to_idx(if_str):
     with open(if_str,"r") as f:
         sti=dict(map(lambda line: line.strip().split(' '),f.readlines()))
     return sti

def unique_users(if_str):
    u=set()
    with open(if_str,"r") as f:
        for line in f:
            user,_,_=line.strip().split('\t')
            if user not in u:
                u.add(user)
    return u 

def save_recommendations(r,songs_file,ofile):
    print "Loading song indices from " + songs_file
    s2i=song_to_idx(songs_file)
    print "Saving recommendations"
    f=open(ofile,"w")
    for uid in r:
        indices=map(lambda s: s2i[s],r[uid])
        f.write(str(uid)+":"+" ".join(indices)+"\n")
    f.close()
    print "Ok."