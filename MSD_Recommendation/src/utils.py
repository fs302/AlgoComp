
import random,time,math
import sys,os

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