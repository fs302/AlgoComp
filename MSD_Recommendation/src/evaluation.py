import utils
import numpy
import sys

def read_ideal_list(ideal_file, user_file, song_file):
    user2id = {v:int(k) for k,v in enumerate(utils.load_users(user_file))}
    song2id = {k:int(v) for k,v in utils.song_to_idx(song_file).items()}
    u2s = {}
    with open(ideal_file,"r") as f:
        for line in f:
            user,song,cnt=line.strip().split('\t')
            userid = user2id[user]
            songid = song2id[song]
            if userid not in u2s:
                u2s[userid] = {}
            u2s[userid][songid] = int(cnt)
    ui_rank = {}
    for userid in u2s:
        rk_list = sorted(u2s[userid].items(),key=lambda x:x[1],reverse=True)
        ui_rank[userid] = [rk_list[i][0] for i in range(len(rk_list))]
    return ui_rank

def read_rec_list(rec_file):
	reclist = {}
	with open(rec_file,"r") as f:
		for line in f:
			data = line.strip().split(':')
			uid = int(data[0])
			reclist[uid] = [int(x) for x in data[1].split(' ')]
	return reclist


def get_AP(k,ideal,rec):
    """
        compute AP
    """
    ideal=set(ideal)
    accumulation=0.0 
    count=0 
    for i in range(len(rec)):
        if i>=k: 
            break
        if rec[i] in ideal: 
            count+=1
            accumulation+=count/(i+1.0)
    m=len(ideal) 
    x=min(m,k)
    if x==0:
        return 0 
    return accumulation/x
            
def get_MAP(k,ideal_map,rec_map):
    """
        compute MAP
    """
    accumulation=0.0
    # for key in ideal_map.keys(): 
    #     accumulation+=get_AP(k, ideal_map[key], rec_map[key]) 
    # if len(ideal_map)==0: 
    #     return 0
    # return accumulation/len(ideal_map)
    eval_cnt = 0
    for key in rec_map.keys(): 
    	if key in ideal_map:
        	accumulation+=get_AP(k, ideal_map[key], rec_map[key]) 
        	eval_cnt += 1
    if eval_cnt==0: 
        return 0
    return accumulation/eval_cnt
    

if __name__== "__main__":
    ideal_file = '../test/year1_valid_triplets_hidden.txt'
    user_file = '../train/kaggle_users.txt'
    song_file = '../train/kaggle_songs.txt'
    if len(sys.argv)>1:
    	submission_file = sys.argv[1]
    print 'Reading Ideal File:'+ideal_file
    ideal = read_ideal_list(ideal_file, user_file, song_file)
    sub1 =  'rec_result.txt'
    print 'Reading Recommend File:'+sub1
    rec = read_rec_list(sub1)
    print 'MAP:',get_MAP(500,ideal,rec)
    sub2 = '../winner_solution/win_submit.txt'
    print 'Reading Recommend File:'+sub2
    rec = read_rec_list(sub2)
    print 'MAP:',get_MAP(500,ideal,rec)