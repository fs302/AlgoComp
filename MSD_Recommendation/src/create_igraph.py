# Create Item-Item Graph and Item Features
import utils
import models
import sys
import heapq

print sys.version

if __name__ == "__main__":
    train_dir = "../train/"
    test_dir = "../test/"
    train_triplets_file=train_dir+"kaggle_visible_evaluation_triplets.txt"
    test_triplets_file=train_dir+"year1_valid_triplets_hidden.txt"
    user_file = train_dir+'kaggle_users.txt'
    song_file = train_dir+'kaggle_songs.txt'

    print "loading user-song matrix."    
    user2id = {v:int(k) for k,v in enumerate(utils.load_users(user_file))}
    song2id = {k:int(v) for k,v in utils.song_to_idx(song_file).items()}

    s2u=utils.song_to_users(train_triplets_file,user2id,song2id)
    print "Song:",len(song2id),"Listened Song:",len(s2u)
    u2s=utils.user_to_songs(train_triplets_file,user2id,song2id)
    print "User:",len(user2id),"Active User:",len(u2s)

    # Item Similarity Network

    sim = 'cos'
    alpha = 0.5
    q = 1
    knn = 100
    model = models.ItemCF(u2s, s2u, sim_method=sim, alpha=alpha, Q=q)
    output = train_dir+"song_sim_edge.txt"
    stats = {}
    with open(output,'w') as f:
        for song1, related_songs in model.song_sim.items():
            neighs =  [(k,v) for k,v in related_songs.items()]
            selected = heapq.nlargest(knn,neighs,key=lambda v:v[1])
            for song2,score in selected:
                f.write(str(song1)+'\t'+str(song2)+'\t'+str(score)+'\n')
    
    # Node Feature

    node_file = train_dir+"node_feature.txt"

    nodeid = 0
    uid2nodeid = {}
    node_features = []
    for user in user2id:
        uid = user2id[user]
        uid2nodeid[uid] = nodeid
        node_type = 0 # user
        node_degree = len(u2s[uid]) if uid in u2s else 0 # user degree
        content = [user, str(nodeid), str(node_type), str(node_degree)]
        node_features.append(content)
        nodeid += 1
    sid2nodeid = {}
    for song in song2id:
        sid = song2id[song]
        sid2nodeid[sid] = nodeid
        node_type = 1 # item
        node_degree = len(s2u[sid]) if sid in s2u else 0# song degree
        content = [user, str(nodeid), str(node_type), str(node_degree)]
        node_features.append(content)
        nodeid += 1
    with open(node_file,'w') as f:
        for content in node_features:
            f.write(' '.join(content)+'\n')

    # Edge

    edge_file = train_dir+"bipartite_net.txt"

    with open(edge_file,'w') as f:
        for uid,sids in u2s.items():
            for sid in sids:
                f.write(str(uid2nodeid[uid])+' '+str(sid2nodeid[sid])+'\n')
        for sid,uids in s2u.items():
            for uid in uids:
                f.write(str(sid2nodeid[sid])+' '+str(uid2nodeid[uid])+'\n')

