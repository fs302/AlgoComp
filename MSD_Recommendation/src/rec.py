# Using models to make recomendation
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys,utils,models,time
from collections import defaultdict
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

class Bipartite(object):
    def __init__(self, u2s, s2u, user2id, song2id):

        self.node_features = []
        self.uid2nodeid = {}
        self.sid2nodeid = {}
        self.adj_lists = defaultdict(set)

        print "Bipartite Network:"
        # node_features
        nodeid = 0
        self.uid2nodeid = {}
        for user in user2id:
            uid = user2id[user]
            if uid not in u2s: # pass non_networked nodes
                continue
            self.uid2nodeid[uid] = nodeid
            node_type = 0 # user
            node_degree = len(u2s[uid]) if uid in u2s else 0 # user degree
            content = [node_type, node_degree]
            self.node_features.append(content)
            nodeid += 1
        for song in song2id:
            sid = song2id[song]
            if sid not in s2u: # pass non_networked nodes
                continue
            self.sid2nodeid[sid] = nodeid
            node_type = 1 # item
            node_degree = len(s2u[sid]) if sid in s2u else 0# song degree
            content = [node_type, node_degree]
            self.node_features.append(content)
            nodeid += 1
        print "\t#",len(self.node_features)," Node Features."

        self.num_nodes = len(self.node_features)
        self.num_feats = len(self.node_features[0])

        self.feature_matrix = np.zeros((self.num_nodes, self.num_feats))
        for i in range(self.num_nodes):
            self.feature_matrix[i,:] = map(float, self.node_features[i])

        # adj_lists
        for uid,sids in u2s.items():
            for sid in sids:
                id1 = self.uid2nodeid[uid]
                id2 = self.sid2nodeid[sid]
                if id1 != None and id2 != None:
                    self.adj_lists[id1].add(id2)
                    self.adj_lists[id2].add(id1)
        print "\tConstructed Adjacency list."

if __name__ == "__main__":
    parser = ArgumentParser("rec",
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--user-min',default=0,type=int,
        help="Min index of rec user.")
    parser.add_argument('--user-max',default=10,type=int,
        help="Max index of rec user.")
    parser.add_argument('--output',default='rec_result.txt',
        help="Output Result File.")
    parser.add_argument('--model', default='DeepWalk',
        help="Recommend Model:ItemCF/UserCF/DeepWalk/PinSage")
    parser.add_argument('--emb_file',default='../train/song_deepwalk.emb',
        help="Item Embedding File.")
    parser.add_argument('--feature',default='../train/node_feature.txt',
        help="Feature File.")
    parser.add_argument('--feature',default='../train/node_feature.txt',
        help="Feature File.")
    parser.add_argument('--sim', default='cos',
        help="Similarity Measure")
    parser.add_argument('--alpha', default=0.5,type=float,
        help="Similarity Measure")
    parser.add_argument('--q', default=1,type=float,
        help="parameter Q")
    parser.add_argument('--knn', default=100,type=int,
        help="k Nearest Neighbor")
    parser.add_argument('--topk', default=500,type=int,
        help="Top k recommend result.")
    parser.add_argument('--debug', default=False,
        help="if print debug log.")
    
    args = parser.parse_args()

    print "user_min: %d , user_max: %d"%(args.user_min,args.user_max)
    sys.stdout.flush()

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
    

    # Make Recommendations
    if args.model == 'ItemCF':
        model = models.ItemCF(u2s, s2u, sim_method=args.sim, alpha=args.alpha, Q=args.q)
    if args.model == 'UserCF':
        model = models.UserCF(u2s, s2u, sim_method=args.sim, alpha=args.alpha, knn=args.knn)
    if args.model == 'DeepWalk':
        model = models.DeepWalk(u2s, s2u, args.emb_file)
    if args.model == 'PinSage':
        # Init
        bipartite = Bipartite(u2s, s2u, user2id, song2id)
        features = nn.Embedding(bipartite.num_nodes, bipartite.num_feats)
        print 'num of features:',bipartite.num_feats
        features.weight = nn.Parameter(torch.FloatTensor(bipartite.feature_matrix), requires_grad=False)
        model = models.UnsupervisedGraphSage(features, bipartite.num_feats
            , bipartite.adj_lists, margin=0.1, hidden_dim=32, embed_dim=64, is_gcn=False)
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.5)
        # Run model
        num_batch = 100
        rand_indices = np.random.permutation(bipartite.num_nodes)
        train = rand_indices[:10000]
        num_ns = 100         # number of negtive samples
        num_neigh = 10   # number of sample neighbors

        times = []
        for batch in range(num_batch):
            batch_nodes = train[:64]
            negtive_samples = rand_indices[-num_ns:]
            random.shuffle(rand_indices)
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = model.loss(batch_nodes, 
                negtive_samples,
                num_neigh)
            print loss.creator
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print batch, loss.data[0]
            params = list(model.parameters())
            print len(params),params

        # for uid in range(args.user_min, args.user_max):
        #     for sid in u2s[uid]:
        #         with torch.no_grad():
        #             print 'user',uid, bipartite.uid2nodeid[uid],model.emb(bipartite.uid2nodeid[uid])
        #             print 'item',sid, bipartite.sid2nodeid[sid],model.emb(bipartite.sid2nodeid[sid])
        #             print uid, sid, model.score(bipartite.uid2nodeid[uid],bipartite.sid2nodeid[sid])

    print "Saving Recommend Result: " + args.output
    f=open(args.output,"w")
    ct = 0
    sti=time.clock()
    for userid in range(args.user_min,args.user_max):
        reclist = model.makeRec(userid, u2s[userid],args.topk)
        indices = [str(song) for song in reclist]
        f.write(str(userid)+":"+" ".join(indices)+"\n")
        ct += 1
        cti = time.clock()-sti
        if args.debug and not (ct) % 10:
            print "No.%d Recommend for %s, tot sec: %f(per %f)" % (ct, userid, cti, cti/(ct))
    f.close()
    print "Recommend Finished."
