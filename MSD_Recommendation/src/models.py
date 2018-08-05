# Models for training
import math
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import random
from encoders import Encoder
import heapq
from aggregators import MeanAggregator

class Model(object):
    def __init__(self,u2s, s2u, model_name):
        self.model_name=model_name
        # print 'Constructing '+self.model_name
        self.s2u=s2u
        self.u2s=u2s
        self.pop_songs = []
        self.get_pop_songs()

    def get_pop_songs(self):
        ranklist = sorted(self.s2u.items(), key=lambda x:len(x[1]), reverse=True)
        self.pop_songs = [x[0] for x in ranklist]
        # print "\tRanking Pop Items. tot:",len(self.pop_songs)

    def formatRecommendation(self, origin_score, user_actions, rec_len):
        reclist = []
        for song, score in origin_score:
            # filter users listend songs
            if song not in user_actions:
                reclist.append(song)
        
        if len(reclist) < rec_len:
            budget = rec_len - len(reclist)
            for song in self.pop_songs:
                if song not in reclist:
                    reclist.append(song)
                    budget -= 1
                    if budget == 0:
                        break
        return reclist[:rec_len]

class ItemCF(Model):
    ''' Implement of ItemBased CF '''
    def __init__(self, u2s, s2u, sim_method='jaccard', alpha=1.0, Q=3, knn=100):
        Model.__init__(self, u2s, s2u, model_name='ItemCF_'+sim_method)
        self.alpha = alpha
        self.Q = Q
        self.song_sim = {}
        self.cal_item_sim(sim_method)
        self.knn=knn

    def cal_item_sim(self, sim_method='jaccard'):
        # print "\tCalc Item Similarity:"+sim_method+',alpha='+str(self.alpha)+",Q="+str(self.Q)
        C = dict()
        N = dict()
        for u, songs in self.u2s.items():
            for i in songs:
                N.setdefault(i,0)
                C.setdefault(i,{})
                N[i] += 1
                for j in songs:
                    if i == j:
                        continue
                    C[i].setdefault(j,0)
                    C[i][j] += 1

        for i, related_items in C.items():
            self.song_sim.setdefault(i,{})
            for j, cij in related_items.items():
                if sim_method=='jaccard':
                    self.song_sim[i][j] = 1.0*cij / (N[i]+N[j]-cij)
                elif sim_method=='cos':
                    self.song_sim[i][j] = 1.0*cij / (math.pow(N[i],1-self.alpha)*math.pow(N[j],self.alpha))
                else:
                    self.song_sim[i][j] = 1.0*cij

    def knn_score(self, user_actions, candidates):
        scores = {}
        for song in candidates:
            pool = {}
            if song not in self.song_sim:
                scores[song] = 0
                continue
            for related_song in user_actions:
                if related_song in self.song_sim[song]:
                    pool[related_song] = self.song_sim[song][related_song]
            sorted_pool = sorted(pool.items(), key=lambda x:x[1], reverse=True)
            scores[song] = sum(dict(sorted_pool[:self.knn]).values())
        return scores

    def score(self, userid, user_actions):
        candidates = {}
        for song in user_actions:
            if not song in self.song_sim:
                continue
            for target_song in self.song_sim[song]:
                candidates.setdefault(target_song,0.0)
                candidates[target_song] += math.pow(self.song_sim[song][target_song],self.Q)
        origin_result = sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        return origin_result

    def makeRec(self, userid, user_actions, rec_len):
        return Model.formatRecommendation(self, self.score(userid, user_actions), user_actions, rec_len)
        

class UserCF(Model):
    ''' Implement of UserBased CF '''
    def __init__(self, u2s, s2u, sim_method='jaccard', alpha=1.0, knn=100):
        Model.__init__(self, u2s, s2u, model_name='UserCF_'+sim_method)
        self.alpha = alpha
        self.user_sim = {}
        self.cal_user_sim(sim_method)
        self.knn = knn

    def cal_user_sim(self, sim_method='jaccard'):
        # print "\tCalc User Similarity:"+sim_method+',alpha='+str(self.alpha)
        C = dict()
        N = dict()
        cnt = 0
        for s, users in self.s2u.items():
            cnt += 1
            if cnt % 10000 == 0:
                print '\tsong#', cnt
            if len(users)>1000 or len(users)<2:
                continue
            for u in users:
                N.setdefault(u,0)
                C.setdefault(u,{})
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    C[u].setdefault(v,0)
                    C[u][v] += 1

        for u, related_users in C.items():
            self.user_sim.setdefault(u,{})
            for v, cuv in related_users.items():
                if sim_method=='jaccard':
                    self.user_sim[u][v] = cuv / (N[u]+N[v]-cuv)
                elif sim_method=='cos':
                    self.user_sim[u][v] = cuv / (math.pow(N[u],self.alpha)*math.pow(N[v],1-self.alpha))
                else:
                    self.user_sim[u][v] = cuv
        #print "\tCalc User Similarity Finished."

    def score(self, userid, user_actions):
        candidates = {}
        if userid in self.user_sim:
            simrank = sorted(self.user_sim[userid].items(),key=lambda x:x[1],reverse=True)
            for user,sim in simrank[:self.knn]:
                for target_song in self.u2s[user]:
                    candidates.setdefault(target_song,0.0)
                    candidates[target_song] += sim
        origin_result = sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        return origin_result

    def makeRec(self, userid, user_actions, rec_len):
        return Model.formatRecommendation(self, self.score(userid, user_actions),user_actions,rec_len)

class DeepWalk(Model):
    ''' Implement of DeepWalk '''
    def __init__(self, u2s, s2u, embeddings_file):
        Model.__init__(self, u2s, s2u, model_name='DeepWalk')
        self.model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
        self.sim_topk = 100

    def score(self, userid, user_actions):
        # str_song = []
        candidates = {}
        origin_score = {}
        embeds = []
        cnt = 0
        for song in user_actions:
            s_song = str(song)
            if s_song in self.model.vocab:
                # str_song.append(s_song)

                tmp_score = self.model.most_similar(s_song,topn=self.sim_topk)
                for target_song, score in tmp_score:
                    candidates.setdefault(target_song,0.0)
                    if score>0.5:
                        candidates[target_song] += score
                    else:
                        cnt += 1
        print 'filter', cnt
        origin_result = sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        return origin_result

    def makeRec(self, userid, user_actions, rec_len):
        return Model.formatRecommendation(self, self.score(userid, user_actions), user_actions, rec_len)

class LTR(torch.nn.Module):
    def __init__(self, D_in, H=32):
        super(LTR, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return torch.sigmoid(y_pred)

    def neg_loss(self, user_embedding, negative_embeddings):
        scores = []
        for ne in negative_embeddings:
            score = self.forward(torch.tensor(np.multiply(user_embedding , ne)))
            scores.append(score)
        return torch.mean(torch.tensor(scores))

    def loss(self, user_embeddings, user_action_embs, negatives):
        loss = 0
        for i, user_embedding in enumerate(user_embeddings):
            pos_score = self.forward(torch.tensor(np.multiply(user_embedding,user_action_embs[i])))
            neg_score = self.neg_loss(user_embedding, negatives[i])
            loss = loss + neg_score-pos_score
        return loss


class DNN(Model):
    ''' Implement of Deep Neural Network Recommender '''
    def __init__(self, u2s, s2u, embeddings_file, embedding_dim=128):
        Model.__init__(self, u2s, s2u, model_name='DNN')
        self.model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
        self.LTRmodel = LTR(embedding_dim)

    def get_embeddings(self, users):
        for u in users:
            for s in self.u2s[u]:
                continue

    def training(self, learning_rate=1e-4):
        # split training and validation 
        num_users = len(u2s)
        rand_indices = np.random.permutation(u2s.keys())
        split_pos = num_users * 0.7
        train_users = rand_indices[:split_pos]
        valid_users = rand_indices[split_pos:]

        # model training
        num_epochs = 10
        times = []
        optimizer = torch.optim.SGD(self.LTRmodel.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            start_time = time.time()
            batch_users = train_users[:16]
            user_embeddings,user_action_embs,negatives = get_embeddings(batch_users) # TODO
            loss = self.LTRmodel.loss(user_embeddings, user_action_embs, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print epoch, times[-1] ,loss.item()
            random.shuffle(train_users)
        
        # model validation


    def recall(self, userid, user_actions):
        pass

    def rank(self, user_emb, item_embs):
        pass

    def score(self, userid, user_actions):
        pass 

    def makeRec(self, userid, user_actions, rec_len):
        return Model.formatRecommendation(self, self.score(userid, user_actions), user_actions, rec_len)
