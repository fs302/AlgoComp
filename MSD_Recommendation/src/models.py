# Models for training
import math

class Model(object):
    def __init__(self,u2s, s2u, model_name):
        self.model_name=model_name
        print 'Constructing '+self.model_name
        self.song2user=s2u
        self.user2song=u2s
        self.pop_songs = []
        self.get_pop_songs()

    def get_pop_songs(self):
        ranklist = sorted(self.song2user.items(), key=lambda x:len(x[1]), reverse=True)
        self.pop_songs = [x[0] for x in ranklist]
        print "\tRanking Pop Songs. tot:",len(self.pop_songs)

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
    def __init__(self, u2s, s2u, sim_method='jaccard', alpha=1.0, Q=3):
        Model.__init__(self, u2s, s2u, model_name='ItemCF_'+sim_method)
        self.alpha = alpha
        self.Q = Q
        self.song_sim = {}
        self.cal_item_sim(sim_method)

    def cal_item_sim(self, sim_method='jaccard'):
        print "\tCalc Item Similarity:"+sim_method+',alpha='+str(self.alpha)+",Q="+str(self.Q)
        C = dict()
        N = dict()
        for u, songs in self.user2song.items():
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
        print "\tCalc User Similarity:"+sim_method+',alpha='+str(self.alpha)
        C = dict()
        N = dict()
        cnt = 0
        for s, users in self.song2user.items():
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
        print "\tCalc User Similarity Finished."

    def score(self, userid, user_actions):
        candidates = {}
        if userid in self.user_sim:
            simrank = sorted(self.user_sim[userid].items(),key=lambda x:x[1],reverse=True)
            for user,sim in simrank[:knn]:
                for target_song in self.user2song[user]:
                    candidates.setdefault(target_song,0.0)
                    candidates[target_song] += sim
        origin_result = sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        return origin_result

    def makeRec(self, userid, user_actions, rec_len):
        return Model.formatRecommendation(self, self.score(userid, user_actions),user_actions,rec_len)
