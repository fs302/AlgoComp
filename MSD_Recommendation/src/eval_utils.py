import time,heapq,math
import torch
from torch.autograd import Variable
import multiprocessing

class evaluation(object):
    def __init__(self, testRatings, testNegatives, K):
        self.testRatings = testRatings
        self.testNegatives = testNegatives
        self.K = K

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in xrange(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def eval_one_rating(self, idx):
        return (0.0, 0.0)
    
    def evaluate(self, num_thread, limit_size=0):
        _rec = {}
        if limit_size==0:
            testidx = xrange(len(self.testRatings))
        else:
            testidx = xrange(min(len(self.testRatings),limit_size))
        hits, ndcgs = [],[]
        st = time.time()
        if(num_thread > 1): # Multi-thread
            pool = multiprocessing.Pool(processes=num_thread)
            res = pool.map(self.eval_one_rating, testidx)
            pool.close()
            pool.join()
            hits = [r[0] for r in res]
            ndcgs = [r[1] for r in res]
            return (hits, ndcgs)
        # Single thread
        for idx in testidx:
            (hr,ndcg) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)  
        return (hits, ndcgs)
    
class vaecf_evaluation(evaluation):
    def __init__(self, pred, testRatings, testNegatives, K):
        super(vaecf_evaluation, self).__init__(testRatings, testNegatives, K)
        self.pred = pred
        
    def eval_one_rating(self,idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        origin_result = self.pred[u]
        for i in xrange(len(items)):
            item = items[i]
            # map_item_score[item] = np.random.random()
            map_item_score[item] = origin_result[item]
        items.pop()
        # print map_item_score

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

class itemcf_evaluation(evaluation):
    
    def __init__(self, model, u2s, testRatings, testNegatives, K):
        super(itemcf_evaluation, self).__init__(testRatings, testNegatives, K)
        self.model = model
        self.u2s = u2s
    
    def eval_one_rating(self, idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        if u in self.u2s:
            origin_result = self.model.knn_score(self.u2s[u], items)
        else:
            origin_result = {}
        for i in xrange(len(items)):
            item = items[i]
            if item in origin_result:
                map_item_score[item] = origin_result[item]
            else:
                map_item_score[item] = 0.0
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)


class usercf_evaluation(evaluation):
    
    def __init__(self, model, u2s, testRatings, testNegatives, K):
        super(usercf_evaluation, self).__init__(testRatings, testNegatives, K)
        self.model = model
        self.u2s = u2s
    
    def eval_one_rating(self, idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        if u in self.u2s:
            origin_result = dict(self.model.score(u, self.u2s[u]))
        else:
            origin_result = {}
        for i in xrange(len(items)):
            item = items[i]
            if item in origin_result:
                map_item_score[item] = origin_result[item]
            else:
                map_item_score[item] = 0.0
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)


class mf_evaluation(evaluation):
    def __init__(self, model, testRatings, testNegatives, K):
        super(mf_evaluation, self).__init__(testRatings, testNegatives, K)
        self.model = model
        
    def eval_one_rating(self,idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        gtItem = rating[1]
        items.append(gtItem)
        u = Variable(torch.LongTensor([int(rating[0])]*len(items)))
        # Get prediction scores
        map_item_score = {}
        predict_score = self.model.forward(u,Variable(torch.LongTensor(items)))
        
        for i in xrange(len(items)):
            item = items[i]
            # map_item_score[item] = np.random.random()
            map_item_score[item] = predict_score[i]
        items.pop()
        # print map_item_score

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

class svd_evaluation(evaluation):
    def __init__(self, u, s, vt, testRatings, testNegatives, K):
        super(mf_evaluation, self).__init__(testRatings, testNegatives, K)
        self.model = model
        
    def eval_one_rating(self,idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        gtItem = rating[1]
        items.append(gtItem)
        u = Variable(torch.LongTensor([int(rating[0])]*len(items)))
        # Get prediction scores
        map_item_score = {}
        predict_score = self.model.forward(u,Variable(torch.LongTensor(items)))
        
        for i in xrange(len(items)):
            item = items[i]
            # map_item_score[item] = np.random.random()
            map_item_score[item] = predict_score[i]
        items.pop()
        # print map_item_score

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)
    
class item_popularity_evaluation(evaluation):
    def __init__(self, item_degree, testRatings, testNegatives, K):
        super(item_popularity_evaluation, self).__init__(testRatings, testNegatives, K)
        self.item_degree = item_degree
        
    def eval_one_rating(self,idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        gtItem = rating[1]
        items.append(gtItem)
        
        # Get prediction scores
        map_item_score = {}
        
        for i in xrange(len(items)):
            item = items[i]
            # map_item_score[item] = np.random.random()
            map_item_score[item] = self.item_degree[int(item)]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)
