import models
import heapq
import math
import multiprocessing
import numpy as np
import argparse
import time


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

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}

    if u in u2s:
        origin_result = _model.knn_score(u2s[u], items)
    else:
        origin_result = {}
    for i in xrange(len(items)):
        item = items[i]
        # map_item_score[item] = np.random.random()
        if item in origin_result:
            map_item_score[item] = origin_result[item]
        else:
            map_item_score[item] = 0.0
    items.pop()
    # print map_item_score
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def evaluate(model, testRatings, testNegatives, K, num_thread, limit_size):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _rec
    _rec = {}
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    hits, ndcgs = [],[]
    st = time.time()
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in xrange(min(len(_testRatings),limit_size)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)  
        if idx % 1000 == 0:
            print idx, hr, ndcg, time.time()-st
    return (hits, ndcgs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--dataset', nargs='?', default='msd-middle',
                        help='Input data.')
    parser.add_argument('--num_thread', type=int, default=1,
                        help='Number of thread.')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of test.')
    parser.add_argument('--knn', type=int, default=10,
                        help='Number of Neighbors.')
    parser.add_argument('--topk', type=int, default=10,
                        help='topk.')
    args = parser.parse_args()
    dataset_name = args.dataset
    train_file = '../train/%s.train.rating' % dataset_name

    u2s = dict()
    s2u = dict()
    print 'reading training file.'
    with open(train_file, 'r') as f:
        for line in f:
            if line != None and line != "":
                arr = line.split("\t")
                user, song = int(arr[0]), int(arr[1])
                if user not in u2s:
                    u2s[user] = set()
                u2s[user].add(song)
                if song not in s2u:
                    s2u[song] = set()
                s2u[song].add(user)


    print 'constructing model.'
    model = models.ItemCF(u2s, s2u, sim_method='cos', alpha=0.5, Q=1, knn=args.knn)


    print 'reading testing file.'
    testRatings =  load_rating_file_as_list("../train/%s.test.rating" % dataset_name)
    testNegatives = load_negative_file("../train/%s.test.negative" % dataset_name)

    print 'evaluating, test size:',len(testRatings),', limit:', args.limit
    (hits, ndcgs) = evaluate(model, testRatings, testNegatives, K=args.topk, num_thread=args.num_thread, limit_size=args.limit)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('HR = %.4f, NDCG = %.4f'  % (hr, ndcg))

