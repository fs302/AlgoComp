# Using models to make recomendation
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys,utils,models,time

if __name__ == "__main__":
    parser = ArgumentParser("rec",
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--user-min',default=0,type=int,
        help="Min index of rec user.")
    parser.add_argument('--user-max',default=100,type=int,
        help="Max index of rec user.")
    parser.add_argument('--output',default='rec_result.txt',
        help="Output Result File.")
    parser.add_argument('--model', default='ItemCF',
        help="Recommend Model")
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
    sys.stdout.flush()

    # Make Recommendations
    if args.model == 'ItemCF':
        model = models.ItemCF(u2s, s2u, sim_method=args.sim, alpha=args.alpha, Q=args.q)
    if args.model == 'UserCF':
        model = models.UserCF(u2s, s2u, sim_method=args.sim, alpha=args.alpha, knn=args.knn)

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