import argparse
from collections import defaultdict
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ------------------------------------- functions --------------------------------------

def intercos(y, keys=None, center=False, sample_size=-1):
    if center: y = StandardScaler(with_std=False).fit(y).transform(y)
    if sample_size >= 0:
        assert(keys is not None)
        assert(sample_size == int(sample_size))
        all_y = []
        for k in np.unique(keys):
            y_k = y[keys==k]
            if y_k.shape[0] < 1: continue
            y_k = y_k[np.random.choice(y_k.shape[0], sample_size, replace=False)]
            all_y.append(y_k)
        y = np.vstack(all_y)
    cos = cosine_similarity(y, y)
    avg_cos = ( np.sum(np.sum(cos)) - cos.shape[0] ) / 2 / ( cos.shape[0]*(cos.shape[0]-1) / 2 )
    return avg_cos

def intracos(y, d, keys, special_dict=None, center=False):
    if center: y = StandardScaler(with_std=False).fit(y).transform(y)
    all_cos = []
    for k in d:
        if special_dict and k in special_dict.values():
            continue
        tmp = y[keys==k]
        if tmp.shape[0] <= 1:
            continue
        if tmp.shape[0] >= 1000: # maximum 1000 points to estimate
            idx = np.random.choice(len(tmp), 1000, replace=False)
            tmp = tmp[idx]
        avg_cos = intercos(tmp)
        all_cos.append(avg_cos)
    avg_cos = np.mean(all_cos)
    return avg_cos

def cai_analysis(d, args):
    """
    run intracos and intercos analyses adapted from:
    https://github.com/TideDancer/IsotropyContxt/blob/main/analysis.py
    """

    keys = []
    y = []
    x = []
    repeats = []
    window_ids = []
    pos_ids = []

    # handle length
    lengths = {}
    for k in d:
        lengths[k] = len(d[k])
        if args.no_cntx_limit >= 1:
            lengths[k] = min(len(d[k]), int(args.no_cntx_limit))  # maxl
        if args.no_cntx_limit > 0 and args.no_cntx_limit < 1:
            lengths[k] = max(1, int(len(d[k]) * args.no_cntx_limit))  # fraction
        if args.no_cntx_limit == -1:
            lengths[k] = max(1, int(np.log2(len(d[k]))))  # log
        if args.no_cntx_limit == -2:
            lengths[k] = int(np.sqrt(len(d[k])))  # sqrt

    # proc data
    for k in d:
        keys += [k] * lengths[k]

        y_tmp = list(map(lambda e: e[0], d[k]))
        if lengths[k] < len(y_tmp):
            idx = np.random.choice(len(y_tmp), lengths[k], replace=False)  # sample without replacement
            y_tmp = [y_tmp[i] for i in idx]
        y += y_tmp

        repeats += [len(d[k])] * lengths[k]
        window_ids += list(map(lambda e: e[1], d[k]))
        pos_ids += list(map(lambda e: e[1] * 512 + e[2], d[k]))

    keys = np.stack(keys)
    window_ids = np.stack(window_ids)
    pos_ids = np.stack(pos_ids)
    y = np.stack(y)
    repeats = np.stack(repeats)
    print('total points: ', y.shape[0])

    ## embed
    var_ratio = 1
    if args.pca > 0:
        pca = PCA(n_components=args.pca).fit(y)
        y = pca.transform(y)
        var_ratio = sum(pca.explained_variance_ratio_)
        print('after PCA:', y.shape)

    ## centered
    if args.center:
        y = StandardScaler(with_std=False).fit(y).transform(y)

    # inter cos
    assert (args.no_cntx_limit > 0)
    avg_cos = intercos(y, args.center)
    print("inter cos: ", avg_cos)

    special_dict = None
    # intra cosine
    avg_cos = intracos(y, d, keys, special_dict, args.center)
    print("intra cos: ", avg_cos)

def main():
    parser = argparse.ArgumentParser(description="analyse token embeds");
    parser.add_argument('-e', "--embeds_path", type=str, default='ptb');
    parser.add_argument('-a', "--analysis_type", type=str, default='cai_2021');
    parser.add_argument('-no', "--no_cntx_limit", type=int, default=100);
    parser.add_argument('-p', "--pca", type=int, default=0);
    parser.add_argument('-c', "--center",  action='store', default=False);
    arguments = parser.parse_args();

    # load embeds
    embeds_dict = json.load((open(arguments.embeds_path, 'r')))

    #run analysis
    cai_analysis(embeds_dict, arguments)


if __name__ == '__main__':
    main();




