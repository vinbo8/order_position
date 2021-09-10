import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


def lid(y, sample_size=-1, k_list=[101], metric='l2', block=50000):
    import faiss
    print('metric:', metric)
    ngpus = faiss.get_num_gpus()
    print("number of GPUs used by faiss:", ngpus)
    if metric == 'cos':
        cpu_index = faiss.IndexFlatIP(y.shape[1])
        y = normalize(y)
    if metric == 'l2':
        cpu_index = faiss.IndexFlatL2(y.shape[1])

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    print('index')
    gpu_index.add(y)
    print('index total:', gpu_index.ntotal)

    if sample_size > 0:
        x = y[np.random.choice(y.shape[0], size=int(sample_size), replace=False)]
    else:
        x = y

    for k in k_list:
        print('start query')
        i = 0
        D = []
        while i < x.shape[0]:
            tmp = x[i:min(i+block, x.shape[0])]
            i += block
            b, _ = gpu_index.search(tmp, k)
            D.append(b)
        D = np.vstack(D)
        print("query finish")

        D = D[:, 1:] # remove the most-left column as it is itself
        if metric == 'cos':
            D = 1-D  # cosine dist = 1 - cosine
            D[D <= 0] = 1e-8
        rk = np.max(D, axis=1)
        rk[rk==0] = 1e-8
        lids = D/rk[:, None]
        lids = -1/np.mean(np.log(lids), axis=1)
        lids[np.isinf(lids)] = y.shape[1] # if inf, set as space dimension
        lids = lids[~np.isnan(lids)] # filter nan
        print('filter nan/inf shape', lids.shape)
        print('k', k-1, 'lid_mean', np.mean(lids), 'lid_std', np.std(lids))

# --------------------------------analyses-------------------------------------------------
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

    save_prefix = args.embeds_path.split('/')[-1] # format: dataset.dictfile

    if args.pca > 0:
        if args.pca > 1: args.pca = int(args.pca)
        save_prefix += '.pca.' + str(args.pca) + '.'
    if args.no_cntx_limit > 0:
        save_prefix += '.no_cntx_limit.' + str(args.no_cntx_limit) + '.'
    if args.no_cntx_limit == -1:
        save_prefix += '.no_cntx_limit.log2.'
    if args.no_cntx_limit == -2:
        save_prefix += '.no_cntx_limit.sqrt.'

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

        y_tmp = list(map(lambda e: e, d[k]))
        if lengths[k] < len(y_tmp):
            idx = np.random.choice(len(y_tmp), lengths[k], replace=False)  # sample without replacement
            y_tmp = [y_tmp[i] for i in idx]
        y += y_tmp

        repeats += [len(d[k])] * lengths[k]
        #window_ids += list(map(lambda e: e[1], d[k]))
        #pos_ids += list(map(lambda e: e[1] * 512 + e[2], d[k]))
    keys = np.stack(keys)
    #window_ids = np.stack(window_ids)
    #pos_ids = np.stack(pos_ids)
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

    # compute lid
    k_list = [101]
    if args.lid:
        lid(y, k_list=k_list, metric=args.lid_metric)

    # cluster
    if args.cluster:
        print('perform clustering on ', y.shape)
        # kmeans with silhouette score
        sil = []
        sil_std = []
        all_labels = []
        cands = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for k in cands:
            score = []
            kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(y)
            labels = kmeans.labels_
            score.append(silhouette_score(y, labels, sample_size=20000))
            sil.append(np.mean(score))
            sil_std.append(np.std(score))
            all_labels.append(labels)
            print("k&score&std:", k, sil[-1], sil_std[-1])
        if max(sil) >= 0.1:
            best_k = cands[sil.index(max(sil))]
            labels = all_labels[sil.index(max(sil))]
            std = sil_std[sil.index(max(sil))]
        else:
            best_k = 1
            labels = np.zeros(y.shape[0])
            std = 0
        print('bestk&sil&std:', best_k, max(sil), std)

    # draw
    if args.draw:
        if args.draw == '3d' or args.draw == 'token':
            pca = PCA(n_components=3).fit(y)
        else:
            pca = PCA(n_components=2).fit(y)

        save_prefix += '.' + args.draw + '.'

        y = pca.transform(y)
        var_ratio = sum(pca.explained_variance_ratio_)

        print('draw')
        fig = plt.figure(figsize=(4, 3))
        if args.draw == '2d':
            plt.scatter(y[:, 0], y[:, 1], s=1, alpha=0.3, marker='.')
        if args.draw == '3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=0.1, alpha=0.3, marker='.')
        if args.draw == 'freq':
            # plt.tricontourf(y[:,0], y[:,1], repeats, levels=15, cmap="RdBu_r")#, linewidths=0.5, colors='k')
            plt.scatter(y[:, 0], y[:, 1], c=repeats, cmap='jet', s=0.1, alpha=0.1, marker='.')
            plt.colorbar()
        if args.draw == 'token':
            colors = ['k', 'r', 'g', 'm', 'b']
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=0.05, alpha=0.01, marker='.', color='y')
            tokens_list = eval(args.draw_token)
            print('tokens to draw: ', tokens_list)
            for i in range(len(tokens_list)):
                t = y[keys == tokens_list[i]]
                print(tokens_list[i], ' occurrence: ', len(t))
                ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=1, alpha=1, marker='o', color=colors[i % 5],
                           label=tokens_list[i])
            legend = ax.legend(markerscale=6)

        plt.title('var ratio r=%.3f' % var_ratio)
        save_prefix = 'images/' + save_prefix
        print('save as:', save_prefix)
        plt.savefig(save_prefix + 'png', format='png')
        plt.close()






