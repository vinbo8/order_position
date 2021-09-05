import argparse
from collections import defaultdict
import pickle as p
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from utils.analyse_token_reps_utils import cai_analysis

def main():
    parser = argparse.ArgumentParser(description="analyse token embeds");
    parser.add_argument('-e', "--embeds_path", type=str, default='ptb');
    parser.add_argument('-a', "--analysis_type", type=str, default='cai_2021');
    parser.add_argument('-no', "--no_cntx_limit", type=int, default=100);
    parser.add_argument('-ns', "--no_samples_limit", type=int, default=20000);
    parser.add_argument('-p', "--pca", type=int, default=0);
    parser.add_argument('-c', "--center",  action='store', default=False);
    parser.add_argument('--lid', action='store_true', default=False, help='task for compute lid')
    parser.add_argument('--lid_metric', type=str, default='l2', help='metric for lid, choose from l2, cos')
    parser.add_argument('--cluster', action='store_true', default=False, help='task for clustering')
    parser.add_argument('--draw', type=str, default=None, help='draw, choose from 2d, 3d, freq')

    arguments = parser.parse_args();

    # load embeds
    embeds_dict = p.load((open(arguments.embeds_path, 'rb')))

    #run analysis
    cai_analysis({k: embeds_dict[k] for k in embeds_dict.keys()[:arguments.no_samples_limit]}, arguments)

if __name__ == '__main__':
    main();




