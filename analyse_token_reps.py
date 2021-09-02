import argparse
from collections import defaultdict
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from utils.analyse_token_reps_utils import cai_analysis

def main():
    parser = argparse.ArgumentParser(description="analyse token embeds");
    parser.add_argument('-e', "--embeds_path", type=str, deault='ptb');
    parser.add_argument('-a', "--analysis_type", type=str, default='cai_2021');
    parser.add_argument('-no', "--no_cntx_limit", type=int, default=100);
    parser.add_argument('-p', "--pca", type=int, default=0);
    parser.add_argument('-c', "--center",  action='store', dault=False);
    parser.add_argument('--lid', action='store_true', default=False, help='task for compute lid')
    parser.add_argument('--lid_metric', type=str, default='l2', help='metric for lid, choose from l2, cos')
    parser.add_argument('--cluster', action='store_true', default=False, help='task for clustering')
    parser.add_argument('--draw', type=str, default=None, help='draw, choose from 2d, 3d, freq')

    arguments = parser.parse_args();

    # load embeds
    embeds_dict = json.load((open(arguments.embeds_path, 'wb')))

    #run analysis
    cai_analysis(embeds_dict, arguments)

if __name__ == '__main__':
    main();




