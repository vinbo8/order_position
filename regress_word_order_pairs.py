from roberta.helpers import load_shuffled_model
from collections import defaultdict
import torch.nn.functional as F
import argparse
import torch
from statistics import mean
from fairseq.data import Dictionary
import numpy as np
import random
from utils.rand_word_order_utils import ud_load_regress, mean_confidence_interval, ud_load_regress_pairwise
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.utils import shuffle
import math

def classify(args, all_examples, all_pairs, all_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model (weights) and extract reps
    roberta = load_shuffled_model(args.model_path)
    roberta.eval()
    all_word_encodings = []
    all_word_labels = []
    all_words = []
    for sent_idx, (sentence, pair_list, label_list) in enumerate(zip(all_examples, all_pairs, all_labels)):
        assert len(label_list) == len(pair_list)
        try:
            with torch.no_grad():
                sent_features = roberta.extract_features_aligned_to_words(str(sentence))
                #assert len(sent_features) - 2 == len(label_list)
                for pair in pair_list:
                    pair_item1 = sent_features[pair[0]]
                    pair_item2 = sent_features[pair[1]]
                    all_word_encodings.append(torch.cat((pair_item1.vector, pair_item2.vector)).cpu().detach().numpy())
                    all_words.append((str(pair_item1),str(pair_item2)))
                all_word_labels.extend(label_list)
        except:
           continue
            #print(sentence, ' :mis-aligned feats')

    # make train / dev / test
    dev_size = math.ceil(len(all_word_encodings) / 6)
    if args.control:
        random.shuffle(all_word_labels)
    if not args.hold_out_words:
        print(len(all_word_encodings), " all_word_encodings")
        print(len(all_word_labels), "all_word_labels")
        train_features, train_labels = np.vstack(all_word_encodings[:-dev_size]), all_word_labels[:-dev_size]
        dev_features, dev_labels = np.vstack(all_word_encodings[-dev_size:]), all_word_labels[-dev_size:]
    elif args.hold_out_words:
        dev_vocab = []
        dev_encodings_list = []
        dev_labels = []
        train_encodings_list = []
        train_labels = []
        for words, enc, label in zip(all_words, all_word_encodings, all_word_labels):
            if len(dev_labels) < dev_size:
                dev_encodings_list.append(enc)
                dev_vocab.append(words[0])
                dev_vocab.append(words[1])
                dev_labels.append(label)
            else:
                break
        dev_vocab = list(set(dev_vocab))
        for word, enc, label in zip(all_words[dev_size:], all_word_encodings[dev_size:], all_word_labels[dev_size:]):
            if word not in dev_vocab:
                train_encodings_list.append(enc)
                train_labels.append(label)
        train_features =  np.vstack(train_encodings_list)
        dev_features = np.vstack(dev_encodings_list)
    # print stats
    #print("O-train: {} P-train: {}  O-dev: {} P-dev: {} !".format(o_count_train, p_count_train, o_count_dev, p_count_dev))

    #train and eval
    print(train_features.shape, "train_features ")
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5).fit(train_features, train_labels)
    #clf = LinearRegression().fit(train_features, train_labels)
    r2 = clf.score(dev_features, dev_labels)
    print(r2, ": r2")
    preds = clf.predict(dev_features)
    avg_dif_abs = np.mean(np.abs(preds-dev_labels))
    print(avg_dif_abs, ": avg_dif_abs")
    vg_dif_squared = np.mean(np.square(preds - dev_labels))
    print(vg_dif_squared, ": vg_dif_squared")
    return r2, avg_dif_abs, vg_dif_squared


def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus")
    parser.add_argument('-d', "--dataset_path", type=str)
    parser.add_argument('-m', "--model_path", type=str)
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10)
    parser.add_argument('-sm', "--no_samples", type=int, default=5)
    parser.add_argument('-ss', "--shuffle_sents", action='store_true', default=False)
    parser.add_argument('-sb', "--shuffle_bpe", action='store_true', default=False)
    parser.add_argument('-hw', "--hold_out_words", action='store_true', default=False)
    parser.add_argument('-c', "--control", action='store_true', default=False)
    parser.add_argument('-r', "--no_runs", type=int, default=4)
    arguments = parser.parse_args()

    print(arguments.model_path, ' :model')
    dataset_file = open(arguments.dataset_path, 'r').read()

    r2_list, avg_dif_abs_list, vg_dif_squared_list = [],[],[]
    for _ in range(arguments.no_runs):
        all_examples, all_labels, all_pairs = ud_load_regress_pairwise(
            dataset_file, arguments.max_sentence_len, arguments.shuffle_sents, sample_no=arguments.no_samples
        )
        print(f'loaded {len(all_examples)} examples')
        r2, avg_dif_abs, vg_dif_squared = classify(arguments, all_examples, all_pairs, all_labels)
        r2_list.append(r2)
        avg_dif_abs_list.append(avg_dif_abs)
        vg_dif_squared_list.append(vg_dif_squared)

    r2_mean, r2_lower_conf_int, r2_upper_conf_int = mean_confidence_interval(r2_list)
    avg_dif_abs_mean, avg_dif_abs_lower_conf_int, avg_dif_abs_upper_conf_int = mean_confidence_interval(avg_dif_abs_list)
    vg_dif_squared_mean, vg_dif_squared_lower_conf_int, vg_dif_squared_upper_conf_int = mean_confidence_interval(vg_dif_squared_list)
    print("r2 avg: {}, r2 lower conf: {}, r2 upper conf: {}".format(
        r2_mean, r2_lower_conf_int, r2_upper_conf_int))
    print("abs error avg: {}, abs error lower conf: {}, abs error upper conf: {}".format(
        avg_dif_abs_mean, avg_dif_abs_lower_conf_int, avg_dif_abs_upper_conf_int))
    print("sq. error avg: {}, sq error lower conf: {}, sq error upper conf: {}".format(
        vg_dif_squared_mean, vg_dif_squared_lower_conf_int, vg_dif_squared_upper_conf_int))



if __name__ == '__main__':
    main();




