from roberta.helpers import load_shuffled_model
from collections import defaultdict
import torch.nn.functional as F
import argparse
import torch
from statistics import mean
from fairseq.data import Dictionary
import numpy as np
import random
from utils.rand_word_order_utils import ud_load_regress
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import math

def classify(args, all_examples, all_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model (weights) and extract reps
    roberta = load_shuffled_model(args.model_path)
    roberta.eval()
    all_word_encodings = []
    all_word_labels = []

    for sent_idx, (sentence, label_list) in enumerate(zip(all_examples, all_labels)):
        try:
            with torch.no_grad():
                sent_features = roberta.extract_features_aligned_to_words(str(sentence))
                assert len(sent_features) - 2 == len(label_list)
                for tok in sent_features[1:-1]:
                    all_word_encodings.append(tok.vector.cpu().detach().numpy())
                all_word_labels.extend(label_list)
        except:
            print(sentence, ' :mis-aligned feats')

    # make train / dev / test
    dev_size = math.ceil(len(all_word_encodings) / 6)
    if not args.hold_out_words:
        train_features, train_labels = np.vstack(all_word_encodings[:-dev_size]), all_word_labels[:-dev_size]
        dev_features, dev_labels = np.vstack(all_word_encodings[-dev_size:]), all_word_labels[-dev_size:]
   # else:
        #for _ in range(dev_size):

    # print stats
    #print("O-train: {} P-train: {}  O-dev: {} P-dev: {} !".format(o_count_train, p_count_train, o_count_dev, p_count_dev))

    #train and eval
    print(train_features.shape, "train_features ")
    clf = LinearRegression().fit(train_features, train_labels)
    #clf.predict(dev_features)
    acc = clf.score(dev_features, dev_labels)
    print(acc, ": acc")

    #return mean_ppl, all_sent_ppl

def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus");
    parser.add_argument('-d', "--dataset_path", type=str);
    parser.add_argument('-m', "--model_path", type=str);
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10);
    parser.add_argument('-p', "--no_perms", type=int, default=1);
    parser.add_argument('-s', "--shuffle_bpe", action='store_true', default=False);
    parser.add_argument('-hw', "--hold_out_words", action='store_true', default=False);

    arguments = parser.parse_args();

    #model
    print(arguments.model_path, ' :model')
    # load dataset
    dataset_file = open(arguments.dataset_path, 'r').read()
    # pass to permute function, returns list of lists where inner list is of all perms per sentence
    all_examples, all_labels, leven_distances_to_orig, bleu_to_orig = ud_load_regress(dataset_file,
        sentence_len_limit=arguments.max_sentence_len, permutation_no=arguments.no_perms)
    print(len(all_examples), ' no examples')
    #shuffle exmaples to mix permed and non
    all_examples, all_labels = shuffle(np.array(all_examples), np.array(all_labels))
    #classify
    _ = classify(arguments, all_examples, all_labels)
    #compute correlation between ppl and levenstein distance
    #corr = spearmanr(all_sent_ppl, leven_distances_to_orig)
    #print(corr, " :correlation of perplexity to leven distance to orig order.")
    #compute correlation between ppl and bleu-4
    #corr = spearmanr(all_sent_ppl, bleu_to_orig)
    #print(corr, " :correlation of perplexity to bleu to orig order.")


if __name__ == '__main__':
    main();




