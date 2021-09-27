from roberta.helpers import load_shuffled_model
import tqdm
from collections import defaultdict
import torch.nn.functional as F
import argparse
import torch
from statistics import mean
from fairseq.data import Dictionary
import numpy as np
from string import punctuation
import random
from utils.rand_word_order_utils import ud_load_classify
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import math


def classify(args, all_examples, all_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roberta = load_shuffled_model(args.model_path)
    roberta.eval()

    all_sent_encodings = []
    for sent_idx, (sentence, label) in tqdm.tqdm(enumerate(zip(all_examples, all_labels))):
        with torch.no_grad():
            if args.shuffle_mode == 'bpe':
                tokens = roberta.encode(sentence)
                #drop start and end tok 0 and 2, shuffle, then add them
                idx = torch.randperm(tokens.nelement()-2)
                tokens = tokens[1:-1]
                tokens = tokens.view(-1)[idx].view(tokens.size())
                tokens = torch.cat((torch.tensor([0]), tokens, (torch.tensor([2]))))
            elif args.shuffle_mode == 'safe_tokens':
                sentence = sentence.split()
                split_with_spaces = [i for i in sentence]
                tokens = [roberta.encode(i)[1:-1] for i in split_with_spaces]
                random.shuffle(tokens)
                tokens = [item for sublist in tokens for item in sublist]
                tokens = torch.stack(tokens)
                tokens = torch.cat((torch.tensor([0]), tokens, torch.tensor([2])))

            features = roberta.extract_features(tokens)
            features = features.squeeze(0).mean(dim=0)
            all_sent_encodings.append(features.cpu().detach().numpy())

    # make train / dev / test
    dev_size = len(all_sent_encodings) // 6
    if not args.hold_out_words:
        train_features, train_labels = np.vstack(all_sent_encodings[:-dev_size]), all_labels[:-dev_size]
        dev_features, dev_labels = np.vstack(all_sent_encodings[-dev_size:]), all_labels[-dev_size:]

    # print stats
    o_count_train = len([l for l in train_labels if l == 'o'])
    p_count_train = len([l for l in train_labels if l == 'p'])
    o_count_dev = len([l for l in dev_labels if l == 'o'])
    p_count_dev = len([l for l in dev_labels if l == 'p'])
    print("O-train: {} P-train: {}  O-dev: {} P-dev: {} !".format(o_count_train, p_count_train, o_count_dev, p_count_dev))

    # train and eval
    clf = LogisticRegression(random_state=42).fit(train_features, train_labels)
    acc = clf.score(dev_features, dev_labels)
    print(acc, ": acc")


def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus")
    parser.add_argument('-d', "--dataset_path", type=str)
    parser.add_argument('-m', "--model_path", type=str)
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10)
    parser.add_argument('-p', "--no_perms", type=int, default=1)
    parser.add_argument("--shuffle_mode", action='store', default='tokens', options=['tokens', 'bpe', 'safe_tokens'])
    parser.add_argument('-hw', "--hold_out_words", action='store_true', default=False)
    arguments = parser.parse_args()

    dataset_file = open(arguments.dataset_path, 'r').read()
    all_examples, all_labels = ud_load_classify(dataset_file, sentence_len_limit=arguments.max_sentence_len)
    print(f'read {len(all_examples)} examples')

    all_examples, all_labels = shuffle(np.array(all_examples), np.array(all_labels))
    classify(arguments, all_examples, all_labels)

    # compute correlation between ppl and levenstein distance
    # corr = spearmanr(all_sent_ppl, leven_distances_to_orig)
    # print(corr, " :correlation of perplexity to leven distance to orig order.")
    # compute correlation between ppl and bleu-4
    # corr = spearmanr(all_sent_ppl, bleu_to_orig)
    # print(corr, " :correlation of perplexity to bleu to orig order.")


if __name__ == '__main__':
    main()




