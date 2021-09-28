from roberta.helpers import load_shuffled_model
import argparse
import torch
from sklearn.model_selection import cross_val_score
import numpy as np
import random
from utils.rand_word_order_utils import ud_load_classify_pairwise, mean_confidence_interval
from sklearn.linear_model import LogisticRegression
import math
from tqdm import tqdm


def classify(args, all_examples, all_pairs, all_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roberta = load_shuffled_model(args.model_path)
    roberta.eval()
    all_word_encodings = []
    all_word_labels = []
    all_words = []
    for sent_idx, (sentence, pair_list, label_list) in tqdm(enumerate(zip(all_examples, all_pairs, all_labels))):
        assert len(label_list) == len(pair_list)
        try:
            with torch.no_grad():
                sentence = str(sentence).split()
                if sent_idx >= len(all_examples) // 5:
                    random.shuffle(sentence)
                sentence = " ".join(sentence)
                sent_features = roberta.extract_features_aligned_to_words(str(sentence))
                for pair in pair_list:
                    pair_item1 = sent_features[pair[0]]
                    pair_item2 = sent_features[pair[1]]
                    all_word_encodings.append(torch.cat((pair_item1.vector, pair_item2.vector)).cpu().detach().numpy())
                    all_words.append((str(pair_item1), str(pair_item2)))
                all_word_labels.extend(label_list)
        except AssertionError:
            continue

    # make train / dev / test
    train_X = np.vstack(all_word_encodings[len(all_examples) // 5:])
    val_X = np.vstack(all_word_encodings[:len(all_examples) // 5])
    train_y = all_word_labels[len(all_examples) // 5:]
    val_y = all_word_labels[:len(all_examples) // 5]
    clf = LogisticRegression(random_state=42).fit(train_X, train_y)
    print(clf.score(val_X, val_y))
    # X, y = np.vstack(all_word_encodings), all_word_labels
    # scores = cross_val_score(clf, X, y, cv=5)
    # print(scores)
    # print(f"{np.mean(scores)} ± {np.std(scores)}")
    # return np.mean(scores)

def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus")
    parser.add_argument('-d', "--dataset_path", type=str)
    parser.add_argument('-m', "--model_path", type=str)
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10)
    parser.add_argument('-s', "--no_samples", type=int, default=5)
    parser.add_argument('-r', "--no_runs", type=int, default=4)
    arguments = parser.parse_args()

    print(arguments.model_path, ' :model')
    dataset_file = open(arguments.dataset_path, 'r').read()

    # acc_list = []
    # for _ in range(arguments.no_runs):
    all_examples, all_labels, all_pairs = ud_load_classify_pairwise(
        dataset_file, arguments.max_sentence_len, arguments.no_samples
    )
    print(len(all_examples), ' no examples')
    classify(arguments, all_examples, all_pairs, all_labels)
    # acc_list.append(acc)
    # acc_mean, acc_lower_conf_int, acc_upper_conf_int = mean_confidence_interval(acc_list)
    # print("acc avg: {}, acc lower conf: {}, acc upper conf: {}".format(
    #     acc_mean, acc_lower_conf_int, acc_upper_conf_int))


if __name__ == '__main__':
    main()




