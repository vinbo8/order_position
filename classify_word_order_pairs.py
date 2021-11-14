from roberta.helpers import load_shuffled_model, load_and_invert
import argparse
import torch
from sklearn.model_selection import cross_val_score
import numpy as np
import random
from utils.rand_word_order_utils import ud_load_classify_pairwise, mean_confidence_interval
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import math
from tqdm import tqdm


def classify(args, all_examples, all_pairs, all_labels, leaveout):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.invert:
        roberta = load_and_invert(args.model_path)
    else:
        roberta = load_shuffled_model(args.model_path)

    roberta.eval()
    all_word_encodings = []
    all_word_labels = []
    all_indices = []
    all_word_tokens = []

    if 'scramble_position' in args.perturb:
        d = roberta.model.encoder.sentence_encoder.embed_positions.weight.data
        d = torch.cat((d[0:1], d[1:][torch.randperm(d.size(0) - 1)]))
        d = d[torch.randperm(d.size(0))]
        roberta.model.encoder.sentence_encoder.embed_positions.weight.data = d

    if 'norm_position' in args.perturb:
        d = roberta.model.encoder.sentence_encoder.embed_positions.weight.data
        mean = d.mean(dim=1).unsqueeze(-1).repeat(1, d.size(-1))
        std = d.std(dim=1).unsqueeze(-1).repeat(1, d.size(-1))
        roberta.model.encoder.sentence_encoder.embed_positions.weight.data = torch.normal(mean, std)

    for sent_idx, (sentence, pair_list, label_list) in tqdm(enumerate(zip(all_examples, all_pairs, all_labels))):
        assert len(label_list) == len(pair_list)
        try:
            with torch.no_grad():
                if 'bpe_nospace' in args.perturb:
                    sentence = sentence.split()
                    sent_features = [roberta.encode(i)[1:-1] for i in sentence]
                    sent_features = torch.stack([item for sublist in sent_features for item in sublist])
                    sent_features = torch.cat((torch.tensor([0]), sent_features, torch.tensor([2])))
                    sentence = " ".join(sentence)

                if 'bpe_space' in args.perturb:
                    sentence = sentence.split()
                    sent_features = [roberta.encode(f" {i}")[1:-1] for i in sentence[1:]]
                    sent_features = [roberta.encode(sentence[0])[1:-1]] + sent_features
                    sent_features = torch.stack([item for sublist in sent_features for item in sublist])
                    sent_features = torch.cat((torch.tensor([0]), sent_features, torch.tensor([2])))
                    sentence = " ".join(sentence)

                if 'only_position' in args.perturb:
                    s_len = len(roberta.encode(" ".join(sentence)))
                    sent_features = roberta.model.encoder.sentence_encoder.embed_positions.weight[:s_len]

                if 'baseline' in args.perturb:
                    sent_features = roberta.encode(sentence)

                if 'shuffle' in args.perturb:
                    sentence = sentence.split()
                    random.shuffle(sentence)
                    sent_features = roberta.encode(" ".join(sentence))
                    sentence = " ".join(sentence)

                if 'only_position' not in args.perturb:
                    sent_features = roberta.extract_features(sent_features).squeeze(0)

                fft = roberta.extract_features_aligned_to_words(str(sentence))
                for pair in pair_list:
                    pair_item1 = sent_features[pair[0]]
                    pair_item2 = sent_features[pair[1]]
                    all_word_encodings.append(torch.cat((pair_item1, pair_item2)).cpu().detach().numpy())
                    all_word_tokens.append((str(fft[pair[0]]), str(fft[pair[1]])))
                    all_indices.append((pair[0], pair[1]))

                all_word_labels.extend(label_list)
        except AssertionError:
            continue

    # make train / dev / test
    clf = LogisticRegression()
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)

    if 'leave' in args.perturb:
        z = list(zip(all_word_tokens, all_word_encodings, all_word_labels, all_indices))
        dev_size = 2000
        train_size = args.train_size
        # leaveout = tuple(leaveout)
        leaveout = [(i, j) for i in leaveout for j in leaveout if i != j]
        X_train = [j for (i, j, k, l) in z if l not in leaveout][:train_size]
        y_train = [k for (i, j, k, l) in z if l not in leaveout][:train_size]

        X_dev = [j for (i, j, k, l) in z if l in leaveout][:dev_size]
        y_dev = [k for (i, j, k, l) in z if l in leaveout][:dev_size]

        print(f"train: {len(X_train)}\tdev: {len(X_dev)}")

    else:
        vocab = []
        dev_size = len(all_word_tokens) // 5
        X_dev, y_dev = [], []
        X_train, y_train = [], []
        for words, enc, label in zip(all_word_tokens, all_word_encodings, all_word_labels):
            if len(y_dev) < dev_size:
                X_dev.append(enc)
                y_dev.append(label)
                vocab.append(words[0])
                vocab.append(words[1])
            else:
                break

        vocab = set(vocab)
        for words, enc, label in zip(all_word_tokens[dev_size:], all_word_encodings[dev_size:], all_word_labels[dev_size:]):
            if words[0] not in vocab and words[1] not in vocab:
                X_train.append(enc)
                y_train.append(label)

        X_train = np.vstack(X_train)
        X_dev = np.vstack(X_dev)

    dummy.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    print(f"{dummy.score(X_dev, y_dev)}\t{clf.score(X_dev, y_dev)}")
    return clf.score(X_dev, y_dev)

    # X, y = np.vstack(all_word_encodings), all_word_labels
    # scores = cross_val_score(clf, X, y, cv=5)
    # dummy_scores = cross_val_score(dummy, X, y, cv=5)
    # print(f"{np.mean(scores)} ± {np.std(scores)}; dummy: {np.mean(dummy_scores)}")
    # return np.mean(scores)


def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus")
    parser.add_argument('-d', "--dataset_path", type=str)
    parser.add_argument('-m', "--model_path", type=str)
    parser.add_argument('-t', "--train_size", type=int, default=10000)
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10)
    parser.add_argument('-i', "--invert", action='store_true')
    parser.add_argument('-s', "--no_samples", type=int, default=5)
    parser.add_argument('-r', "--no_runs", type=int, default=3)
    parser.add_argument('-p', "--perturb", action='store')
    arguments = parser.parse_args()

    print(arguments.model_path, ' :model')
    dataset_file = open(arguments.dataset_path, 'r').read()

    acc_list = []
    for _ in range(arguments.no_runs):
        all_examples, all_labels, all_pairs, leaveout = ud_load_classify_pairwise(
            arguments, dataset_file, arguments.max_sentence_len, arguments.no_samples
        )
        acc = classify(arguments, all_examples, all_pairs, all_labels, leaveout)
        acc_list.append(acc)

    print(acc_list)
    print(np.mean(acc_list))
    # acc_mean, acc_lower_conf_int, acc_upper_conf_int = mean_confidence_interval(acc_list)
    # print("acc avg: {}, acc lower conf: {}, acc upper conf: {}".format(
    #     acc_mean, acc_lower_conf_int, acc_upper_conf_int))


if __name__ == '__main__':
    main()




