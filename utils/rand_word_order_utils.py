from conllu import parse
import itertools
import random
from Levenshtein import distance as levenshtein_distance
import nltk
import math
import scipy
import numpy as np
import copy

# ------------------------------------- functions --------------------------------------

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def ud_permute(ud_data, sentence_len_limit=None, no_sentences=None,
               shuffle_level='n1', permutation_type='linear', permutation_limit=None):
    # parse data
    sentences = parse(ud_data)
    # prep
    all_permuted_sentences = []
    leven_distances_to_orig = []
    bleu_to_orig = []

    # iterate over all sentences
    for sentence in sentences:
        #limit sentence len and no sents
        if len(sentence) < sentence_len_limit and len(sentence) > 4 and len(all_permuted_sentences) < no_sentences:
            # just randomly shuffle all toks irrespective of heirarch. structure
            if permutation_type == 'linear':
                # sent to list of tokens
                token_list = [t['form'] for t in sentence]
                # permute
                permutation_list = list(itertools.permutations(token_list))
                # if sub-sampling on
                if permutation_limit:
                    permutation_list = random.sample(permutation_list, permutation_limit)
                all_permuted_sentences.append(permutation_list)
                # compute leven distaces
                for permutation in permutation_list:
                    ld = levenshtein_distance(' '.join(token_list), ' '.join(permutation))
                    leven_distances_to_orig.append(ld)
                    bs = nltk.translate.bleu_score.sentence_bleu(' '.join(token_list), ' '.join(permutation))
                    bleu_to_orig.append(bs)
    return all_permuted_sentences, leven_distances_to_orig, bleu_to_orig


def ud_load_classify(ud_data, sentence_len_limit=None):
    # parse data
    sentences = parse(ud_data)
    all_examples = []
    all_labels = []
    max_sents = len([s for s in sentences if 3 < len(s) < sentence_len_limit])
    random.shuffle(sentences)
    for sentence in sentences:
        if 3 < len(sentence) < sentence_len_limit:
            token_list = [t['form'] for t in sentence]
            if len(all_examples) < max_sents // 2:
                all_examples.append(' '.join(token_list))
                all_labels.append('p')
            else:
                all_examples.append(' '.join(token_list))
                all_labels.append('o')

    return all_examples, all_labels



def ud_load_regress(ud_data, sentence_len_limit=None,
               shuffle_level=False, permutation_type='linear', permutation_no=1, hold_out_words=False):
    # parse data
    sentences = parse(ud_data)
    # prep
    all_examples = []
    all_labels = []
    leven_distances_to_orig = []
    bleu_to_orig = []
    total_no_sents_at_len = len([s for s in sentences if len(s) < sentence_len_limit and len(s) > 3])
    # iterate over all sentences
    for sentence in sentences:
        #limit sentence len and no sents
        if len(sentence) < sentence_len_limit and len(sentence) > 3:
            # just randomly shuffle all toks irrespective of heirarch. structure
            if permutation_type == 'linear':
                # sent to list of tokens
                token_list = [t['form'] for t in sentence]
                label = [n / len(token_list)  for n in range(len(token_list))]
                # have half examples be permed
                if len(all_examples) < math.ceil(total_no_sents_at_len / 2):
                    # permute
                    #sample permutation
                    if shuffle_level:
                        c = list(zip(token_list, label))
                        random.shuffle(c)
                        token_list, label = zip(*c)

                    assert len(token_list) == len(label)
                    all_examples.append(' '.join(token_list))
                    all_labels.append(label)
                    # compute leven distaces
                    #for permutation in permutation_list:
                    ld = levenshtein_distance(' '.join(token_list), ' '.join(token_list))
                    leven_distances_to_orig.append(ld)
                    bs = nltk.translate.bleu_score.sentence_bleu(' '.join(token_list), ' '.join(token_list))
                    bleu_to_orig.append(bs)
                else:
                    assert len(token_list) == len(label)
                    all_examples.append(' '.join(token_list))
                    all_labels.append(label)
                    leven_distances_to_orig.append(0.0)
                    bleu_to_orig.append(100.0)

    return all_examples, all_labels,  leven_distances_to_orig, bleu_to_orig


def ud_load_regress_pairwise(ud_data, sentence_len_limit, shuffle_level, sample_no=3):
    sentences = parse(ud_data)
    all_examples, all_pairs, all_labels = [], [], []
    num_sents = len([s for s in sentences if 3 < len(s) < sentence_len_limit])
    # iterate over all sentences
    for sentence in sentences:
        if 3 < len(sentence) < sentence_len_limit:
            token_list = [t['form'] for t in sentence]
            labels, word_pairs, sent_pairs, sent_token_idxs = [], [], [], []
            for s in range(sample_no):
                token_idx = random.sample(range(len(token_list)), 2)
                sent_token_idxs.append(token_idx)
                dist = np.abs(token_idx[0] - token_idx[1]) - 1
                labels.append(dist)
                word_pairs.append((token_list[token_idx[0]], token_list[token_idx[1]]))

            if len(all_examples) < num_sents // 2:
                if shuffle_level:
                    c = list(enumerate(token_list))
                    random.shuffle(c)
                    original_indices, token_list = zip(*c)
                else:
                    original_indices = [n for n in range(len(token_list))]
                all_examples.append(' '.join(token_list))
                for i, tidx_pairs in enumerate(sent_token_idxs):
                    sent_pairs.append([original_indices.index(tidx_pairs[0]), original_indices.index(tidx_pairs[1])])
                all_pairs.append(sent_pairs)
                all_labels.append(labels)

    return all_examples, all_labels,  all_pairs


def ud_load_classify_pairwise(args, ud_data, sentence_len_limit, sample_no):
    # parse data
    leaveout = 0
    if 'leave' in args.perturb:
        leaveout = random.sample(range(1, 10), 2)

    sentences = parse(ud_data)
    random.shuffle(sentences)
    all_examples, all_pairs, all_labels = [], [], []
    for sentence in sentences:
        if 3 < len(sentence) < sentence_len_limit:
            token_list = [t['form'] for t in sentence if isinstance(t['id'], int)]
            labels, word_pairs, sent_pairs, sent_token_idxs = [], [], [], []
            gen = 0
            while gen < sample_no:
                token_idx = random.sample(range(1, len(token_list) + 1), 2)
                if 'leave' in args.perturb and ((token_idx[0] in leaveout) ^ (token_idx[1] in leaveout)):
                    continue

                if token_idx in sent_token_idxs:
                    gen += 1
                    continue

                sent_token_idxs.append(token_idx)
                order = 0 if token_idx[0] > token_idx[1] else 1
                labels.append(order)
                word_pairs.append((token_idx[0], token_idx[1]))
                gen += 1

            all_examples.append(' '.join(token_list))
            all_pairs.append(word_pairs)
            all_labels.append(labels)

    return all_examples, all_labels,  all_pairs, leaveout

