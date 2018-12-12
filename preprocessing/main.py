# coding: utf-8
import argparse
import json
import time
import math
import os
import nltk
from nltk.corpus import wordnet
from random import shuffle
import codecs
import string

parser = argparse.ArgumentParser(description='Preprocessing for finding synonym/antonym relations')
parser.add_argument('--data', type=str, default='../data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--out-dir', type=str, default='../data/wikitext-2/annotated',
                    help='location of the output directory')
parser.add_argument('--bptt', type=int, default=35,
                    help='bptt length')
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--max-pair', type=int, default=100,
                    help='max no of pairs of wordnet relations')

args = parser.parse_args()
word2idx = {}
idx2word = []

# taken from https://stackoverflow.com/questions/8689795/how-can-i-remove-non-ascii-characters-but-leave-periods-and-spaces-using-python
def remove_non_ascii(text):
    printable = set(string.printable)
    cleaned_text = filter(lambda x: x in printable, text)
    return ''.join([ch for ch in cleaned_text]) # a hack to convert filter object to string for Python 3

def add_word(word):
    if word not in word2idx:
        idx2word.append(word)
        word2idx[word] = len(idx2word) - 1
    return word2idx[word]


def create_vocab(in_path):
    with codecs.open(in_path, 'r', encoding="utf8") as f:
        add_word('<eos>')
        for line in f:
            words = line.split()
            for w in words:
                add_word(w)

def create_corpus(in_path, out_path):
    f1 = codecs.open(out_path, 'w', encoding="utf-8")
    with codecs.open(in_path, 'r', encoding="utf8") as f:
        tokens = []
        for line in f:
            words = line.split()
            words = words + ['<eos>']
            tokens.extend(words)

        num_batches = math.ceil(len(tokens)/args.batch_size)

        batched_input = []
        for batch in range(0, len(tokens), num_batches):
            batched_input.append(tokens[batch:batch + num_batches])

        for i in range(0, num_batches, args.bptt):
            seq_len = min(args.bptt, num_batches - i - 1)

            for k in range(args.batch_size):
                synonyms = set([])
                antonyms = set([])
                hypernyms = set([])
                text = batched_input[k][i:i+seq_len]
                target = batched_input[k][i+1:i+1+seq_len]

                # use simple nltk pos tagger for now
                pos_tags = nltk.pos_tag(text)
                for j, w in enumerate(text):
                    # consider only adjectives for synonyms and antonyms
                    for syn in wordnet.synsets(w):
                        for lemma in syn.lemmas():
                            name = lemma.name()

                            if name == w:
                                continue

                            tup = (w, name)
                            if name in word2idx:
                                synonyms.add(tup)

                            for ant in lemma.antonyms():
                                name = ant.name()
                                tup = (w, name)
                                if name in word2idx:

                                    antonyms.add(tup)
                    for syn in wordnet.synsets(w):
                        
                        hyp = syn.hypernyms()
                        for h in hyp:
                            for lemma in h.lemmas():
                                name = lemma.name()

                                if name == w:
                                    continue

                                tup = (w, name)
                                if name in word2idx:
                                    hypernyms.add(tup)

                synonyms = list(synonyms)
                antonyms = list(antonyms)
                hypernyms = list(hypernyms)
                shuffle(synonyms)
                shuffle(antonyms)
                shuffle(hypernyms)

                text_str = ' '.join(text)
                target_str = ' '.join(target)

                synonym_str = ' '.join([','.join(syn) for syn in synonyms[:args.max_pair]])
                antonym_str = ' '.join([','.join(ant) for ant in antonyms[:args.max_pair]])
                hypernym_str = ' '.join([','.join(hyp) for hyp in hypernyms[:args.max_pair]])

                output = {
                            'text': text_str,
                            'target': target_str,
                            'synonyms': synonym_str,
                            'antonyms': antonym_str,
                            'hypernyms': hypernym_str
                         }
                f1.write(str(json.dumps(output)) + '\n')
                f1.flush()

create_vocab(os.path.join(args.data, 'train.txt'))
# create_vocab(os.path.join(args.data, 'test.txt'))
# create_vocab(os.path.join(args.data, 'valid.txt'))
print('Creating train files')
create_corpus(os.path.join(args.data, 'train.txt'), os.path.join(args.out_dir, 'train.txt'))
print('Creating test files')
create_corpus(os.path.join(args.data, 'test.txt'), os.path.join(args.out_dir, 'test.txt'))
print('Creating valid files')
create_corpus(os.path.join(args.data, 'valid.txt'), os.path.join(args.out_dir, 'valid.txt'))
