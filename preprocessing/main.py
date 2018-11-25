# coding: utf-8
import argparse
import time
import math
import os
from nltk.corpus import wordnet
import codecs

parser = argparse.ArgumentParser(description='Preprocessing for finding synonym/antonym relations')
parser.add_argument('--data', type=str, default='../data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--out-dir', type=str, default='../data/wikitext-2/annotated',
                    help='location of the output directory')

args = parser.parse_args()
word2idx = {}
idx2word = []

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
        for line in f:
            synonyms = []
            antonyms = []
            words = line.split() + ['<eos>']
            for w in words:
                syns = wordnet.synsets(w)
                if syns:
                    for s in syns:
                        if s.name().split('.')[0] in idx2word:
                            synonyms.append((w, s.name().split('.')[0]))
                        for s_ in s.lemmas():
                            if s_.antonyms():
                                for s__ in s_.antonyms():
                                    if s__.name().split()[0] in idx2word:
                                        antonyms.append((w, s__.name().split()[0]))
            if len(synonyms) == 0:
                synonyms = [('<unk>', '<unk>')]
            if len(antonyms) == 0:
                antonyms = [('<unk>', '<unk>')]
            f1.write(str(words) + '\t' + str(synonyms) + '\t' + str(antonyms) + '\n')
            f1.flush()



create_vocab(os.path.join(args.data, 'train.txt'))
create_vocab(os.path.join(args.data, 'test.txt'))
create_vocab(os.path.join(args.data, 'valid.txt'))
print('Creating train files')
create_corpus(os.path.join(args.data, 'train.txt'), os.path.join(args.out_dir, 'train.txt'))
print('Creating test files')
create_corpus(os.path.join(args.data, 'test.txt'), os.path.join(args.out_dir, 'test.txt'))
print('Creating valid files')
create_corpus(os.path.join(args.data, 'valid.txt'), os.path.join(args.out_dir, 'valid.txt'))
