# coding: utf-8
import argparse
import time
import math
import os
import nltk
from nltk.corpus import wordnet
import codecs
import string

parser = argparse.ArgumentParser(description='Preprocessing for finding synonym/antonym relations')
parser.add_argument('--data', type=str, default='../data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--out-dir', type=str, default='../data/wikitext-2/annotated',
                    help='location of the output directory')
parser.add_argument('--bptt', type=int, default=35,
                    help='bptt length')
parser.add_argument('--max-pair', type=int, default=35,
                    help='max no of synonyms')

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
        for line in f:
            synonyms = set([])
            antonyms = set([])
            line = remove_non_ascii(line.strip())

            if not line:
                continue
            words = line.split()
            words = words + ['<eos>']
            # use simple nltk pos tagger for now
            pos_tags = nltk.pos_tag(words)

            # for i in range(0, len(words) - args.bptt + 1):
                # window = words[i:i+args.bptt]
                # for j, w in enumerate(window):
            for i, w in enumerate(words):
                    # consider only adjectives for synonyms and antonyms
                if pos_tags[i][1] == 'JJ':
                    for syn in wordnet.synsets(w):
                        for lemma in syn.lemmas():
                            name = lemma.name()
                            if name in word2idx:
                                if len(synonyms) < args.max_pair:
                                    synonyms.add((w, name))
                                else:
                                    break
                            for ant in lemma.antonyms():
                                name = ant.name()
                                if name in word2idx:
                                    if len(antonyms) < args.max_pair:
                                        antonyms.add((w, name))
                                    else:
                                        break

                word_str = ' '.join(words)
                synonym_str = ' '.join([','.join(syn) for syn in synonyms])
                antonym_str = ' '.join([','.join(ant) for ant in antonyms])
                f1.write('{}\n'.format('\t'.join([word_str, synonym_str, antonym_str])))
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
