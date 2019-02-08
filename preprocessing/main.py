# coding: utf-8
import argparse
import json
import time
import math
import os
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from random import shuffle
import codecs
import string

stopwords = nltk.corpus.stopwords.words('english')

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

def preprocessing(text):
    return [tok for tok in text if tok not in stopwords and tok not in string.punctuation]

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

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_lexical_relations(word, pos_tag, word2idx):
    synonyms = set([]); antonyms = set([]);
    hypernyms = set([]); hyponyms = set([]);
    meronyms = set([]); holonyms = set([])
    try:
        synsets = wordnet.synsets(word)
    except:
        pass

    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name()
            if name == word:
                continue

            tup = (word, name)
            if name in word2idx:
                synonyms.add(tup)

            for ant in lemma.antonyms():
                name = ant.name()
                tup = (word, name)
                if name in word2idx:
                    antonyms.add(tup)

        if syn.pos() != 'v':
            hyp = syn.hypernyms() + syn.instance_hypernyms()
            for h in hyp:
                for lemma in h.lemmas():
                    name = lemma.name()
                    if name == word:
                        continue

                    tup = (word, name)
                    if name in word2idx:
                        hypernyms.add(tup)

            hyp = syn.hyponyms() + syn.instance_hyponyms()
            for h in hyp:
                for lemma in h.lemmas():
                    name = lemma.name()
                    if name == word:
                        continue

                    tup = (word, name)
                    if name in word2idx:
                        hyponyms.add(tup)

        mer = syn.member_meronyms() + syn.part_meronyms() + syn.substance_meronyms()
        for m in mer:
            for lemma in m.lemmas():
                name = lemma.name()
                if name == word:
                    continue

                tup = (name, word)
                if name in word2idx:
                    meronyms.add(tup)

        mer = syn.member_holonyms() + syn.part_holonyms() + syn.substance_holonyms()
        for m in mer:
            for lemma in m.lemmas():
                name = lemma.name()
                if name == word:
                    continue

                tup = (name, word)
                if name in word2idx:
                    holonyms.add(tup)

    return synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms



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
                hyponyms = set([])
                meronyms = set([])
                holonyms = set([])
                text = batched_input[k][i:i+seq_len]
                target = batched_input[k][i+1:i+1+seq_len]

                preprocessed_text = preprocessing(text)
                # use simple nltk pos tagger for now
                pos_tags = nltk.pos_tag(preprocessed_text)
                for w, p in zip(preprocessed_text, pos_tags):
                    # consider only adjectives for synonyms and antonyms
                    p = get_wordnet_pos(p[1])
                    if p is None:
                        continue
                    word_syn, word_ant, word_hyp, word_hypo, \
                        word_mer, word_hol = get_lexical_relations(w, p, word2idx)
                    synonyms.update(word_syn)
                    antonyms.update(word_ant)
                    hypernyms.update(word_hyp)
                    hyponyms.update(word_hypo)
                    meronyms.update(word_mer)
                    holonyms.update(word_hol)

                synonyms = list(synonyms)
                antonyms = list(antonyms)
                hypernyms = list(hypernyms)
                hyponyms = list(hyponyms)
                meronyms = list(meronyms)
                holonyms = list(holonyms)
                shuffle(synonyms)
                shuffle(antonyms)
                shuffle(hypernyms)
                shuffle(hyponyms)
                shuffle(meronyms)
                shuffle(holonyms)

                text_str = ' '.join(text)
                target_str = ' '.join(target)

                synonym_str = ' '.join([','.join(syn) for syn in synonyms[:args.max_pair]])
                antonym_str = ' '.join([','.join(ant) for ant in antonyms[:args.max_pair]])
                hypernym_str = ' '.join([','.join(hyp) for hyp in hypernyms[:args.max_pair]])
                hyponym_str = ' '.join([','.join(hyp) for hyp in hyponyms[:args.max_pair]])
                meronym_str = ' '.join([','.join(mer) for mer in meronyms[:args.max_pair]])
                holonym_str = ' '.join([','.join(mer) for mer in holonyms[:args.max_pair]])

                output = {
                            'text': text_str,
                            'target': target_str,
                            'synonyms': synonym_str,
                            'antonyms': antonym_str,
                            'hypernyms': hypernym_str,
                            'hyponyms': hyponym_str,
                            'meronyms': meronym_str,
                            'holonyms': holonym_str
                         }
                f1.write(str(json.dumps(output)) + '\n')
                f1.flush()

create_vocab(os.path.join(args.data, 'train.txt'))
# create_vocab(os.path.join(args.data, 'test.txt'))
# create_vocab(os.path.join(args.data, 'valid.txt'))

out_dir = os.path.join(args.data, 'annotated_{}_{}'.format(args.bptt, args.batch_size))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print('Creating train files')
create_corpus(os.path.join(args.data, 'train.txt'), os.path.join(out_dir, 'train.txt'))
print('Creating test files')
create_corpus(os.path.join(args.data, 'test.txt'), os.path.join(out_dir, 'test.txt'))
print('Creating valid files')
create_corpus(os.path.join(args.data, 'valid.txt'), os.path.join(out_dir, 'valid.txt'))
