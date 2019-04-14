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
import random
import string
import glob
from collections import Counter

stopwords = nltk.corpus.stopwords.words('english')
random.seed(1234)

parser = argparse.ArgumentParser(description='Preprocessing for finding synonym/antonym relations')
parser.add_argument('--data', type=str, default='../data/glove',
                    help='location of the data corpus')
parser.add_argument('--out-dir', type=str, default='../data/glove/annotated',
                    help='location of the output directory')
parser.add_argument('--bptt', type=int, default=1,
                    help='bptt length')
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--ss_t', type=float, default=1e-5,
                    help='Subsampling Threshold')
parser.add_argument('--model', type=str, default='rnn',
                    help='Model type being used to train the embedding. Options are: [rnn, CBOW, retro]')
parser.add_argument('--max-pair', type=int, default=15,
                    help='max no of pairs of wordnet relations')
parser.add_argument('--lower', action='store_true',
                    help='Lowercase lemmas from wordnet.')
parser.add_argument('--version', type=int, default=1,
                    help='Version of the code to run.')

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


def create_vocab(in_path, add_eos=True):
    for w in get_tokens_from_file(in_path, add_eos):
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


def get_lexical_relations(word, word2idx):
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
            if args.lower:
                name = name.lower()

            tup = (word, name)
            if name != word and name in word2idx:
                synonyms.add(tup)

            for ant in lemma.antonyms():
                name = ant.name()
                if args.lower:
                    name = name.lower()

                tup = (word, name)
                if name in word2idx:
                    antonyms.add(tup)

        # hyp = syn.hypernyms() + syn.instance_hypernyms()
        hyp =  syn.instance_hypernyms()

        for h in hyp:
            if args.version < 2 and syn.pos() == 'v':
                continue

            for lemma in h.lemmas():
                name = lemma.name()
                if args.lower:
                    name = name.lower()

                if name == word:
                    continue

                tup = (word, name)
                if name in word2idx:
                    hypernyms.add(tup)

        hyp = syn.instance_hyponyms()
        if min([len(x) for x in syn.hypernym_paths()]) > 5:
            hyp += syn.hyponyms()

        for h in hyp:
            if args.version < 2 and syn.pos() == 'v':
                continue

            for lemma in h.lemmas():
                name = lemma.name()
                if args.lower:
                    name = name.lower()

                if name == word:
                    continue

                tup = (word, name)
                if name in word2idx:
                    hyponyms.add(tup)
                    hypernyms.add((name, word))

        mer = syn.member_meronyms() + syn.part_meronyms() + syn.substance_meronyms()
        for m in mer:

            for lemma in m.lemmas():
                name = lemma.name()
                if args.lower:
                    name = name.lower()

                if name == word:
                    continue

                tup = (name, word)
                if name in word2idx:
                    meronyms.add(tup)

        mer = syn.member_holonyms() + syn.part_holonyms() + syn.substance_holonyms()
        for m in mer:

            for lemma in m.lemmas():
                name = lemma.name()
                if args.lower:
                    name = name.lower()

                if name == word:
                    continue

                tup = (name, word)
                if name in word2idx:
                    holonyms.add(tup)
                    meronyms.add((word, name))

    return synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms

global_syn2rel = {}
global_synonyms = set([])
global_antonyms = set([])
global_hypernyms = set([])
global_hyponyms = set([])
global_meronyms = set([])
global_holonyms = set([])

def get_lexical_relations_seq(text):
    synonyms = set([])
    antonyms = set([])
    hypernyms = set([])
    hyponyms = set([])
    meronyms = set([])
    holonyms = set([])
    preprocessed_text = preprocessing(text)

    for w in preprocessed_text:
        if w not in global_syn2rel:
            # use simple nltk pos tagger for now
            pos_tags = nltk.pos_tag(w)
            # consider only adjectives for synonyms and antonyms
            p = get_wordnet_pos(pos_tags[-1][1])
            global_syn2rel[w] = { 'pos': p}

            if p is None:
                continue

            word_syn, word_ant, \
                word_hyp, word_hypo, \
                word_mer, word_hol = get_lexical_relations(w, word2idx)

            global_syn2rel[w].update({
                    'synonyms': word_syn,
                    'antonyms': word_ant,
                    'hypernyms': word_hyp,
                    'hyponyms': word_hypo,
                    'meronyms': word_mer,
                    'holonyms': word_hol
                    })

        else:
            if global_syn2rel[w]['pos'] is None:
                continue

            word_syn = global_syn2rel[w]['synonyms']
            word_ant = global_syn2rel[w]['antonyms']
            word_hyp = global_syn2rel[w]['hypernyms']
            word_hypo = global_syn2rel[w]['hyponyms']
            word_mer = global_syn2rel[w]['meronyms']
            word_hol = global_syn2rel[w]['holonyms']

        synonyms.update(word_syn)
        antonyms.update(word_ant)
        hypernyms.update(word_hyp)
        hyponyms.update(word_hypo)
        meronyms.update(word_mer)
        holonyms.update(word_hol)


        global_synonyms.update(word_syn)
        global_antonyms.update(word_ant)
        global_hypernyms.update(word_hyp)
        global_hyponyms.update(word_hypo)
        global_meronyms.update(word_mer)
        global_holonyms.update(word_hol)

    synonyms = sorted(list(synonyms))
    antonyms = sorted(list(antonyms))
    hypernyms = sorted(list(hypernyms))
    hyponyms = sorted(list(hyponyms))
    meronyms = sorted(list(meronyms))
    holonyms = sorted(list(holonyms))
    shuffle(synonyms)
    shuffle(antonyms)
    shuffle(hypernyms)
    shuffle(hyponyms)
    shuffle(meronyms)
    shuffle(holonyms)

    # synonym_str = ' '.join([','.join(syn) for syn in synonyms[:args.max_pair]])
    # antonym_str = ' '.join([','.join(ant) for ant in antonyms[:args.max_pair]])
    # hypernym_str = ' '.join([','.join(hyp) for hyp in hypernyms[:args.max_pair]])
    # hyponym_str = ' '.join([','.join(hyp) for hyp in hyponyms[:args.max_pair]])
    # meronym_str = ' '.join([','.join(mer) for mer in meronyms[:args.max_pair]])
    # holonym_str = ' '.join([','.join(mer) for mer in holonyms[:args.max_pair]])

    synonym_a = ' '.join([(syn[0]) for syn in synonyms[:args.max_pair]])
    synonym_b = ' '.join([(syn[1]) for syn in synonyms[:args.max_pair]])
    antonym_a = ' '.join([ant[0] for ant in antonyms[:args.max_pair]])
    antonym_b = ' '.join([ant[1] for ant in antonyms[:args.max_pair]])
    hypernym_a = ' '.join([(hyp[0]) for hyp in hypernyms[:args.max_pair]])
    hypernym_b = ' '.join([(hyp[1]) for hyp in hypernyms[:args.max_pair]])
    hyponym_a = ' '.join([(hyp[0]) for hyp in hyponyms[:args.max_pair]])
    hyponym_b = ' '.join([(hyp[1]) for hyp in hyponyms[:args.max_pair]])
    meronym_a = ' '.join([(mer[0]) for mer in meronyms[:args.max_pair]])
    meronym_b = ' '.join([(mer[1]) for mer in meronyms[:args.max_pair]])
    holonym_a = ' '.join([(mer[0]) for mer in holonyms[:args.max_pair]])
    holonym_b = ' '.join([(mer[1]) for mer in holonyms[:args.max_pair]])

    return {
                # 'synonyms': synonym_str,
                # 'antonyms': antonym_str,
                # 'hypernyms': hypernym_str,
                # 'hyponyms': hyponym_str,
                # 'meronyms': meronym_str,
                # 'holonyms': holonym_str
                'synonyms_a': synonym_a,
                'synonyms_b': synonym_b,
                'antonyms_a': antonym_a,
                'antonyms_b': antonym_b,
                'hypernyms_a': hypernym_a,
                'hypernyms_b': hypernym_b,
                'hyponyms_a': hyponym_a,
                'hyponyms_a': hyponym_b,
                'meronyms_a': meronym_a,
                'meronyms_b': meronym_b,
                'holonyms_a': holonym_a,
                'holonyms_b': holonym_b
           }


def get_tokens_from_file(in_path, add_eos=True):
    with codecs.open(in_path, 'r', encoding="utf8") as f:
        tokens = []
        for line in f:
            words = line.split()
            if add_eos:
                words = words + ['<eos>']
            tokens.extend(words)
    return tokens

def get_tokens_with_dict(in_path, add_eos=True):
    with codecs.open(in_path, 'r', encoding="utf8") as f:
        tokens = []
        for line in f:
            words = line.split()
            if add_eos:
                words = words + ['<eos>']
            tokens.extend(words)

    return tokens, Counter(tokens)


def create_rnn_corpus(in_path, out_path):
    tokens = get_tokens_from_file(in_path, add_eos=True)

    num_batches = int(math.ceil(len(tokens)/args.batch_size))
    batched_input = []
    for batch in range(0, len(tokens), num_batches):
        batched_input.append(tokens[batch:batch + num_batches])

    with codecs.open(out_path, 'w', encoding="utf-8") as out_file:
        outputs = []
        for i in range(0, num_batches, args.bptt):
            seq_len = min(args.bptt, num_batches - i - 1)

            if seq_len < args.bptt:
                continue

            for k in range(args.batch_size):

                text = batched_input[k][i:i+seq_len]
                if len(text) == 0:
                    continue

                target = batched_input[k][i+1:i+1+seq_len]

                text_str = ' '.join(text)
                target_str = ' '.join(target)

                output = {'text': text_str,
                          'target': target_str}
                output.update(get_lexical_relations_seq(text))
                outputs.append(output)

            print('{}/{}\r'.format(i, num_batches), end='\r')

        for output in outputs:
            out_file.write(str(json.dumps(output)) + '\n')

def create_glove_corpus(in_path, out_path):
    tokens = get_tokens_from_file(in_path, add_eos=False)

    with codecs.open(out_path, 'w', encoding="utf-8") as out_file:
        outputs = []
        for i, token in enumerate(tokens):

            if not token:
                continue

            output = get_lexical_relations_seq([token])
            output['text'] = token
            output['target'] = token
            outputs.append(output)
            print('{0}\r'.format(i), end='\r')

        for output in outputs:
            out_file.write(str(json.dumps(output)) + '\n')

def create_cbow_corpus(in_path, out_path):
    tokens = get_tokens_from_file(in_path, add_eos=False)
    context=4

    with codecs.open(out_path, 'w', encoding="utf-8") as out_file:
        for i in range(4, len(tokens)-4):
            text = tokens[i-4:i] + tokens[i+1:i+5]
            target = tokens[i]

            text_str = ' '.join(text)

            output = get_lexical_relations_seq([target])
            output['text'] = text_str
            output['target'] = target

            out_file.write(str(json.dumps(output)) + '\n')
            out_file.flush()

def create_skipgram_corpus(in_path, out_path):
    tokens, w2freq = get_tokens_with_dict(in_path, add_eos=False)
    total = sum(w2freq.values(), 0.0)
    w2freq = {key:w2freq[key]/total for key in w2freq.keys()}

    context=4

    with codecs.open(out_path, 'w', encoding="utf-8") as out_file:
        for i in range(context, len(tokens)-context):
            text = tokens[i]
            p = 1. - math.sqrt(1e-5/w2freq[text])
            if random.random() > p:
                output = get_lexical_relations_seq([text])

                output['text'] = text
                target = tokens[i-context:i] + tokens[i+1:i+context+1]
                output['target'] = ' '.join(target)

                out_file.write(str(json.dumps(output)) + '\n')
                out_file.flush()

if args.model == 'retro':
    create_vocab(os.path.join(args.data, 'vocab.txt'), add_eos=False)
else:
    create_vocab(os.path.join(args.data, 'train.txt'))

out_dir = os.path.join(args.data,
                        'annotated_{}_{}_{}'.format(args.model, args.bptt, args.batch_size) if args.model == 'rnn' else \
                        'annotated_{}'.format(args.model))
if args.version > 1:
    out_dir += '_v{}'.format(args.version)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

print("Output Dir: %s" % out_dir)

if args.model == 'retro':
    create_corpus = create_glove_corpus
    train, valid, test = ['vocab.txt'] * 3
    args.lower = True
elif args.model == 'rnn':
    create_corpus = create_rnn_corpus
    train, valid, test = ['train.txt', 'valid.txt', 'test.txt']
elif args.model == 'cbow':
    create_corpus = create_cbow_corpus
    train, valid, test = ['train.txt', 'valid.txt', 'test.txt']
elif args.model == 'skipgram':
    create_corpus = create_skipgram_corpus
    train, valid, test = ['train.txt', 'valid.txt', 'test.txt']

print('Creating train files')
create_corpus(os.path.join(args.data, train), os.path.join(out_dir, 'train.txt'))
print('Creating test files')
create_corpus(os.path.join(args.data, test), os.path.join(out_dir, 'test.txt'))
print('Creating valid files')
create_corpus(os.path.join(args.data, valid), os.path.join(out_dir, 'valid.txt'))

if args.model == 'retro':
    with open('syn_v{}.txt'.format(args.version), 'w') as syn:
        for syn_pair in global_synonyms:
            syn.write('%s\t%s\n' % syn_pair)

    with open('ant_v{}.txt'.format(args.version), 'w') as ant:
        for ant_pair in global_antonyms:
           ant.write('%s\t%s\n' % ant_pair)

    with open('hyp_v{}.txt'.format(args.version), 'w') as hyp:
        for hyp_pair in global_hyponyms:
            hyp.write('%s\t%s\n' % hyp_pair)

    with open('mer_v{}.txt'.format(args.version), 'w') as mer:
        for mer_pair in global_meronyms:
            mer.write('%s\t%s\n' % mer_pair)

for pkl_file in glob.glob('/'.join([out_dir, '*.pkl'])):
    os.remove(pkl_file)
