import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import argparse
import csv
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import _pickle as pickle

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--emb', type=str, default='../emb_Vanilla.pkl',
                    help='location of the trained embeddings')
parser.add_argument('--vocab', type=str, default='../vocab_wikitext-103.pkl',
                    help='location of the vocab')
parser.add_argument('--analogy-task', action='store_true',
                    help='get Google Analogy Task Results')
parser.add_argument('--sim-task', action='store_true',
                    help='use similarity task')
parser.add_argument('--text', action='store_true', help='read embedding files in text format')

args = parser.parse_args()

def read_text_file(file_path):
    print('Reading embedding files')
    vocab, emb = [], []
    f = open(file_path, 'r')
    for lines in f:
        lines = lines.strip().split()
        vocab.append(lines[0])
        emb.append([float(val) for val in lines[1:]])
    vocab = Vocab(vocab)
    print('Finished Reading Embedding File')
    print(np.array(emb).shape)
    return vocab, np.array(emb, dtype=np.float64)


class Vocab():
    def __init__(self, vocab):
        self.itos = vocab
        self.stoi = {w:i for i, w in enumerate(vocab)}



class WordSimilarity():
    def __init__(self, datasets):
        self.datasets = datasets

    def load_vocab(self):
        self.vocab = pickle.load(open(args.vocab, 'rb'))

        self.emb = pickle.load(open(args.emb, 'rb'))

    def cossim(self, v1, v2):
        dot = v1.dot(v2)
        return dot/(v1.norm() * v2.norm())

    def load_similarity(self):
        for d in self.datasets:
            print('*'*89)
            print(d)
            f = open('unsup_datasets/' + d + '.csv', 'r')
            data_reader = csv.reader(f, delimiter=',')
            missing_words = 0
            correct_pred = 0
            total_examples = 0

            gold_sim = []
            pred_sim = []
            for i, row in enumerate(data_reader):
                all_present = True
                if i == 0:
                    continue
                w1, w2, sim = row[0], row[1], float(row[2])
                if i%1000 == 0:
                    print('Processed ' + str(i+1) + ' test examples')
                try:
                    w1_ = self.vocab.stoi[w1]
                    w1 = self.emb[w1_]
                    if w1_ == 0:
                        all_present = False
                except:
                    all_present = False
                    # w1 = self.sem_emb[self.['<unk>']].reshape(-1, 1)
                try:
                    w2_ = self.vocab.stoi[w2]
                    w2 = self.emb[w2_]
                    if w2_ == 0:
                        all_present = False
                except:
                    all_present = False
                    # w2 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)
                if all_present:
                    pred_sim.append(self.cossim(w1, w2).item())
                    gold_sim.append(sim)
                    total_examples += 1

            print('Spearman: ' + str(spearmanr(gold_sim, pred_sim)[0]))
            print('Pearson: ' + str(pearsonr(gold_sim, pred_sim)[0]))


class AnalogyExperiment():
    def __init__(self):
        pass

    def load_vocab(self):
        self.vocab = pickle.load(open(args.vocab,'rb'))
        print(self.vocab)
        self.emb = pickle.load(open(args.emb, 'rb'))

    def cossim(self, v1, v2):
        dot = v1.dot(v2)
        return dot/(v1.norm() * v2.norm())

    def load_google_dataset(self):
        # for d in ['googleanalogytestset', 'biggeranalogytestset']:
        # for d in ['googleanalogytestset']:
        for d in ['semantics_analogytest', 'syntactic_analogytest']:
            print('*'*89)
            print(d)
            print('-'*89)
            f = open('unsup_datasets/' + d + '.csv', 'r')
            data_reader = csv.reader(f, delimiter=',')
            missing_words = 0
            correct_pred = 0
            total_examples = 0

            for i, row in enumerate(data_reader):
                all_present = True
                w1, w2, w3, w4 = row[0], row[1], row[2], row[3]
                if i%1000 == 0:
                    print('Processed ' + str(i+1) + ' test examples')
                try:
                    w1_ = self.vocab.stoi[w1.lower()]
                    w1 = self.emb[self.vocab.stoi[w1.lower()]].view(1, -1)

                except:
                    all_present = False
                    # w1 = self.sem_emb[self.['<unk>']].reshape(-1, 1)
                try:
                    w2_ = self.vocab.stoi[w2.lower()]
                    w2 = self.emb[self.vocab.stoi[w2.lower()]].view(1, -1)
                except:
                    all_present = False
                    # w2 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)
                try:
                    w3_ = self.vocab.stoi[w3.lower()]
                    w3 = self.emb[self.vocab.stoi[w3.lower()]].view(1, -1)
                except:
                    all_present = False
                    # w3 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)

                try:
                    w4 = self.vocab.stoi[w4.lower()]
                except:
                    all_present = False
                    # w4 = self.w2idx['<unk>']
                if all_present and bool(0 not in set([w1_, w2_, w3_, w4])):
                    cos_add = F.cosine_similarity(w2, self.emb, dim=1) + F.cosine_similarity(w3, self.emb, dim=1) - F.cosine_similarity(w1, self.emb, dim=1) #vocab, emsize
                    # sim_all = np.matmul(self.emb, np.concatenate((w1, w2, w3), axis=1))
                    # cos_add = sim_all[:,1] + sim_all[:,2] - sim_all[:,0]
                    for wi in (w1_, w2_, w3_):
                        try:
                            # w_id = self.vocab.stoi[wi]
                            cos_add[wi] = -np.inf

                        except KeyError:
                            missing_words += 1
                    best_idx = np.argmax(cos_add.cpu().numpy())
                    if best_idx == w4:
                        correct_pred += 1
                    total_examples += 1

            print(correct_pred)
            print(total_examples)
            print(float(correct_pred)/float(total_examples))


if args.text:
    vocab, emb = read_text_file(args.emb)
if args.analogy_task:
    ae = AnalogyExperiment()
    if args.text:
        ae.vocab = vocab
        ae.emb = torch.tensor(emb).cuda()
    else:
        ae.load_vocab()
    ae.load_google_dataset()
elif args.sim_task:
    datasets = ['bakerverb143', 'men3k', 'men_dev', 'men_test', 'radinskymturk', 'semeval17task2_test', 'semeval17task2_trial', 'simlex999', 'simverb3500', 'wordsim353_relatedness', 'wordsim353_similarity', 'yangpowersverb130', 'rarewords']
    ae = WordSimilarity(datasets)
    if args.text:
        ae.vocab = vocab
        ae.emb = torch.tensor(emb).cuda()
    else:
        ae.load_vocab()
    ae.load_similarity()
