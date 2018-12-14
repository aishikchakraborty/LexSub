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
parser.add_argument('--analogy-task', action='store_true',
                    help='get Google Analogy Task Results')
parser.add_argument('--sim-task', action='store_true',
                    help='use similarity task')
args = parser.parse_args()

class WordSimilarity():
    def __init__(self, datasets):
        self.vocab = pickle.load(open('../vocab_wikitext-2.pkl', 'rb'))
        self.datasets = datasets

    def load_vocab(self):
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
                gold_sim.append(sim)
                if i%1000 == 0:
                    print('Processed ' + str(i+1) + ' test examples')
                try:
                    w1 = self.emb[self.vocab.stoi[w1]]
                except:
                    all_present = False
                    # w1 = self.sem_emb[self.['<unk>']].reshape(-1, 1)
                try:
                    w2 = self.emb[self.vocab.stoi[w2]]
                except:
                    all_present = False
                    # w2 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)
                if all_present:
                    pred_sim.append(self.cossim(w1, w2).item())
                    total_examples += 1
            print('Spearman: ' + str(spearmanr(gold_sim, pred_sim)[0]))
            print('Pearson: ' + str(pearsonr(gold_sim, pred_sim)[0]))


class AnalogyExperiment():
    def __init__(self):
        self.vocab = pickle.load(open('../vocab.pkl', 'rb'))

    def load_vocab(self):
        self.emb = pickle.load(open(args.emb, 'rb'))

    def cossim(self, v1, v2):
        dot = v1.dot(v2)
        return dot/(v1.norm() * v2.norm())

    def load_google_dataset(self):
        f = open('unsup_datasets/googleanalogytestset.csv', 'r')
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
                w1 = self.emb[self.vocab.stoi[w1]].reshape(-1, 1)
            except:
                all_present = False
                # w1 = self.sem_emb[self.['<unk>']].reshape(-1, 1)
            try:
                w2 = self.emb[self.vocab.stoi[w2]].reshape(-1, 1)
            except:
                all_present = False
                # w2 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)
            try:
                w3 = self.sem_emb[self.vocab.stoi[w3]].reshape(-1, 1)
            except:
                all_present = False
                # w3 = self.sem_emb[self.w2idx['<unk>']].reshape(-1, 1)

            try:
                w4 = self.vocab.stoi[w4]
            except:
                all_present = False
                # w4 = self.w2idx['<unk>']
            if all_present:
                sim_all = np.matmul(self.emb, np.concatenate((w1, w2, w3), axis=1))
                cos_add = sim_all[:,1] + sim_all[:,2] - sim_all[:,0]
                for wi in (w1, w2, w3):
                    try:
                        w_id = self.vocab.stoi[wi]
                        cos_add[w_id] = -np.inf

                    except KeyError:
                        missing_words += 1
                best_idx = np.argmax(cos_add)
                if best_idx == w4:
                    correct_pred += 1
                total_examples += 1

        print(correct_pred)
        print(total_examples)
        print(float(correct_pred)/float(total_examples))


if args.analogy_task:
    ae = AnalogyExperiment()
    ae.load_vocab()
    ae.load_google_dataset()
elif args.sim_task:
    datasets = ['bakerverb143', 'men_dev', 'men_test', 'radinskymturk', 'semeval17task2_test', 'semeval17task2_trial', 'simlex999', 'simverb3500', 'wordsim353_relatedness', 'wordsim353_similarity', 'yangpowersverb130']
    ae = WordSimilarity(datasets)
    ae.load_vocab()
    ae.load_similarity()
