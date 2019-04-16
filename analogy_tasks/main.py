import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import argparse
import csv
import json
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score
import _pickle as pickle

import hypernymysuite_eval

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--emb', type=str, default='../emb_Vanilla.pkl',
                    help='location of the trained embeddings')
parser.add_argument('--emb2', type=str, help='location of second embedding. This is only different from emb1 in case of hyp and mer subspace embedding.')
parser.add_argument('--vocab', type=str, default='../vocab_wikitext-103.pkl',
                    help='location of the vocab')
parser.add_argument('--analogy-task', action='store_true',
                    help='get Google Analogy Task Results')
parser.add_argument('--sim-task', action='store_true', help='use similarity task')
parser.add_argument('--hypernymy', action='store_true', help='Run HypernymySuite experiments.')
parser.add_argument('--neighbors', action='store_true', help='Dump neighbors')
parser.add_argument('--text', action='store_true', help='read embedding files in text format')
parser.add_argument('--prefix', type=str, default='', help='Prefix to add to word similarity task names.')
parser.add_argument('--output_file', type=str, default=None, help='File to which to dump the outputs. Used specifically for dumping neighbours')

args = parser.parse_args()

def read_text_file(file_path):
    print('Reading embedding files')
    vocab, emb = [], []
    f = open(file_path, 'r')
    for lines in f:
        lines = lines.strip().split()
        vocab.append(lines[0])
        emb.append([float(val) for val in lines[1:]])
    
    # Counterfitting doesn't have unk token. 
    # It deletes unk token which we need for similarity and other intrinsic tasks. 
    if 'unk' not in vocab:
        vocab.append('unk')
        emb.append([0] * len(emb[-1]))

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
        self.vocab = pickle.load(open(args.vocab, 'rb')).stoi

        self.emb1 = pickle.load(open(args.emb, 'rb'))
        if args.emb2:
            self.emb2 = pickle.load(open(args.emb2, 'rb'))
        else:
            self.emb2 = pickle.load(open(args.emb, 'rb'))

    def cossim(self, v1, v2):
        dot = v1.dot(v2)
        return dot/(v1.norm() * v2.norm())

    def load_similarity(self):
        for d in self.datasets:
            print('*'*89)
            print(args.prefix + '_' + d if args.prefix != '' else d)
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

                w1_ = self.vocab.get(w1, self.vocab['unk'])
                w1 = self.emb1[w1_]

                w2_ = self.vocab.get(w2, self.vocab['unk'])
                w2 = self.emb2[w2_]

                pred_sim.append(F.cosine_similarity(w1.view(1, -1), w2.view(1, -1)).item())
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

class RankedNeighbors():
    def __init__(self, topk=100):
        self.topk=topk
        self.output_filename = args.output_file
        self.neighbors = {'syn': {}, 'hyp': {}, 'mer': {}}

        for rel,neighbor_file, order in [('syn', '../preprocessing/syn_v2.txt', None),
                                  ('hyp', '../preprocessing/hyp_v2.txt', 'reverse'),
                                  ('mer', '../preprocessing/mer_v2.txt', 'reverse')]:
            with open(neighbor_file) as f:
                for line in f:
                    word1, word2 = line.split()
                    if order == 'reverse':
                        word3 = word2
                        word2 = word1
                        word1 = word3

                    if word1 not in self.neighbors[rel]:
                        self.neighbors[rel][word1] = set([])

                    self.neighbors[rel][word1].add(word2)

    def load_vocab(self):
        self.vocab = pickle.load(open(args.vocab, 'rb'))

        self.emb1 = pickle.load(open(args.emb, 'rb'))
        if args.emb2:
            self.emb2 = pickle.load(open(args.emb2, 'rb'))
        else:
            self.emb2 = pickle.load(open(args.emb, 'rb'))


    def dump_neighbors(self):
        rel_aps={'syn': [], 'hyp': [], 'mer':[]}
        with open(self.output_filename, 'w') as out_file:
            words=set([])
            for d in ['unsup_datasets/simlex999.csv','unsup_datasets/hyperlex.csv']:
                data_reader = csv.reader(open(d), delimiter=',')
                for i, row in enumerate(data_reader):
                    if i == 0:
                        continue
                    w1, w2  = row[0], row[1]
                    words.add(w1)
                    words.add(w2)

            words = sorted(list(words))
            for i, x in enumerate(words):
                if x not in self.vocab.stoi:
                    continue

                x = x.strip()
                i = self.vocab.stoi[x]
                wemb = self.emb1[i].view(1, -1)
                neighbors = []
                scores = F.cosine_similarity(wemb, self.emb2, dim=1)
                scores[i] = -float("inf")

                score_topk, topk = torch.topk(scores, k=self.topk)
                for score, idx in list(zip(score_topk, topk))[:10]:
                    neighbors.append('%s(%0.3f)'% (self.vocab.itos[idx], score))
                out_file.write('%s\t%s\n' % (self.vocab.itos[i], '\t'.join(neighbors)))
                aps = []
                for rel in ['syn', 'hyp', 'mer']:
                    if x in self.neighbors[rel]:
                        y_true=[self.vocab.itos[j] in self.neighbors[rel].get(x, set([])) for j in topk]
                        y_score=score_topk.cpu().numpy()
                        score=average_precision_score(y_true, y_score)
                        if np.isnan(score):
                            score = 0
                        rel_aps[rel].append(score)
            
            for rel in ['syn', 'hyp', 'mer']:
                print('Mean Average Precision *%s* Score: %.4f' %(rel, np.mean(rel_aps[rel])))

class HypernymySuiteModel(object):
    """
    Base class for all hypernymy suite models.

    To use this, must implement these methods:

        predict(self, hypo: str, hyper: str): float, which makes a
            prediction about two words.
        vocab: dict[str, int], which tells if a word is in the
            vocabulary.

    Your predict method *must* be prepared to handle OOV terms, but it may
    returning any sentinel value you wish.

    You can optionally implement
        predict_many(hypo: list[str], hyper: list[str]: array[float]

    The skeleton method here will just call predict() in a for loop, but
    some methods can be vectorized for improved performance. This is the
    actual method called by the evaluation script.
    """

    vocab = {}

    def __init__(self):
        self.vocab = None
        self.emb1 = None
        self.emb2 = None

    def load_vocab(self):
        self.vocab = pickle.load(open(args.vocab, 'rb')).stoi
        self.emb1 = pickle.load(open(args.emb, 'rb'))
        if args.emb2:
            self.emb2 = pickle.load(open(args.emb2, 'rb'))
        else:
            self.emb2 = pickle.load(open(args.emb, 'rb'))

    def predict(self, hypo, hyper):
        """
        Core modeling procedure, estimating the degree to which hypo is_a hyper.

        This is an abstract method, describing the interface.

        Args:
            hypo: str. A hypothesized hyponym.
            hyper: str. A hypothesized hypernym.

        Returns:
            float. The score estimating the degree to which hypo is_a hyper.
                Higher values indicate a stronger degree.
        """
        w1_ = self.vocab.get(hypo, self.vocab['unk'])
        w1 = self.emb1[w1_]


        w2_ = self.vocab.get(hyper, self.vocab['unk'])
        w2 = self.emb2[w2_]

        return F.cosine_similarity(w1.view(1, -1), w2.view(1, -1)).item()


    def predict_many(self, hypos, hypers):
        """
        Make predictions for many pairs at the same time. The default
        implementation just calls predict() many times, but many models
        benefit from vectorization.

        Args:
            hypos: list[str]. A list of hypothesized hyponyms.
            hypers: list[str]. A list of corresponding hypothesized hypernyms.
        """
        result = []
        for x, y in zip(hypos, hypers):
            result.append(self.predict(x, y))
        return np.array(result, dtype=np.float32)

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
    datasets = ['bakerverb143', 'men3k', 'men_dev', 'men_test', 'radinskymturk', 'semeval17task2_test', 'semeval17task2_trial', 'simlex999', 'simverb3500', 'wordsim353_relatedness', 'wordsim353_similarity', 'yangpowersverb130', 'rarewords','hyperlex', 'hyperlex-nouns', 'hyperlex_test']
    # datasets = ['men3k', 'wordsim353_relatedness', 'simlex999', 'simverb3500', 'hyperlex', 'hyperlex-nouns', 'hyperlex_test']
    ae = WordSimilarity(datasets)
    if args.text:
        ae.vocab = vocab.stoi
        ae.emb1 = torch.tensor(emb).cuda()
        ae.emb2 = torch.tensor(emb).cuda()
    else:
        ae.load_vocab()
    ae.load_similarity()

elif args.hypernymy:

    model = HypernymySuiteModel()
    if args.text:
        model.vocab = vocab.stoi
        model.emb1 = torch.tensor(emb).cuda()
        model.emb2 = torch.tensor(emb).cuda()
    else:
        model.load_vocab()
    result = hypernymysuite_eval.all_evaluations(model, args)
    with open(args.output_file, 'w') as out_file:
        out_file.write(json.dumps(result))

elif args.neighbors:
    ae = RankedNeighbors()
    if args.text:
        ae.vocab = vocab
        ae.emb1 = torch.tensor(emb).cuda()
        ae.emb2 = torch.tensor(emb).cuda()
    else:
        ae.load_vocab()
    ae.dump_neighbors()
