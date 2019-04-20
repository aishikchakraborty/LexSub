import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--emb', type=str, default='../emb_Vanilla.pkl',
                    help='location of the trained embeddings')

args = parser.parse_args()

def read_text_file(file_path):
    print('Reading embedding files')
    emb = {}
    f = open(file_path, 'r')
    for lines in f:
        lines = lines.strip().split()
        emb[lines[0]] = [float(val) for val in lines[1:]]
    # vocab = Vocab(vocab)
    print('Finished Reading Embedding File')
    return emb

GLOVE_PATH = '../data/glove/glove.6B.300d.txt'

glove_emb = read_text_file(GLOVE_PATH)
emb = read_text_file(args.emb)
common_vocab  = set(glove_emb.keys()).intersection(set(emb.keys()))
# print(len(common_vocab))

glove_emb_common = np.array([glove_emb[v] for v in common_vocab])
emb_common = np.array([emb[v] for v in common_vocab])

# print(glove_emb_common.shape)
# print(emb_common.shape)
mean_shift = np.mean(np.sum(np.power(glove_emb_common - emb_common, 2), axis=1))
print(mean_shift)
