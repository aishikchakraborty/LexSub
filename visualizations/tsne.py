import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import argparse
import csv
import json
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import _pickle as pickle
from tensorboardX import SummaryWriter

# import matplotlib
# matplotlib.use('Agg')
#
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
#
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
# RS = 123

# python visualizations/tsne.py --emb output/wikitext-2_skipgram_syn_lower/2019_03_21_23_49/emb_wikitext-2_skipgram_syn_300_300_100_cosine_wn_v2.pkl --emb2 output/wikitext-2_skipgram_syn_lower/2019_03_21_23_49/emb_wikitext-2_skipgram_syn_300_300_100_cosine_wn_v2.pkl --vocab output/wikitext-2_skipgram_syn_lower/2019_03_21_23_49/vocab_wikitext-2.pkl
# from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--emb', type=str, default='../emb_Vanilla.pkl',
                    help='location of the trained embeddings')
parser.add_argument('--emb2', type=str, help='location of second embedding. This is only different from emb1 in case of hyp and mer subspace embedding.')
parser.add_argument('--vocab', type=str, default='../vocab_wikitext-103.pkl',
                    help='location of the vocab')
parser.add_argument('--top-k', type=int, default=10,
                    help='number of neighbours')
parser.add_argument('--random-samples', type=int, default=10,
                    help='number of random samples to consider')

args = parser.parse_args()
np.random.seed(100)
summary_filename = 'visualizations/logs/tsne'
import os; os.system('rm -rf ' + summary_filename);
writer = SummaryWriter(summary_filename)

vocab = pickle.load(open(args.vocab, 'rb'))
word_indices = np.random.choice(len(vocab), args.random_samples, replace=False)

emb1 = pickle.load(open(args.emb, 'rb'))
emb2 = pickle.load(open(args.emb2, 'rb'))
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

all_embs = np.zeros((args.top_k * (args.random_samples+1), emb1.shape[1]))
word_label = []
print([vocab.itos[x] for x in word_indices])
for i, x in enumerate(word_indices):
    wemb = emb1[x].view(1, -1)
    all_embs[i, :] = wemb.cpu().numpy()
    neighbors = []
    word_label.append(vocab.itos[x])
    scores = F.cosine_similarity(wemb, emb2, dim=1)
    scores[x] = -float("inf")
    score_topk, topk = torch.topk(scores, k=args.top_k)
    all_embs[i*args.top_k+1:i*args.top_k+args.top_k+1, :] = emb2[topk].cpu().numpy()
    nn_words = [vocab.itos[w] for w in topk]
    word_label.extend(nn_words)

writer.add_embedding(torch.tensor(all_embs), metadata=word_label)

# print(all_embs.shape)
# tsne_results = tsne.fit_transform(all_embs)
# print(tsne_results.shape)
# cluster_no = np.array(cluster_no)
# f, ax, sc, txts = fashion_scatter(tsne_results, cluster_no)
# f.savefig('visualizations/test1.png')
