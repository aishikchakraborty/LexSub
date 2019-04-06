import _pickle as pickle
import numpy as np
import sys

vocab = []
emb = []
f = open(sys.argv[1], 'r')
for lines in f:
    lines = lines.strip().split()
    vocab.append(lines[0])
    emb.append([float(val) for val in lines[1:]])

emb = np.array(emb)
print(emb.shape)

f1 = open(sys.argv[2], 'w')
for w in vocab:
    f1.write(w + '\n')
