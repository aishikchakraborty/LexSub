import _pickle as pickle
import numpy as np

vocab = []
emb = []
f = open('../../data/glove/glove.6B.300d.txt', 'r')
for lines in f:
    lines = lines.strip().split()
    vocab.append(lines[0])
    emb.append([float(val) for val in lines[1:]])

emb = np.array(emb)
print(emb.shape)

f1 = open('../../data/glove/vocab.txt', 'w')
for w in vocab:
    f1.write(w + '\n')
pickle.dump(emb, open('../../data/glove/emb.pb', 'wb'))
