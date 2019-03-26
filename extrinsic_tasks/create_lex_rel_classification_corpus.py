import random

syn_v2 = set([])
hyp_v2 = set([])
mer_v2 = set([])
ant_v2 = set([])
hyperlex = set([])
datasets = set([])

for i, line in enumerate(open('../preprocessing/syn_v2.txt')):
    syn_v2.add(tuple(line.split()))

for i, line in enumerate(open('../preprocessing/ant_v2.txt')):
    ant_v2.add(tuple(line.split()))

for i, line in enumerate(open('../preprocessing/hyp_v2.txt')):
    hyp_v2.add(tuple(line.split()))

for i, line in enumerate(open('../preprocessing/mer_v2.txt')):
    mer_v2.add(tuple(line.split()))

for i, line in enumerate(open('../extrinsic_tasks/datasets/hyperlex/hyperlex-all.txt')):
    if i == 0:
        continue
    row = line.strip().split()
    word_pair = (row[0], row[1])

    if row[3] not in ['syn', 'hyp-1', 'mero', 'ant']:
        continue

    for x in [syn_v2, ant_v2, hyp_v2, mer_v2]:
        if word_pair in x:
            x.remove(word_pair)

    if row[3] == 'hyp-1':
        row[3] = 'hyp'

    hyperlex.add((row[0], row[1], row[3]))

count={}
for x in hyperlex:
    if x[2] not in count:
        count[x[2]] = 0
    count[x[2]] += 1


min_len = min([len(x) for x in [syn_v2, ant_v2, hyp_v2, mer_v2]])

for x in [('syn', syn_v2), ('hyp', hyp_v2), ('mero', mer_v2), ('ant', ant_v2)]:

    label = x[0]
    for y in x[1]:
        datasets.add((y[0], y[1], label))

valid = set([])
for x in [('syn', syn_v2), ('hyp', hyp_v2), ('mero', mer_v2), ('ant', ant_v2)]:
    for y in random.sample(x[1], int(len(x[1])*0.2)):
        tup=(y[0], y[1], x[0])
        valid.add(tup)
        datasets.remove(tup)

with open('../extrinsic_tasks/datasets/lex_rel_train.txt', 'w') as train_file, \
        open('../extrinsic_tasks/datasets/lex_rel_valid.txt', 'w') as valid_file, \
        open('../extrinsic_tasks/datasets/lex_rel_test.txt', 'w') as test_file:

    for d in datasets:
        train_file.write('%s %s %s\n' % d)

    for d in valid:
        valid_file.write('%s %s %s\n' % d)

    for d in hyperlex:
        test_file.write('%s %s %s\n' % d)


import pdb;pdb.set_trace()
pass
