# coding: utf-8
import argparse
import hashlib
import math
import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import _pickle as pickle
from tensorboardX import SummaryWriter

import model

import csv
csv.field_size_limit(100000000)

from random import shuffle
from torchtext import data, datasets
import torchtext
import csv
csv.field_size_limit(100000000)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--annotated_dir', type=str, help='name of the directory with annontated data.')
parser.add_argument('--data_version', type=str, help='Version of Wordnet data to use.')
parser.add_argument('--model', type=str, default='rnn',
                    help='type of model. Options are [retro, rnn, cbow]')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--lex', '-l', action="append", type=str, default=[], dest='lex_rels',
                    help='list of type of lexical relations to capture. Options | syn | hyp | mer')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--wn_hid', type=int, default=100,
                    help='Dimension of the WN subspace')
parser.add_argument('--margin', type=int, default=2,
                    help='define the margin for the max-margin loss')
parser.add_argument('--patience', type=int, default=1,
                    help='How long before you reduce the LR.')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=14,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--random_seed', type=int, default=13370,
                    help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337,
                    help='numpy random seed')
parser.add_argument('--ss_t', type=float, default=1e-5,
                    help="subsample threshold")
parser.add_argument('--torch_seed', type=int, default=133,
                    help='pytorch random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu x')
parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/',
                    help='path to save the final model')
parser.add_argument('--save-emb', type=str, default='embeddings/',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--adaptive', action='store_true',
                    help='Use adaptive softmax. This speeds up computation.')
parser.add_argument('--wn_ratio', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--distance', type=str, default='cosine',
                    help='Type of distance to use. Options are [pairwise, cosine]')
parser.add_argument('--optim', type=str, default='sgd',
                    help='Type of optimizer to use. Options are [sgd, adagrad, adam]')
parser.add_argument('--reg', action='store_true', help='Regularize.')
parser.add_argument('--fixed_wn', action='store_true', help='Fixed WN proj matrices to identity matrix.')
parser.add_argument('--random_wn', action='store_true', help='Fix random WN proj matrix and not learn it.')
parser.add_argument('--common_vs', action='store_true', help='Collapse subspaces to original vs for fair comparison with other techniques.')
parser.add_argument('--lower', action='store_true', help='Lowercase for data.')
parser.add_argument('--extend_wn', action='store_true', help='This flag allows the final embedding to be concatenation of wn embedding and lm embedding.')
parser.add_argument('--nce', action='store_true', help='Use nce for training.')
parser.add_argument('--nce_loss', type=str, default='nce', help='Type of nce to use.')
parser.add_argument('--num_neg_sample_subspace', type=int, default=10,
                    help='Number of negative samples to use while training lexical subspace.')
parser.add_argument('--max_vocab_size', type=int, default=None,
                    help='Vocab size to use for the dataset.')
parser.add_argument('--retro_emb_data_dir', type=str, default='data/glove',
                    help='Number of negative samples to use while training lexical subspace.')
parser.add_argument('--retro_emb_file', type=str, default='glove.6B.300d.txt',
                    help='Number of negative samples to use while training lexical subspace.')
args = parser.parse_args()

print(args)
if args.random_seed is not None:
    random.seed(args.random_seed)
if args.numpy_seed is not None:
    np.random.seed(args.numpy_seed)
if args.torch_seed is not None:
    torch.manual_seed(args.torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

class Dataset(data.TabularDataset):
    def __init__(self, dataset, fields):
       super(Dataset, self).__init__(dataset, 'json', fields=fields)

    @classmethod
    def splits(cls, fields, dataset_dir=None, train_file=None, valid_file=None, test_file=None, **kwargs):
        if dataset_dir:
            train_file = train_file or os.path.join(dataset_dir, 'train.txt')
            valid_file = valid_file or os.path.join(dataset_dir, 'valid.txt')
            test_file = test_file or os.path.join(dataset_dir, 'test.txt')

        return (cls(train_file, fields, **kwargs),
                cls(valid_file, fields, **kwargs),
                cls(test_file, fields, **kwargs))

    @classmethod
    def iters(cls, dataset_dir=None, train_file=None, valid_file=None, test_file=None,
                device=-1, batch_size=args.batch_size, load_from_file=False, version=1,
                retro_emb_data_dir='data/glove', retro_emb_file='glove.6B.300d.txt', **kwargs):

        def preprocessing(prop_list):
            if len(prop_list) == 0:
                return ['<pad>']
            return prop_list

        TEXT_FIELD = data.Field(batch_first=False, include_lengths=False, lower=args.lower)
        # WORDNET_TEXT_FIELD = data.Field(fix_length=2, lower=args.lower)
        WORDNET_TEXT_FIELD = data.Field(preprocessing=preprocessing, lower=args.lower)
        field_dict = {
                'text': ('text', TEXT_FIELD),
                'target': ('target', TEXT_FIELD),
                # 'synonyms': ('synonyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing)),
                # 'antonyms': ('antonyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing)),
                # 'hypernyms': ('hypernyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing)),
                # 'meronyms': ('meronyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing))
                 'synonyms_a': ('synonyms_a', WORDNET_TEXT_FIELD),
                 'synonyms_b': ('synonyms_b', WORDNET_TEXT_FIELD),
                 'antonyms_a': ('antonyms_a', WORDNET_TEXT_FIELD),
                 'antonyms_b': ('antonyms_b', WORDNET_TEXT_FIELD),
                 'hypernyms_a': ('hypernyms_a', WORDNET_TEXT_FIELD),
                 'hypernyms_b': ('hypernyms_b', WORDNET_TEXT_FIELD),
                 'meronyms_a': ('meronyms_a', WORDNET_TEXT_FIELD),
                 'meronyms_b': ('meronyms_b', WORDNET_TEXT_FIELD)
                }
        suffix = hashlib.md5('{}-{}-{}-{}-{}-lower_{}'.format(version, dataset_dir,
                                                     train_file, valid_file, test_file, args.lower)
                                            .encode()).hexdigest()

        examples_path = os.path.join(dataset_dir, '{}.pkl'.format(suffix))

        save_iters = False
        if not load_from_file:
            try:
                examples = torch.load(examples_path)
            except:
                load_from_file = True
                save_iters = True

        if load_from_file:
                print('Loading from file')
                dataset = cls.splits(field_dict, dataset_dir, train_file, valid_file, test_file, **kwargs)
                if save_iters:
                    torch.save([d.examples for d in dataset], examples_path)

        if not load_from_file:
            dataset = [data.Dataset(ex, field_dict.values()) for ex in examples]

        train, valid, test = dataset

        if args.model == 'retro':
            vec = torchtext.vocab.Vectors(retro_emb_file, cache=retro_emb_data_dir)
            TEXT_FIELD.build_vocab(train, vectors=vec, max_size=args.max_vocab_size)
        else:
            TEXT_FIELD.build_vocab(train, max_size=args.max_vocab_size)
        WORDNET_TEXT_FIELD.vocab = TEXT_FIELD.vocab

        if args.model == 'rnn':
            train_iter, valid_iter, test_iter = data.Iterator.splits((train, valid, test),
                                                    batch_size=batch_size, device=device,
                                                    shuffle=False, repeat=False, sort=False)
        else:
            print('Using Bucket Iterator')
            train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test),
                                                    batch_size=batch_size, device=device,
                                                    shuffle=bool(args.model != 'rnn'), repeat=False, sort=False)

        return train_iter, valid_iter, test_iter, TEXT_FIELD.vocab, TEXT_FIELD.vocab.vectors


def dist_fn(x1, x2, dim=1):
    if args.distance == 'cosine':
        return  1 - F.cosine_similarity(x1,x2, dim=dim)
    elif args.distance == 'pairwise':
        return F.pairwise_distance(x1, x2)**2/x1.size(-1)

data_dir = './data/' + args.data

annotated_data_dir = args.annotated_dir or 'annotated_{}_{}_{}'.format(args.model, args.bptt, args.batch_size) if args.model == 'rnn' else \
                    'annotated_{}'.format(args.model)

if args.data_version:
    annotated_data_dir += '_v{}'.format(args.data_version)


lex_rels = '_'.join(args.lex_rels) if len(args.lex_rels) > 0 else 'vanilla'
summary_filename = os.path.join(args.save, 'logs_' + args.data + '_' + args.model + '_' + lex_rels + '_' + str(args.emsize) + '_' + str(args.nhid) + '_' + str(args.wn_hid) + '_' + args.distance)

os.system('rm -rf ' + summary_filename)
os.mkdir(summary_filename)
writer = SummaryWriter(summary_filename)

train_iter, valid_iter, test_iter, vocab, pretrained = Dataset.iters(dataset_dir=os.path.join(data_dir, annotated_data_dir), device=device, retro_emb_data_dir=args.retro_emb_data_dir, retro_emb_file=args.retro_emb_file)

# This is the default WikiText2 iterator from TorchText.
# Using this to compare our iterator. Will delete later.
# train_iter, valid_iter, test_iter = datasets.WikiText2.iters(batch_size=args.batch_size, bptt_len=args.bptt,
#                                                              device=device, root=args.data)
# vocab = train_iter.dataset.fields['text'].vocab
if args.model != 'rnn':
    train_iter = [x for x in train_iter]

# valid_iter = [x for x in valid_iter]
# test_iter = [x for x in test_iter]

print('Loaded batches')
ntokens = len(vocab)
pad_idx = vocab.stoi['<pad>']


lr = args.lr
best_val_loss = None

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

cutoffs = [100, 1000, 5000] if args.data == 'wikitext-2' else [2800, 10000, 30000, 76000]

idx2freq = [0] * ntokens
for key, value in vocab.freqs.items():
    idx2freq[vocab.stoi[key]] = value
idx2freq = torch.tensor(idx2freq, dtype=torch.float)

if args.model == 'rnn':
    wn_offset = args.emsize if args.extend_wn else 0
    em_dim = args.emsize + len(args.lex_rels) * args.wn_hid if args.extend_wn else args.emsize

    lm_model = model.RNNModel(args.rnn_type, ntokens, em_dim, args.nhid, args.nlayers, idx2freq,
                              args.dropout, cutoffs=cutoffs, tie_weights=args.tied, adaptive=args.adaptive,
                              proj_lm=args.extend_wn, lm_dim=args.emsize,
                              fixed=args.fixed_wn, random=args.random_wn, nce=args.nce, nce_loss=args.nce_loss).to(device)
    wn_model = model.WNModel(args.lex_rels, idx2freq, lm_model.encoder, em_dim, args.wn_hid, pad_idx,
                             wn_offset=wn_offset,
                             antonym_margin=args.margin,
                             fixed=args.fixed_wn,
                             random=args.random_wn,
                             dist_fn=dist_fn,
                             num_neg_samples=args.num_neg_sample_subspace).to(device)
    model = model.WNLM(lm_model, wn_model).to(device)
elif args.model == 'retro':
    gl_model = model.GloveEncoderModel(ntokens, args.emsize, pretrained.to(device), dist_fn=dist_fn).to(device)
    wn_model = model.WNModel(args.lex_rels, idx2freq, gl_model.encoder, args.emsize, args.wn_hid, pad_idx,
                             wn_offset=0,
                             antonym_margin=args.margin,
                             fixed=args.fixed_wn,
                             random=args.random_wn,
                             dist_fn=dist_fn,
                             num_neg_samples=args.num_neg_sample_subspace,
                             common_vs=args.common_vs).to(device)
    model = model.GloveModel(gl_model, wn_model).to(device)
elif args.model == 'cbow':
    wn_offset = args.emsize if args.extend_wn else 0
    em_dim = args.emsize + len(args.lex_rels) * args.wn_hid if args.extend_wn else args.emsize

    lm_model = model.CBOWModel(ntokens, em_dim, idx2freq, cutoffs=cutoffs, adaptive=args.adaptive,
                              proj_lm=args.extend_wn, lm_dim=args.emsize,
                              fixed=args.fixed_wn, random=args.random_wn, nce=args.nce, nce_loss=args.nce_loss).to(device)
    wn_model = model.WNModel(args.lex_rels, idx2freq, lm_model.encoder, em_dim, args.wn_hid, pad_idx,
                             wn_offset=wn_offset,
                             antonym_margin=args.margin,
                             fixed=args.fixed_wn,
                             random=args.random_wn,
                             dist_fn=dist_fn,
                             num_neg_samples=args.num_neg_sample_subspace).to(device)

    model = model.WNLM(lm_model, wn_model).to(device)
elif args.model == 'skipgram':
    wn_offset = args.emsize if args.extend_wn else 0
    em_dim = args.emsize + len(args.lex_rels) * args.wn_hid if args.extend_wn else args.emsize

    lm_model = model.SkipGramModel(ntokens, em_dim, idx2freq, cutoffs=cutoffs, adaptive=args.adaptive,
                              proj_lm=args.extend_wn, lm_dim=args.emsize,
                              fixed=args.fixed_wn, random=args.random_wn, nce=args.nce, nce_loss=args.nce_loss).to(device)
    wn_model = model.WNModel(args.lex_rels, idx2freq, lm_model.encoder, em_dim, args.wn_hid, pad_idx,
                             wn_offset=wn_offset,
                             antonym_margin=args.margin,
                             fixed=args.fixed_wn,
                             random=args.random_wn,
                             dist_fn=dist_fn,
                             num_neg_samples=args.num_neg_sample_subspace).to(device)

    model = model.WNLM(lm_model, wn_model).to(device)
else:
    raise ValueError('Illegal model type: %s. Options are [rnn, cbow, retro]' % args.model)

#model = model.RNNWordnetModel(args.rnn_type, ntokens, args.emsize, args.nhid, args.nlayers, args.wn_hid, args.dropout, args.tied, args.adaptive, cutoffs).to(device)

criterion = nn.NLLLoss()

optimizer = torch.optim.Adagrad(model.parameters(), lr=lr) if args.optim == 'adagrad' \
                else torch.optim.Adam(model.parameters(), lr=lr) if args.optim == 'adam' \
                else torch.optim.SGD(model.parameters(), lr=lr)

milestones=[100] if args.optim != 'sgd' else \
            ([3,6,7] if args.data == 'wikitext-103' else \
                [10, 15, 25, 35]  if args.data == 'wikitext-2' else [2, 5, 10, 25])
print(milestones)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)

print('Lex Rel List: {}'.format(args.lex_rels))
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'rnn' and args.nce:
        model.lm.criterion.loss_type='full'

    total_loss = 0.
    total_loss_syn = 0.
    total_loss_ant = 0.
    total_loss_hyp = 0.
    total_loss_mern = 0.

    if args.model != 'retro':
        hidden = model.init_hidden(args.batch_size)

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data, targets = batch.text, batch.target
            # synonyms, antonyms, hypernyms, meronyms = batch.synonyms, batch.antonyms, batch.hypernyms, batch.meronyms
            synonyms_a, synonyms_b, antonyms_a, antonyms_b, hypernyms_a, hypernyms_b, meronyms_a, meronyms_b = batch.synonyms_a, batch.synonyms_b, batch.antonyms_a, batch.antonyms_b, batch.hypernyms_a, batch.hypernyms_b, batch.meronyms_a, batch.meronyms_b
            synonyms = torch.stack((synonyms_a, synonyms_b), dim=2).view(-1, 2)
            antonyms = torch.stack((antonyms_a, antonyms_b), dim=2).view(-1, 2)
            hypernyms = torch.stack((hypernyms_a, hypernyms_b), dim=2).view(-1, 2)
            meronyms = torch.stack((meronyms_a, meronyms_b), dim=2).view(-1, 2)

            if args.model == 'retro':
                output_dict = model(data, synonyms, antonyms, hypernyms, meronyms)
                emb, emb_glove = output_dict['glove_emb']
                loss = output_dict.get('glove_loss',
                                        torch.mean(dist_fn(emb, emb_glove)))
            else:
                output_dict = model(data, hidden, targets, synonyms, antonyms, hypernyms, meronyms)
                output, hidden = output_dict['log_probs'], output_dict['hidden_vec']
                hidden = repackage_hidden(hidden)

                if 'loss_lm' in output_dict:
                    loss = output_dict['loss_lm']
                else:
                    loss = criterion(output.view(-1, ntokens), targets.view(-1))
            # if args.model == 'skipgram':
            #     loss = output_dict['loss_ppl']
            total_loss += loss
            if 'syn' in args.lex_rels:
                emb_syn1, emb_syn2 = output_dict['syn_emb']
                syn_mask = 1 - (synonyms[:,0] == pad_idx).float()
                syn_len = torch.sum(syn_mask)
                loss_syn = output_dict.get('loss_syn',
                                            torch.sum(dist_fn(emb_syn1, emb_syn2) * syn_mask)/syn_len)
                total_loss_syn += loss_syn

                emb_ant1, emb_ant2 = output_dict['ant_emb']
                ant_mask = 1 - (antonyms[:,0] == pad_idx).float()
                ant_len = torch.sum(ant_mask)
                loss_ant = output_dict.get('loss_ant',
                                            torch.sum(F.relu(args.margin - dist_fn(emb_ant1, emb_ant2)) * ant_mask)/ant_len)
                total_loss_ant += loss_ant

            if 'hyp' in args.lex_rels:
                if 'loss_hyp' in output_dict:
                    loss_hyp = output_dict['loss_hyp']
                else:
                    emb_hyp1, emb_hyp2 = output_dict['hyp_emb']
                    hyp_mask = 1 - (hypernyms[:,0] == pad_idx).float()
                    hyp_len = torch.sum(hyp_mask)
                    loss_hyp = torch.sum(dist_fn(emb_hyp1, emb_hyp2) * hyp_mask)/hyp_len

                total_loss_hyp += loss_hyp

            if 'mer' in args.lex_rels:
                if 'loss_mer' in output_dict:
                    loss_mer = output_dict['loss_mer']
                else:
                    emb_mern1, emb_mern2 = output_dict['mer_emb']
                    mer_mask = 1 - (meronyms[:,0] == pad_idx).float()
                    mer_len = torch.sum(mer_mask)
                    loss_mer = torch.sum(dist_fn(emb_mern1, emb_mern2) * mer_mask)/mer_len

                total_loss_mern += loss_mer

    if args.model == 'rnn' and args.nce:
        model.lm.criterion.loss_type = args.nce_loss

    return total_loss/(len(data_source) - 1), total_loss_syn/(len(data_source) - 1), total_loss_ant/(len(data_source) - 1), \
            total_loss_hyp/ (len(data_source) - 1), total_loss_mern/(len(data_source) - 1)


def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss_ = 0.
    total_loss_hyp = 0.
    total_loss_syn = 0.
    total_loss_ant = 0.
    total_loss_mern = 0.
    total_loss_reg = 0.
    start_time = time.time()
    if args.model != 'retro':
        hidden = model.init_hidden(args.batch_size)

    if args.model != 'rnn':
        shuffle(train_iter)

    for idx, batch in enumerate(train_iter):
        data, targets = batch.text, batch.target
        # synonyms, antonyms, hypernyms, meronyms = batch.synonyms, batch.antonyms, batch.hypernyms, batch.meronyms
        synonyms_a, synonyms_b, antonyms_a, antonyms_b, hypernyms_a, hypernyms_b, meronyms_a, meronyms_b = batch.synonyms_a, batch.synonyms_b, batch.antonyms_a, batch.antonyms_b, batch.hypernyms_a, batch.hypernyms_b, batch.meronyms_a, batch.meronyms_b
        synonyms = torch.stack((synonyms_a, synonyms_b), dim=2).view(-1, 2)
        antonyms = torch.stack((antonyms_a, antonyms_b), dim=2).view(-1, 2)
        hypernyms = torch.stack((hypernyms_a, hypernyms_b), dim=2).view(-1, 2)
        meronyms = torch.stack((meronyms_a, meronyms_b), dim=2).view(-1, 2)

        optimizer.zero_grad()
        wn_ratio = args.wn_ratio

        if args.model == 'retro':
            output_dict = model(data, synonyms, antonyms, hypernyms, meronyms)
            emb, emb_glove = output_dict['glove_emb']
            loss = output_dict.get('glove_loss',
                                    torch.mean(dist_fn(emb, emb_glove)))
        else:
            output_dict = model(data, hidden, targets, synonyms, antonyms, hypernyms, meronyms)

            output, hidden = output_dict['log_probs'], output_dict['hidden_vec']
            hidden = repackage_hidden(hidden)

            if 'loss_lm' in output_dict:
                loss = output_dict['loss_lm']
            else:
                loss = criterion(output.view(-1, ntokens), targets.view(-1))

        total_loss = loss


        if 'syn' in args.lex_rels:
            emb_syn1, emb_syn2 = output_dict['syn_emb']
            syn_mask = 1 - (synonyms[:,0] == pad_idx).float()
            syn_len = torch.sum(syn_mask)
            loss_syn = output_dict.get('loss_syn',
                                        torch.sum(dist_fn(emb_syn1, emb_syn2) * syn_mask)/syn_len)

            emb_ant1, emb_ant2 = output_dict['ant_emb']
            ant_mask = 1 - (antonyms[:,0] == pad_idx).float()
            ant_len = torch.sum(ant_mask)
            loss_ant = output_dict.get('loss_ant',
                                        torch.sum(F.relu(args.margin - dist_fn(emb_ant1, emb_ant2)) * ant_mask)/ant_len)

            total_loss += wn_ratio * (loss_syn + loss_ant)
            total_loss_syn += loss_syn.item()
            total_loss_ant += loss_ant.item()

        if 'hyp' in args.lex_rels:
            if 'loss_hyp' in output_dict:
                loss_hyp = output_dict['loss_hyp']
            else:
                emb_hyp1, emb_hyp2 = output_dict['hyp_emb']
                hyp_mask = 1 - (hypernyms[:,0] == pad_idx).float()
                hyp_len = torch.sum(hyp_mask)
                loss_hyp = torch.sum(dist_fn(emb_hyp1, emb_hyp2) * hyp_mask)/hyp_len

            total_loss += wn_ratio * loss_hyp
            total_loss_hyp += loss_hyp.item()

        if 'mer' in args.lex_rels:
            if 'loss_mer' in output_dict:
                loss_mer = output_dict['loss_mer']
            else:
                emb_mern1, emb_mern2 = output_dict['mer_emb']
                mer_mask = 1 - (meronyms[:,0] == pad_idx).float()
                mer_len = torch.sum(mer_mask)
                loss_mer = torch.sum(dist_fn(emb_mern1, emb_mern2) * mer_mask)/mer_len

            total_loss += wn_ratio * loss_mer
            total_loss_mern += loss_mer.item()

        if args.reg:
            reg_loss = output_dict.get('reg_loss', 0)
            total_loss_reg = reg_loss
            total_loss += reg_loss

        total_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        # if args.model == 'skipgram':
        #     loss = output_dict['loss_ppl']
        total_loss_ += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss_ / args.log_interval
            curr_syn_loss = total_loss_syn / args.log_interval
            curr_ant_loss = total_loss_ant / args.log_interval
            curr_hyp_loss = total_loss_hyp / args.log_interval
            curr_mern_loss = total_loss_mern / args.log_interval
            curr_reg_loss = total_loss_reg / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | syn loss {:5.2f} | ant loss {:5.2f} | hyp loss {:5.2f} | mer loss {:5.2f} | reg_loss {:5.2f}'
                    .format(epoch, idx, len(train_iter), optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                        cur_loss, math.exp(min(cur_loss, 10)), curr_syn_loss, curr_ant_loss, curr_hyp_loss, curr_mern_loss, curr_reg_loss))
            global_step = epoch*args.batch_size + idx
            writer.add_scalar('Train/LMLoss', cur_loss, global_step)
            writer.add_scalar('Train/SynLoss', curr_syn_loss, global_step)
            writer.add_scalar('Train/AntLoss', curr_ant_loss, global_step)
            writer.add_scalar('Train/HypLoss', curr_hyp_loss, global_step)
            writer.add_scalar('Train/MernLoss', curr_mern_loss, global_step)

            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)

            start_time = time.time()
            total_loss_ = 0
            total_loss_syn = 0
            total_loss_ant = 0
            total_loss_hyp = 0
            total_loss_mern = 0

    print()

patience = 0

model_name = os.path.join(args.save, 'model_' + args.data + '_' + args.model + '_' + lex_rels + '_' + str(args.emsize) + '_' + str(args.nhid) + '_' + str(args.wn_hid) + '_' + args.distance + ('_wn_v{}'.format(args.data_version) if args.data_version else '') + '.pt')
emb_name = os.path.join(args.save_emb, 'emb_' + args.data + '_' + args.model + '_' + lex_rels + '_' + str(args.emsize) + '_' + str(args.nhid) + '_' + str(args.wn_hid) + '_' + args.distance + ('_wn_v{}'.format(args.data_version) if args.data_version else '') + '.pkl')
emb_name_txt = os.path.join(args.save_emb, 'emb_' + args.data + '_' + args.model + '_' + lex_rels + '_' + str(args.emsize) + '_' + str(args.nhid) + '_' + str(args.wn_hid) + '_' + args.distance + ('_wn_v{}'.format(args.data_version) if args.data_version else '') + '.txt')

rel_emb_name_temp = os.path.join(args.save_emb, 'emb_%s_' + args.data + '_' + args.model + '_' + lex_rels + '_' + str(args.emsize) + '_' + str(args.nhid) + '_' + str(args.wn_hid) + '_' + args.distance + ('_wn_v{}'.format(args.data_version) if args.data_version else '') + '.pkl')

vocab_name = os.path.join(args.save, 'vocab_' + args.data + '.pkl')
pickle.dump(vocab, open(vocab_name, 'wb'))
print('Vocab Saved')

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)

        if args.model != 'retro':
            val_loss, loss_syn, loss_ant, loss_hyp, loss_mer = evaluate(valid_iter)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | syn loss {:5.2f} | ant loss {:5.2f} | hyp loss {:5.2f} | mer loss {:5.2f}'
                        .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss),
                                        loss_syn, loss_ant, loss_hyp, loss_mer))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_name, 'wb') as f:
                    torch.save(model, f)
                print('Saving learnt embeddings : %s' % emb_name)
                pickle.dump(model.encoder.weight.data, open(emb_name, 'wb'))

                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            scheduler.step()
            if False and patience > 3:
                break
        elif (epoch == args.epochs) or (epoch % 10 == 0):
            print('Saving Model')
            with open(model_name, 'wb') as f:
                torch.save(model, f)
            print('Saving learnt embeddings : %s' % emb_name)
            pickle.dump(model.encoder.weight.data, open(emb_name, 'wb'))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(model_name, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if args.model=='rnn' and args.rnn_type != 'QRNN':
        model.lm.rnn.flatten_parameters()


# Run on test data.
if args.model != 'retro':
    test_loss, test_syn, test_ant, test_hyp, test_mer = evaluate(test_iter)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | syn loss {:5.2f} | ant loss {:5.2f} | hyp loss {:5.2f} | mer loss {:5.2f}'.format(
        test_loss, math.exp(test_loss), test_syn, test_ant, test_hyp, test_mer))
    print('=' * 89)

print('Saving final learnt embeddings ')
pickle.dump(model.encoder.weight.data, open(emb_name, 'wb'))
with open(emb_name_txt, 'w') as f:
    final_emb = model.encoder.weight.data.cpu().numpy()
    for i in range(final_emb.shape[0]):
        f.write(vocab.itos[i] + ' ')
        f.write(' '.join([str(x) for x in final_emb[i, :]]) + '\n')

for rel in args.lex_rels:
    rel_emb_name = rel_emb_name_temp % rel
    print('Saving lexical subspace embeddings : %s' % rel_emb_name)
    if rel == 'syn':
        rel_emb = model.wn.syn_proj(model.wn.embedding.weight)
        pickle.dump(rel_emb.data, open(rel_emb_name, 'wb'))
    elif rel == 'hyp':
        rel_emb_1 = model.wn.hypn_proj(model.wn.embedding.weight)
        pickle.dump(rel_emb_1.data, open(rel_emb_name_temp % ('hypn_hypernyms'), 'wb'))
        rel_emb_2 = model.wn.hypn_rel(rel_emb_1)
        pickle.dump(rel_emb_2.data, open(rel_emb_name_temp % ('hypn_hyponyms'), 'wb'))

    elif rel == 'mer':
        rel_emb_1 = model.wn.mern_proj(model.wn.embedding.weight)
        pickle.dump(rel_emb_1.data, open(rel_emb_name_temp % ('mern_meronyms'), 'wb'))
        rel_emb_2 = model.wn.mern_rel(rel_emb_1)
        pickle.dump(rel_emb_2.data, open(rel_emb_name_temp % ('mern_holonyms'), 'wb'))
