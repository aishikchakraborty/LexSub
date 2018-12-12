# coding: utf-8
import argparse
import hashlib
import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import _pickle as pickle
# from tqdm import tqdm

import model

import csv
csv.field_size_limit(100000000)

from torchtext import data, datasets

import csv
csv.field_size_limit(100000000)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--mdl', type=str, default='Vanilla',
                    help='type of model Vanilla | syn | hyp')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--margin', type=int, default=1,
                    help='define the margin for the max-margin loss')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu x')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/',
                    help='path to save the final model')
parser.add_argument('--save-emb', type=str, default='embeddings/',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
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
                device=-1, batch_size=args.batch_size, load_from_file=False, version=1, **kwargs):

        def preprocessing(prop_list):
            if len(prop_list) == 0:
                return ['<pad>', '<pad>']
            return [x.split(',') for x in prop_list]

        TEXT_FIELD = data.Field(batch_first=False, include_lengths=False)
        WORDNET_TEXT_FIELD = data.Field(fix_length=2)
        field_dict = {
                'text': ('text', TEXT_FIELD),
                'target': ('target', TEXT_FIELD),
                'synonyms': ('synonyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing)),
                'antonyms': ('antonyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing)),
                'hypernyms': ('hypernyms', data.NestedField(WORDNET_TEXT_FIELD, preprocessing=preprocessing))
                }

        suffix = hashlib.md5('{}-{}-{}-{}-{}'.format(version, dataset_dir,
                                                     train_file, valid_file, test_file)
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
                dataset = cls.splits(field_dict, dataset_dir, train_file, valid_file, test_file, **kwargs)
                if save_iters:
                    torch.save([d.examples for d in dataset], examples_path)

        if not load_from_file:
            dataset = [data.Dataset(ex, field_dict.values()) for ex in examples]

        train, valid, test = dataset
        TEXT_FIELD.build_vocab(train)
        WORDNET_TEXT_FIELD.vocab = TEXT_FIELD.vocab

        train_iter, valid_iter, test_iter = data.Iterator.splits((train, valid, test),
                                                batch_size=batch_size, device=device,
                                                shuffle=False, repeat=False, sort=False)
        return train_iter, valid_iter, test_iter, TEXT_FIELD.vocab

train_iter, valid_iter, test_iter, vocab = Dataset.iters(dataset_dir=os.path.join('./data', args.data, 'annotated'), device=device)

# This is the default WikiText2 iterator from TorchText.
# Using this to compare our iterator. Will delete later.
# train_iter, valid_iter, test_iter = datasets.WikiText2.iters(batch_size=args.batch_size, bptt_len=args.bptt,
                                                             # device=device, root=args.data)
# vocab = train_iter.dataset.fields['text'].vocab

ntokens = len(vocab)
pad_idx = vocab.stoi['<pad>']
pickle.dump(vocab, open('vocab_' + str(args.data) + '.pkl', 'wb'))
print('Vocab Saved')

lr = args.lr
best_val_loss = None

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
model = model.RNNWordnetModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(data_source):


            data, targets = batch.text, batch.target
            synonyms, antonyms, hypernyms = batch.synonyms, batch.antonyms, batch.hypernyms

            targets = targets.view(-1)

            mask = 1 - (targets == pad_idx).float()
            # output, hidden = model(data, hidden)
            output, emb_syn1, emb_syn2, emb_ant1, emb_ant2, emb_hyp1, emb_hyp2, hidden = model(data, hidden, synonyms, antonyms, hypernyms)

            hidden = repackage_hidden(hidden)
            total_loss += (torch.sum(criterion(output.view(-1, ntokens), targets) * mask)/torch.sum(mask)).item()
    return total_loss / (len(data_source) - 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, verbose=True, factor=0.1)
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss_ = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for idx, batch in enumerate(train_iter):
        data, targets = batch.text, batch.target
        synonyms, antonyms, hypernyms = batch.synonyms, batch.antonyms, batch.hypernyms
        synonyms = synonyms.view(-1, 2)
        antonyms = antonyms.view(-1, 2)
        hypernyms = hypernyms.view(-1, 2)
        targets = targets.view(-1)

        mask = 1 - (targets == pad_idx).float()
        optimizer.zero_grad()

        # output, hidden = model(data, hidden)
        output, emb_syn1, emb_syn2, emb_ant1, emb_ant2, emb_hyp1, emb_hyp2, hidden = model(data, hidden, synonyms, antonyms, hypernyms)

        output = output.view(-1, ntokens)

        hidden = repackage_hidden(hidden)
        loss = torch.sum(criterion(output, targets) * mask)/torch.sum(mask)
        if args.mdl == 'Vanilla':
            total_loss = loss
        elif args.mdl == 'syn':
            loss_syn = torch.mean(torch.sum(torch.pow(emb_syn1 - emb_syn2, 2), dim=-1))
            loss_hyp = torch.mean(torch.sum(torch.pow(emb_hyp1 - emb_hyp2, 2), dim=-1))
            loss_ant = torch.abs(args.margin - torch.mean(torch.sum(torch.pow(emb_ant1 - emb_ant2, 2), dim=-1)))
            total_loss = loss + loss_syn + loss_ant
        elif args.mdl == 'syn+hyp':
            loss_syn = torch.mean(torch.sum(torch.pow(emb_syn1 - emb_syn2, 2), dim=-1))
            loss_hyp = torch.mean(torch.sum(torch.pow(emb_hyp1 - emb_hyp2, 2), dim=-1))
            loss_ant = torch.abs(args.margin - torch.mean(torch.sum(torch.pow(emb_ant1 - emb_ant2, 2), dim=-1)))
            total_loss = loss + loss_syn + loss_ant + loss_hyp
        total_loss.backward()


        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss_ += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss_ / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, idx, len(train_iter), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            start_time = time.time()
            total_loss_ = 0

    print()

patience = 0
model_name = os.path.join(args.save, 'model_' + args.data + '_' + args.mdl + '.pt')
emb_name = os.path.join(args.save_emb, 'emb_' + args.data + '_' + args.mdl + '_' + str(args.emsize) + '.pkl')
emb_name_txt = os.path.join(args.save_emb, 'emb_' + args.data + '_' + args.mdl + '_' + str(args.emsize) + '.txt')
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_iter)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_name, 'wb') as f:
                torch.save(model, f)
            print('Saving learnt embeddings ')
            pickle.dump(model.encoder.weight.data, open(emb_name, 'wb'))

            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        scheduler.step(val_loss)
        if patience > 3:
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(model_name, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()


# Run on test data.
test_loss = evaluate(test_iter)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
print('Saving final learnt embeddings ')
pickle.dump(model.encoder.weight.data, open(emb_name, 'wb'))
with open(emb_name_txt, 'w') as f:
    final_emb = model.encoder.weight.data.cpu().numpy()
    for i in range(final_emb.shape[0]):
        f.write(vocab.itos[i] + ' ')
        f.write(' '.join([str(x) for x in final_emb[i, :]]) + '\n')

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
