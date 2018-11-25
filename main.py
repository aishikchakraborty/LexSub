# coding: utf-8
import argparse
import hashlib
import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.onnx

# This should no longer be needed.
import data as d

import model

from torchtext import data, datasets

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

class Dataset(data.TabularDataset):
    def __init__(self, dataset, fields):
       super(Dataset, self).__init__(dataset, 'tsv', fields=fields)

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
                gpu=-1, batch_size=args.batch_size, load_from_file=False, version=1, **kwargs):

        def preprocessing(prop_list):
            return [x.split(',') for x in prop_list]

        TEXT_FIELD = data.Field(batch_first=True)
        fields = [
                ('text', TEXT_FIELD),
                ('synonyms', data.NestedField(TEXT_FIELD, preprocessing=preprocessing)),
                ('antonyms', data.NestedField(TEXT_FIELD, preprocessing=preprocessing))
                ]

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
                dataset = cls.splits(fields, dataset_dir, train_file, valid_file, test_file, **kwargs)
                if save_iters:
                    torch.save([d.examples for d in dataset], examples_path)

        if not load_from_file:
            dataset = [data.Dataset(ex, fields) for ex in examples]

        train, valid, test = dataset
        TEXT_FIELD.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test),
                                                    batch_size=args.batch_size, shuffle=True, repeat=False, sort=False,
                                                    device=gpu, sort_key= lambda x : len(x.context))
        return train_iter, valid_iter, test_iter, TEXT_FIELD.vocab

train_iter, valid_iter, test_iter, vocab = Dataset.iters(dataset_dir='./data/wikitext-2/annotated/')

ntokens = len(vocab)
print ntokens
pad_idx = vocab.stoi['<pad>']

lr = args.lr
best_val_loss = None

# How to use these iters.
for batch in train_iter:
    # (batch_size, seq_lens)
    text_idxs = batch.text
    print text_idxs.size()
    # (batch_size, max_num_synonyms, 2)
    synonyms = batch.synonyms
    print synonyms.size()
    # (batch_size, max_num_antonyms, 2)
    antonyms = batch.antonyms

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
criterion = nn.CrossEntropyLoss()

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for batch in data_source:
            data = batch.text
            pad_data = np.ones((data.size(0), 1))*pad_idx
            targets = torch.cat((data[:, 1:], torch.LongTensor(pad_data)), 1)

            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            # hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    i = 0
    for batch in train_iter:
        i += 1
        data = batch.text
        print data
        synonyms = batch.synonyms
        antonyms = batch.antonyms
        pad_data = np.ones((data.size(0), 1))*pad_idx
        targets = torch.cat((data[:, 1:], torch.LongTensor(pad_data)), 1)

        data = data.transpose(1, 0).to(device)
        targets = targets.view(-1).to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)

        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()


        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


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
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
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

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


# TODO: Following portion of code should be deleted.
###############################################################################
# Load data
###############################################################################

corpus = d.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)



###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


# def train():
#     # Turn on training mode which enables dropout.
#     model.train()
#     total_loss = 0.
#     start_time = time.time()
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(args.batch_size)
#     for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
#         data, targets = get_batch(train_data, i)
#         # Starting each batch, we detach the hidden state from how it was previously produced.
#         # If we didn't, the model would try backpropagating all the way to start of the dataset.
#         hidden = repackage_hidden(hidden)
#         model.zero_grad()
#         output, hidden = model(data, hidden)
#         loss = criterion(output.view(-1, ntokens), targets)
#         loss.backward()
#
#         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#         for p in model.parameters():
#             p.data.add_(-lr, p.grad.data)
#
#         total_loss += loss.item()
#
#         if batch % args.log_interval == 0 and batch > 0:
#             cur_loss = total_loss / args.log_interval
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
#                     'loss {:5.2f} | ppl {:8.2f}'.format(
#                 epoch, batch, len(train_data) // args.bptt, lr,
#                 elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
#             total_loss = 0
#             start_time = time.time()
#
#
# def export_onnx(path, batch_size, seq_len):
#     print('The model is also exported in ONNX format at {}'.
#           format(os.path.realpath(args.onnx_export)))
#     model.eval()
#     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
#     hidden = model.init_hidden(batch_size)
#     torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.

# At any point you can hit Ctrl + C to break out of training early.
