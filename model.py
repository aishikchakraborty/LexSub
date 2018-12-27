import torch.nn as nn
import torch.nn.functional as F

class RNNWordnetModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, wn_hid, dropout=0.5, tie_weights=False, adaptive=False, cutoffs=[1000, 10000]):
        super(RNNWordnetModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.adaptive = adaptive
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        if adaptive:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=cutoffs)
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        self.wn_proj = nn.Linear(ninp, wn_hid)

        self.hypernym_proj = nn.Linear(ninp, nhid)
        self.inner_hyp_proj = nn.Linear(nhid, nhid)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp or adaptive:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adaptive:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, synonym, antonym, hypernym):
        emb = self.drop(self.encoder(input))
        emb_syn1 = self.wn_proj(self.encoder(synonym[:, 0]))
        emb_syn2 = self.wn_proj(self.encoder(synonym[:, 1]))

        emb_ant1 =self.wn_proj(self.encoder(antonym[:, 0]))
        emb_ant2 = self.wn_proj(self.encoder(antonym[:, 1]))


        emb_hyp1 = self.wn_proj(self.encoder(hypernym[:, 0]))
        emb_hyp2 = self.wn_proj(self.encoder(hypernym[:, 1]))

        emb_hyp1 = self.inner_hyp_proj(emb_hyp1)

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        if self.adaptive:
            decoded = self.adaptive_softmax.log_prob(output.view(output.size(0)*output.size(1), output.size(2)))
        else:
            decoded = F.log_softmax(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))), dim=-1)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, emb_syn1, emb_syn2, emb_ant1, emb_ant2, emb_hyp1, emb_hyp2, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # print output.size()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
