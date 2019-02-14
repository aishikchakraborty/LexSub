import torch.nn as nn
import torch.nn.functional as F
import torch

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

        self.syn_proj = nn.Linear(ninp, wn_hid, bias=False)

        self.hypn_proj = nn.Linear(ninp, wn_hid, bias=False)
        self.hypn_rel = nn.Linear(wn_hid, wn_hid)

        self.mern_proj = nn.Linear(ninp, wn_hid, bias=False)
        self.mern_rel = nn.Linear(wn_hid, wn_hid)

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
        self.wn_hid = wn_hid

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adaptive:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, target, synonym, antonym, hypernym, meronym):
        emb = self.drop(self.encoder(input))
        emb_syn1 = self.syn_proj(self.encoder(synonym[:, 0]))
        emb_syn2 = self.syn_proj(self.encoder(synonym[:, 1]))

        emb_ant1 =self.syn_proj(self.encoder(antonym[:, 0]))
        emb_ant2 = self.syn_proj(self.encoder(antonym[:, 1]))

        emb_hypn1 = self.hypn_proj(self.encoder(hypernym[:, 0]))
        emb_hypn2 = self.hypn_proj(self.encoder(hypernym[:, 1]))
        emb_hypn1 = self.hypn_rel(emb_hypn1)

        emb_mern1 = self.mern_proj(self.encoder(meronym[:, 0]))
        emb_mern2 = self.mern_proj(self.encoder(meronym[:, 1]))
        emb_mern1 = self.mern_rel(emb_mern1)

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        if self.adaptive:
            decoded = self.adaptive_softmax.log_prob(output.view(output.size(0)*output.size(1), output.size(2)))
        else:
            decoded = F.log_softmax(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))), dim=-1)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        eye = torch.eye(self.wn_hid, device=self.syn_proj.weight.device)
        reg_loss =  torch.sum(\
                             torch.pow(torch.mm(self.syn_proj.weight, self.hypn_proj.weight.t()), 2) + \
                             torch.pow(torch.mm(self.syn_proj.weight, self.mern_proj.weight.t()), 2) + \
                             torch.pow(torch.mm(self.hypn_proj.weight, self.mern_proj.weight.t()), 2)) # + \
                             # torch.pow(torch.sum(self.syn_proj.weight**2) - 1, 2) + \
                             # torch.pow(torch.sum(self.hypn_proj.weight**2) - 1, 2) + \
                             # torch.pow(torch.sum(self.mern_proj.weight**2) - 1, 2)
        output_dict = {
                'log_probs': decoded,
                'hidden_vec': hidden,
                'syn_emb': (emb_syn1, emb_syn2),
                'ant_emb': (emb_ant1, emb_ant2),
                'hyp_emb': (emb_hypn1, emb_hypn2),
                'mer_emb': (emb_mern1, emb_mern2),
                'loss_reg': reg_loss
            }
        return output_dict

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, adaptive=False):
        super(RNNModel, self).__init__()
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


        self.criterion = nn.NLLLoss()

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adaptive:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, targets=None):
        output_dict={}
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        if self.adaptive:
            decoded = self.adaptive_softmax.log_prob(output.view(output.size(0)*output.size(1), output.size(2)))
        else:
            decoded = F.log_softmax(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))), dim=-1)

        output_dict['log_probs'] = decoded.view(output.size(0), output.size(1), decoded.size(1))
        output_dict['hidden_vec'] = hidden

        if targets is not None:
            output_dict['loss_lm'] = self.criterion(output_dict['log_probs'].view(-1, self.ntoken),
                                                    targets.view(-1))

        return output_dict

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class GloveEncoderModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, pretrained, dist_fn=F.pairwise_distance, dropout=0.5):
        super(GloveEncoderModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.glove_encoder = nn.Embedding(ntoken, ninp)
        self.glove_encoder.weight.data.copy_(pretrained)
        self.glove_encoder.requires_grad=False
        self.ntoken = ntoken
        self.dist_fn = dist_fn

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        output_dict={}
        emb = self.drop(self.encoder(input))
        emb_glove = self.drop(self.glove_encoder(input))
        output_dict['glove_emb'] = (emb, emb_glove)
        output_dict['glove_loss'] = torch.mean(self.dist_fn(emb, emb_glove))
        return output_dict

class WNModel(nn.Module):
    def __init__(self, embedding, emb_dim, wn_dim, pad_idx, antonym_margin=1, dist_fn=F.pairwise_distance, fixed=False, random=False):
        super(WNModel, self).__init__()
        self.embedding = embedding
        self.emb_dim = emb_dim
        self.wn_dim = wn_dim
        self.pad_idx = pad_idx

        self.syn_proj = nn.Linear(emb_dim, wn_dim, bias=False)
        self.hypn_proj = nn.Linear(emb_dim, wn_dim, bias=False)
        self.mern_proj = nn.Linear(emb_dim, wn_dim, bias=False)

        if fixed:
            self.syn_proj.weight.data.zero_()
            self.hypn_proj.weight.data.zero_()
            self.mern_proj.weight.data.zero_()

            eye = torch.eye(wn_dim)
            self.syn_proj.weight.data[:,0:wn_dim] = eye
            self.hypn_proj.weight.data[:,wn_dim:2*wn_dim] = eye
            self.mern_proj.weight.data[:, -wn_dim:] = eye


        if fixed or random:
            for param in self.syn_proj.parameters():
                param.requires_grad = False

            for param in self.hypn_proj.parameters():
                param.requires_grad = False

            for param in self.mern_proj.parameters():
                param.requires_grad = False

        self.hypn_rel = nn.Linear(wn_dim, wn_dim)
        self.mern_rel = nn.Linear(wn_dim, wn_dim)

        self.antonym_margin = antonym_margin

        self.dist_fn = dist_fn

    def forward(self, synonyms=None, antonyms=None, hypernyms=None, meronyms=None):
        output_dict = {}
        if synonyms is not None:
            emb_syn1 = self.syn_proj(self.embedding(synonyms[:, 0]))
            emb_syn2 = self.syn_proj(self.embedding(synonyms[:, 1]))
            syn_mask = 1 - (synonyms[:,0] == self.pad_idx).float()
            syn_len = torch.sum(syn_mask)
            output_dict['loss_syn'] = torch.sum(self.dist_fn(emb_syn1, emb_syn2) * syn_mask)/syn_len
            output_dict['syn_emb'] = (emb_syn1, emb_syn2)

        if antonyms is not None:
            emb_ant1 =self.syn_proj(self.embedding(antonyms[:, 0]))
            emb_ant2 = self.syn_proj(self.embedding(antonyms[:, 1]))
            ant_mask = 1 - (antonyms[:,0] == self.pad_idx).float()
            ant_len = torch.sum(ant_mask)
            output_dict['loss_ant'] = torch.sum(F.relu(self.antonym_margin - self.dist_fn(emb_ant1, emb_ant2)) * ant_mask)/ant_len
            output_dict['ant_emb'] = (emb_ant1, emb_ant2)

        if hypernyms is not None:
            emb_hypn1 = self.hypn_proj(self.embedding(hypernyms[:, 0]))
            emb_hypn2 = self.hypn_proj(self.embedding(hypernyms[:, 1]))
            emb_hypn1 = self.hypn_rel(emb_hypn1)
            hyp_mask = 1 - (hypernyms[:,0] == self.pad_idx).float()
            hyp_len = torch.sum(hyp_mask)
            output_dict['loss_hyp'] = torch.sum(self.dist_fn(emb_hypn1, emb_hypn2) * hyp_mask)/hyp_len
            output_dict['hyp_emb'] = (emb_hypn1, emb_hypn2)

        if meronyms is not None:
            emb_mern1 = self.mern_proj(self.embedding(meronyms[:, 0]))
            emb_mern2 = self.mern_proj(self.embedding(meronyms[:, 1]))
            emb_mern1 = self.mern_rel(emb_mern1)
            mer_mask = 1 - (meronyms[:,0] == self.pad_idx).float()
            mer_len = torch.sum(mer_mask)
            output_dict['loss_mer'] = torch.sum(self.dist_fn(emb_mern1, emb_mern2) * mer_mask)/mer_len
            output_dict['mer_emb'] = (emb_mern1, emb_mern2)

        return output_dict

class WNLM(nn.Module):
    def __init__(self, lm_module, wn_module):
        super(WNLM, self).__init__()
        self.lm = lm_module
        self.wn = wn_module
        self.encoder = self.lm.encoder
        self.rnn = self.lm.rnn

    def init_weights(self):
        self.lm.init_weights()
        self.wn.init_weights()

    def init_hidden(self, bsz):
        self.lm.init_hidden(bsz)

    def forward(self, input, hidden, targets=None, synonyms=None, antonyms=None, hypernyms=None, meronyms=None):
        lm_out_dict = self.lm(input, hidden, targets)
        wn_out_dict = self.wn(synonyms, antonyms, hypernyms, meronyms)

        # This merges two dictionaries and creates a single dict with fields from
        # both of them.
        return {**lm_out_dict, **wn_out_dict}

class GloveModel(nn.Module):
    def __init__(self, glove_module, wn_module):
        super(GloveModel, self).__init__()
        self.gl = glove_module
        self.wn = wn_module
        self.encoder = self.gl.encoder

    def init_weights(self):
        self.wn.init_weights()
        self.gl.init_weights()


    def forward(self, input, synonyms=None, antonyms=None, hypernyms=None, meronyms=None):
        gl_out_dict = self.gl(input)
        wn_out_dict = self.wn(synonyms, antonyms, hypernyms, meronyms)

        # This merges two dictionaries and creates a single dict with fields from
        # both of them.
        return {**gl_out_dict, **wn_out_dict}
