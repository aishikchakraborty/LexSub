import torch.nn as nn
import torch.nn.functional as F
import torch

from nce import IndexLinear
from torchqrnn import QRNN

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

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, vocab_freq,
                    dropout=0.5, cutoffs=[1000, 10000], tie_weights=False, adaptive=False,
                    proj_lm=False, lm_dim=None, fixed=False, random=False, nce=False, nce_loss='nce'):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if not proj_lm or lm_dim is None:
            lm_dim = ninp

        self.adaptive = adaptive
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(lm_dim, nhid, nlayers, dropout=dropout)
        elif rnn_type == 'QRNN':
            self.rnn = QRNN(lm_dim, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(lm_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)


        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != lm_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        if nce:
            def build_unigram_noise(freq):
                """build the unigram noise from a list of frequency
                Parameters:
                    freq: a tensor of #occurrences of the corresponding index
                Return:
                    unigram_noise: a torch.Tensor with size ntokens,
                    elements indicate the probability distribution
                """
                total = freq.sum()
                noise = freq / total
                assert abs(noise.sum() - 1) < 0.001
                return noise

            self.criterion = IndexLinear(nhid, ntoken,
                                noise=build_unigram_noise(vocab_freq),
                                noise_ratio=int(ntoken/10),
                                loss_type=nce_loss)
        else:
            if adaptive:
                self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=cutoffs)
            else:
                self.decoder = nn.Linear(nhid, ntoken)

            self.criterion = nn.NLLLoss()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.proj_lm = proj_lm
        self.fixed = fixed
        self.random = random
        self.lm_dim = lm_dim
        self.nce = nce

        if self.proj_lm:
            self.lm_proj = nn.Linear(ninp, lm_dim, bias=False)

            if fixed or random:
                for param in self.lm_proj.parameters():
                    param.requires_grad = False

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adaptive and not self.nce:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.proj_lm:
            if self.fixed:
                self.lm_proj.weight.data.zero_()
                self.lm_proj.weight.data[:, 0:self.lm_dim] = torch.eye(self.lm_dim)

    def forward(self, input, hidden, targets=None):
        output_dict={}
        emb = self.encoder(input)
        if self.proj_lm:
            emb = self.lm_proj(emb)
        emb = self.drop(emb)

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        output_dict['hidden_vec'] = hidden
        output_dict['log_probs'] = None
        if targets is not None and self.nce:
            output_dict['loss_lm'] = self.criterion(targets.view(-1, 1), output.unsqueeze(1))
        else:
            if self.adaptive:
                decoded = self.adaptive_softmax.log_prob(output.view(output.size(0)*output.size(1), output.size(2)))
            else:
                decoded = F.log_softmax(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))), dim=-1)

            output_dict['log_probs'] = decoded.view(output.size(0), output.size(1), decoded.size(1))

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

class CBOWModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, vocab_freq, cutoffs=[1000, 10000], adaptive=False,
                proj_lm=False, lm_dim=None, fixed=False, random=False, nce=False, nce_loss='nce'):
        super(CBOWModel, self).__init__()
        subsample_threshold = 1e-5
        self.encoder = nn.Embedding(ntoken, ninp)

        if not proj_lm or lm_dim is None:
            lm_dim = ninp



        self.ntoken = ntoken
        self.proj_lm = proj_lm
        self.fixed = fixed
        self.random = random
        self.lm_dim = lm_dim
        self.adaptive = adaptive
        self.nce = nce


        if self.proj_lm:
            self.lm_proj = nn.Linear(ninp, lm_dim, bias=False)

            if fixed or random:
                for param in self.lm_proj.parameters():
                    param.requires_grad = False

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adaptive and not self.nce:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.proj_lm:
            if self.fixed:
                self.lm_proj.weight.data.zero_()
                self.lm_proj.weight.data[:, 0:self.lm_dim] = torch.eye(self.lm_dim)

    def forward(self, input, hidden, targets=None):
        output_dict={}
        emb = self.encoder(input)
        if self.proj_lm:
            emb = self.lm_proj(emb)
        # print(emb.shape)

        output = torch.sum(emb, dim=0)

        output_dict['log_probs'] = None
        output_dict['hidden_vec'] = output

        if targets is not None and self.nce:
            output_dict['loss_lm'] = self.criterion(targets, output)
        else:
            if self.adaptive:
                decoded = self.adaptive_softmax.log_prob(output)
            else:
                decoded = F.log_softmax(self.decoder(output), dim=-1)

            output_dict['log_probs'] = decoded
            output_dict['hidden_vec'] = decoded

            if targets is not None:
                output_dict['loss_lm'] = self.criterion(output_dict['log_probs'].view(-1, self.ntoken),
                                                        targets.view(-1))

        return output_dict

    def init_hidden(self, bsz):
        pass


class SkipGramModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, vocab_freq, cutoffs=[1000, 10000], adaptive=False,
                proj_lm=False, lm_dim=None, fixed=False, random=False, nce=False, nce_loss='nce'):
        super(SkipGramModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Embedding(ntoken, ninp)

        self.weights = vocab_freq / vocab_freq.sum()
        self.weights = self.weights.pow(0.75)
        self.weights = self.weights/self.weights.sum()


        # if not proj_lm or lm_dim is None:
        #     lm_dim = ninp
        #
        #
        # if nce:
        #     def build_unigram_noise(freq):
        #         """build the unigram noise from a list of frequency
        #         Parameters:
        #             freq: a tensor of #occurrences of the corresponding index
        #         Return:
        #             unigram_noise: a torch.Tensor with size ntokens,
        #             elements indicate the probability distribution
        #         """
        #         total = freq.sum()
        #         noise = freq / total
        #         assert abs(noise.sum() - 1) < 0.001
        #         return noise
        #
        #     self.criterion = IndexLinear(lm_dim, ntoken,
        #                         noise=build_unigram_noise(vocab_freq),
        #                         noise_ratio=int(ntoken/100),
        #                         loss_type=nce_loss)
        # else:
        #     if adaptive:
        #         self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(lm_dim, ntoken, cutoffs=cutoffs)
        #     else:
        #         self.decoder = nn.Linear(lm_dim, ntoken)
        #     self.criterion = nn.NLLLoss()
        self.ninp = ninp
        self.ntoken = ntoken
        self.proj_lm = proj_lm
        self.fixed = fixed
        self.random = random
        self.lm_dim = lm_dim
        self.adaptive = adaptive
        self.nce = nce


        if self.proj_lm:
            self.lm_proj = nn.Linear(ninp, lm_dim, bias=False)

            if fixed or random:
                for param in self.lm_proj.parameters():
                    param.requires_grad = False

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # if not self.adaptive and not self.nce:
        #     self.decoder.bias.data.zero_()
        #     self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.proj_lm:
            if self.fixed:
                self.lm_proj.weight.data.zero_()
                self.lm_proj.weight.data[:, 0:self.lm_dim] = torch.eye(self.lm_dim)


    def forward(self, input, hidden, targets=None):
        output_dict={}
        n_negs = 20

        batch_size = input.size(1)
        emb_dim = self.ninp
        if targets is not None:
            context_size = targets.size(0)
        else:
            context_size = 8

        emb_input = self.encoder(input).view(batch_size, emb_dim, -1)
        nwords = torch.multinomial(self.weights, batch_size * context_size * n_negs, replacement=True).view(batch_size, -1).cuda()
        emb_output = self.decoder(targets).view(batch_size, context_size, -1)
        emb_nwords = self.decoder(nwords).view(batch_size, context_size*n_negs, -1).neg()
        # print(emb_input.shape)
        # print(emb_output.shape)
        # print(emb_nwords.shape)
        oloss = torch.bmm(emb_output, emb_input).squeeze().sigmoid().log().mean(1) #bsz, context_size
        nloss = torch.bmm(emb_nwords, emb_input).squeeze().sigmoid().log().view(-1, context_size, n_negs).sum(2).mean(1)

        output_dict['log_probs'] = emb_output
        output_dict['hidden_vec'] = emb_output
        output_dict['loss_lm'] = -(oloss + nloss).mean()
        output_dict['loss_ppl'] = -(oloss).mean()

        return output_dict

    def init_hidden(self, bsz):
        pass

class GloveEncoderModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, pretrained, dist_fn=F.pairwise_distance, dropout=0.5):
        super(GloveEncoderModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.glove_encoder = pretrained
        self.ntoken = ntoken
        self.dist_fn = dist_fn

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        output_dict={}
        emb = self.encoder(input)
        emb_glove = self.glove_encoder[input]
        emb, emb_glove = torch.squeeze(emb, 0), torch.squeeze(emb_glove, 0)
        output_dict['glove_emb'] = (emb, emb_glove)
        output_dict['glove_loss'] = torch.mean(F.mse_loss(emb, emb_glove))
        return output_dict

class WNModel(nn.Module):
    def __init__(self, lex_rels, vocab_freq, embedding, emb_dim, wn_dim, pad_idx, wn_offset=0, antonym_margin=1, dist_fn=F.pairwise_distance, fixed=False, random=False, num_neg_samples=10, common_vs=False):
        super(WNModel, self).__init__()

        if common_vs:
            wn_dim = emb_dim
            fixed=True

        self.embedding = embedding
        self.emb_dim = emb_dim
        self.wn_dim = wn_dim
        self.pad_idx = pad_idx
        self.lex_rels = lex_rels

        #creating a smoothed unigram distr. on vocab
        self.weights = vocab_freq / vocab_freq.sum()
        self.weights = self.weights.pow(0.75)
        self.weights = self.weights/self.weights.sum()
        self.n_negs = num_neg_samples

        if 'syn' in lex_rels:
            self.syn_proj = nn.Linear(emb_dim, wn_dim, bias=False)
            self.antonym_margin = antonym_margin

        if 'hyp' in lex_rels:
            self.hypn_proj = nn.Linear(emb_dim, wn_dim, bias=False)
            self.hypn_rel = nn.Linear(wn_dim, wn_dim)

        if 'mer' in lex_rels:
            self.mern_proj = nn.Linear(emb_dim, wn_dim, bias=False)
            self.mern_rel = nn.Linear(wn_dim, wn_dim)

        if fixed:
            eye = torch.eye(wn_dim)

            if 'syn' in lex_rels:
                self.syn_proj.weight.data.zero_()
                self.syn_proj.weight.data[:,wn_offset:wn_offset+wn_dim] = eye
                if not common_vs:
                    wn_offset += wn_dim

            if 'hyp' in lex_rels:
                self.hypn_proj.weight.data.zero_()
                self.hypn_proj.weight.data[:,wn_offset:wn_offset+wn_dim] = eye
                if not common_vs:
                    wn_offset += wn_dim

            if 'mer' in lex_rels:
                self.mern_proj.weight.data.zero_()
                self.mern_proj.weight.data[:, wn_offset:wn_offset+wn_dim] = eye
                if not common_vs:
                    wn_offset += wn_dim

        if fixed or random:
            if 'syn' in lex_rels:
                for param in self.syn_proj.parameters():
                    param.requires_grad = False

            if 'hyp' in lex_rels:
                for param in self.hypn_proj.parameters():
                    param.requires_grad = False

            if 'mer' in lex_rels:
                for param in self.mern_proj.parameters():
                    param.requires_grad = False

        self.dist_fn = dist_fn

    def forward(self, synonyms=None, antonyms=None, hypernyms=None, meronyms=None):
        output_dict = {}
        if 'syn' in self.lex_rels and synonyms is not None:
            emb_syn1 = self.syn_proj(self.embedding(synonyms[:, 0]))
            emb_syn2 = self.syn_proj(self.embedding(synonyms[:, 1]))
            syn_mask = 1 - (synonyms[:,0] == self.pad_idx).float()
            syn_len = torch.sum(syn_mask)
            # output_dict['loss_syn'] = torch.sum(self.dist_fn(emb_syn1, emb_syn2) * syn_mask)/max(syn_len, 1)
            batch_size = synonyms.size(0)
            nwords = torch.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1).cuda()
            emb_syn_neg = self.syn_proj(self.embedding(nwords.view(batch_size, self.n_negs)).view(-1, self.emb_dim)).view(batch_size, self.n_negs, -1)
            output_dict['loss_syn'] = torch.sum((self.dist_fn(emb_syn1, emb_syn2) \
                                        + F.relu(0.1 - self.dist_fn(emb_syn1.view(batch_size, 1, -1), emb_syn_neg, dim=2)).mean(1) \
                                        + F.relu(self.dist_fn(emb_syn1.view(batch_size, 1, -1), emb_syn_neg, dim=2) - 1.5).mean(1) \
                                    ) * syn_mask)/max(syn_len, 1)
            output_dict['syn_emb'] = (emb_syn1, emb_syn2)

        if 'syn' in self.lex_rels and antonyms is not None:
            emb_ant1 =self.syn_proj(self.embedding(antonyms[:, 0]))
            emb_ant2 = self.syn_proj(self.embedding(antonyms[:, 1]))
            ant_mask = 1 - (antonyms[:,0] == self.pad_idx).float()
            ant_len = torch.sum(ant_mask)
            output_dict['loss_ant'] = torch.sum(F.relu(self.antonym_margin - self.dist_fn(emb_ant1, emb_ant2)) * ant_mask)/max(ant_len, 1)
            output_dict['ant_emb'] = (emb_ant1, emb_ant2)

        if 'hyp' in self.lex_rels and hypernyms is not None:
            emb_hypn1 = self.hypn_proj(self.embedding(hypernyms[:, 0]))
            emb_hypn2 = self.hypn_proj(self.embedding(hypernyms[:, 1]))
            emb_hypn1 = self.hypn_rel(emb_hypn1)
            hyp_mask = 1 - (hypernyms[:,0] == self.pad_idx).float()
            hyp_len = torch.sum(hyp_mask)
            # output_dict['loss_hyp'] = torch.sum(self.dist_fn(emb_hypn1, emb_hypn2) * hyp_mask)/max(hyp_len, 1) + 0.000* (torch.norm(self.hypn_rel.weight) - 1)**2
            batch_size = hypernyms.size(0)
            nwords = torch.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1).cuda()
            emb_hyp_neg = self.hypn_rel(self.hypn_proj(self.embedding(nwords.view(batch_size, self.n_negs)).view(-1, self.emb_dim)).view(batch_size, self.n_negs, -1))

            output_dict['loss_hyp'] = torch.sum((self.dist_fn(emb_hypn1, emb_hypn2) \
                                        + 3.0 * F.relu(0.1 - self.dist_fn(emb_hypn2.view(batch_size, 1, -1), emb_hyp_neg, dim=2)).mean(1) \
                                    ) * hyp_mask)/max(hyp_len, 1)

            output_dict['hyp_emb'] = (emb_hypn1, emb_hypn2)

        if 'mer' in self.lex_rels and meronyms is not None:
            emb_mern1 = self.mern_proj(self.embedding(meronyms[:, 0]))
            emb_mern2 = self.mern_proj(self.embedding(meronyms[:, 1]))
            emb_mern1 = self.mern_rel(emb_mern1)
            mer_mask = 1 - (meronyms[:,0] == self.pad_idx).float()
            mer_len = torch.sum(mer_mask)
            # output_dict['loss_mer'] = torch.sum(self.dist_fn(emb_mern1, emb_mern2) * mer_mask)/max(mer_len, 1)
            batch_size = meronyms.size(0)
            nwords = torch.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1).cuda()
            emb_mer_neg = self.mern_rel(self.mern_proj(self.embedding(nwords.view(batch_size, self.n_negs)).view(-1, self.emb_dim)).view(batch_size, self.n_negs, -1))

            output_dict['loss_mer'] = torch.sum((self.dist_fn(emb_mern1, emb_mern2) \
                                        + F.relu(0.1 - self.dist_fn(emb_mern2.view(batch_size, 1, -1), emb_mer_neg, dim=2)).mean(1) \
                                    )* mer_mask)/max(mer_len, 1)
            output_dict['mer_emb'] = (emb_mern1, emb_mern2)

        return output_dict

class WNLM(nn.Module):
    def __init__(self, lm_module, wn_module):
        super(WNLM, self).__init__()
        self.lm = lm_module
        self.wn = wn_module
        self.encoder = self.lm.encoder

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
