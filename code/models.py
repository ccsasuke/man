import torch
import torch.nn.functional as functional
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers import *
from options import opt


class DanFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 sum_pooling,
                 dropout,
                 batch_norm=False):
        super(DanFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.hidden_size = hidden_size

        if sum_pooling:
            self.avg = SummingLayer(self.word_emb)
        else:
            self.avg = AveragingLayer(self.word_emb)
        
        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(vocab.emb_size, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), opt.act_unit)

    def forward(self, input):
        return self.fcnet(self.avg(input))


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
        
        # at least 1 hidden layer so that the output size is hidden_size
        assert num_layers > 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            # if batch_norm:
            #     self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), opt.act_unit)

    def forward(self, input):
        data, _ = input
        data = autograd.Variable(data)
        batch_size = len(data)
        embeds = self.word_emb(data)

        # conv
        embeds = embeds.unsqueeze(1) # batch_size, 1, seq_len, emb_size
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 bdrnn,
                 attn_type):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.bdrnn = bdrnn
        self.attn_type = attn_type
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        self.n_cells = self.num_layers*2 if bdrnn else self.num_layers
        
        self.word_emb = vocab.init_embed_layer()
        self.rnn = nn.LSTM(input_size=vocab.emb_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        if attn_type == 'dot':
            self.attn = DotAttentionLayer(hidden_size)

    def forward(self, input):
        data, lengths = input
        data = autograd.Variable(data)
        lengths_list = lengths.tolist()
        batch_size = len(data)
        embeds = self.word_emb(data)
        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = autograd.Variable(embeds.data.new(*state_shape))
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        if self.attn_type == 'last':
            return ht[-1] if not self.bdrnn \
                          else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.attn_type == 'avg':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return torch.sum(unpacked_output, 1) / autograd.Variable(lengths.float()).view(-1, 1)
        elif self.attn_type == 'dot':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return self.attn((unpacked_output, lengths))
        else:
            raise Exception('Please specify valid attention (pooling) mechanism')


class MlpFeatureExtractor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(MlpFeatureExtractor, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential()
        num_layers = len(hidden_sizes)
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                self.net.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.net.add_module('f-relu-{}'.format(i), opt.act_unit)

        if dropout > 0:
            self.net.add_module('f-dropout-final', nn.Dropout(p=dropout))
        self.net.add_module('f-linear-final', nn.Linear(hidden_sizes[-1], output_size))
        if batch_norm:
            self.net.add_module('f-bn-final', nn.BatchNorm1d(output_size))
        self.net.add_module('f-relu-final', opt.act_unit)

    def forward(self, input):
        return self.net(input)

class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class DomainClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_domains,
                 loss_type,
                 dropout,
                 batch_norm=False):
        super(DomainClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_domains = num_domains
        self.loss_type = loss_type
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        if loss_type.lower() == 'gr' or loss_type.lower() == 'bs':
            self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        scores = self.net(input)
        if self.loss_type.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores
