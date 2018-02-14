import torch
from torch import autograd, nn
import torch.nn.functional as functional

import utils

class AveragingLayer(nn.Module):
    def __init__(self, word_emb):
        super(AveragingLayer, self).__init__()
        self.word_emb = word_emb

    def forward(self, input):
        """
        input: (data, lengths): (IntTensor(batch_size, max_sent_len), IntTensor(batch_size))
        """
        data, lengths = input
        data = autograd.Variable(data)
        lengths = autograd.Variable(lengths)
        embeds = self.word_emb(data)
        X = embeds.sum(1).squeeze(1)
        lengths = lengths.view(-1, 1).expand_as(X)
        return torch.div(X, lengths.float())


class SummingLayer(nn.Module):
    def __init__(self, word_emb):
        super(SummingLayer, self).__init__()
        self.word_emb = word_emb

    def forward(self, input):
        """
        input: (data, lengths): (IntTensor(batch_size, max_sent_len), IntTensor(batch_size))
        """
        data, _ = input
        data = autograd.Variable(data)
        embeds = self.word_emb(data)
        X = embeds.sum(1).squeeze()
        return X


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
        mask = autograd.Variable((idxes<lengths.unsqueeze(1)).float())

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output
        

