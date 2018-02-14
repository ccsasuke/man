import numpy as np
import torch
import torch.nn as nn

from options import opt


class Vocab:
    def __init__(self, txt_file):
        with open(txt_file, 'r') as inf:
            parts = inf.readline().split()
            assert len(parts) == 2
            self.vocab_size, self.emb_size = int(parts[0]), int(parts[1])
            opt.vocab_size = self.vocab_size
            opt.emb_size = self.emb_size
            # add an UNK token
            self.unk_tok = '<unk>'
            self.unk_idx = 0
            self.vocab_size += 1
            self.v2wvocab = ['<unk>']
            self.w2vvocab = {'<unk>': 0}
            self.embeddings = np.empty((self.vocab_size, self.emb_size), dtype=np.float)
            cnt = 1
            for line in inf.readlines():
                parts = line.rstrip().split(' ')
                word = parts[0]
                # add to vocab
                self.v2wvocab.append(word)
                self.w2vvocab[word] = cnt
                # load vector
                vector = [float(x) for x in parts[-self.emb_size:]]
                self.embeddings[cnt] = vector
                cnt += 1

        self.eos_tok = '</s>'
        opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
        # randomly initialize <unk> vector
        self.embeddings[self.unk_idx] = np.random.normal(0, 1, size=self.emb_size)
        # normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1,1)
        # zero </s>
        self.embeddings[self.eos_idx] = 0

    def base_form(word):
        return word.strip().lower()

    def init_embed_layer(self):
        word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.eos_idx)
        if not opt.random_emb:
            word_emb.weight.data = torch.from_numpy(self.embeddings).float()
        return word_emb

    def lookup(self, word):
        word = Vocab.base_form(word)
        if word in self.w2vvocab:
            return self.w2vvocab[word]
        return self.unk_idx

    def prepare_inputs(self, dataset):
        for x in dataset.X:
            x['inputs'] = [self.lookup(w) for w in x['tokens']]
            
