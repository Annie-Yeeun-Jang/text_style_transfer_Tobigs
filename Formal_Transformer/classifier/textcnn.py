
import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, embeding):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)
        if embeding is not None:
            self.embeding.weight.data = torch.FloatTensor(embeding)

    def forward(self, x):
        if len(x.size()) == 2:
            y = self.embeding(x)
        else:
            y = torch.matmul(x, self.embeding.weight)
        return y


class TextCNN(nn.Module):
    '''A style classifier TextCNN'''

    def __init__(self, embed_dim, vocab_size, filter_sizes,
                 num_filters, embedding=None, dropout=0.0):
        super(TextCNN, self).__init__()

        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim, embedding)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 2)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)

        return logit

    def build_embeder(self, vocab_size, embed_dim, embedding=None):
        embeder = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(embeder.weight, mean=0, std=embed_dim ** -0.5)
        if embedding is not None:
            embeder.weight.data = torch.FloatTensor(embedding)

        return embeder
