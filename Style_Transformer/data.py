import time
import numpy as np
import torchtext
from torchtext.legacy import data
from embeddings import GloveEmbedding
from utils import tensor2text


class DatasetIterator(object):
    def __init__(self, native_iter, nonnative_iter):
        self.native_iter = native_iter
        self.nonnative_iter = nonnative_iter

    def __iter__(self):
        for batch_native, batch_nonnative in zip(iter(self.native_iter), iter(self.nonnative_iter)):
            if batch_native.text.size(0) == batch_nonnative.text.size(0):
                yield batch_native.text, batch_nonnative.text


def load_dataset(config, train_native='native_train2.csv', train_nonnative='nonnative_train2.csv',
                 dev_native='native_train2.csv', dev_nonnative='nonnative_train2.csv',
                 test_native='native_train2.csv', test_nonnative='nonnative_train2.csv'):
    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>', fix_length = config.max_length)

    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_native_set, train_nonnative_set = map(dataset_fn, [train_native, train_nonnative])
    dev_native_set, dev_nonnative_set = map(dataset_fn, [dev_native, dev_nonnative])
    test_native_set, test_nonnative_set = map(dataset_fn, [test_native, test_nonnative])

    TEXT.build_vocab(train_native_set, train_nonnative_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()

        vectors = torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())

        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text), # 비슷한 길이의 문장으로만 배치 구성 (padding 최소화)
        sort_within_batch=False,
        device=config.device
    )

    train_native_iter, train_nonnative_iter = map(lambda x: dataiter_fn(x, True),
                                                  [train_native_set, train_nonnative_set])
    dev_native_iter, dev_nonnative_iter = map(lambda x: dataiter_fn(x, False), [dev_native_set, dev_nonnative_set])
    test_native_iter, test_nonnative_iter = map(lambda x: dataiter_fn(x, False), [test_native_set, test_nonnative_set])

    train_iters = DatasetIterator(train_native_iter, train_nonnative_iter)
    dev_iters = DatasetIterator(dev_native_iter, dev_nonnative_iter)
    test_iters = DatasetIterator(test_native_iter, test_nonnative_iter)

    return train_iters, dev_iters, test_iters, vocab


if __name__ == '__main__':
    train_iter, _, _, vocab = load_dataset('../data/dataset/')
    print(len(vocab))
    for batch in train_iter:
        text = tensor2text(vocab, batch.text)
        print('\n'.join(text))
        print(batch.label)
        break