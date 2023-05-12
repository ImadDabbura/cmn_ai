from typing import Iterable, Sequence

from .utils import uniqueify


class Processor:
    def process(self, item):
        return item


class CategoryProcessor(Processor):
    def __init__(self):
        self.vocab = None

    def __call__(self, items: Sequence):
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {k: v for v, k in enumerate(self.vocab)}
        return self.process(items)

    def process(self, items):
        if isinstance(items, Iterable):
            return [self.otoi[item] for item in items]
        return self.otoi[items]

    def deprocess(self, idxs):
        if isinstance(idxs, Iterable):
            return [self.vocab[idx] for idx in idxs]
        return self.vocab[idxs]
