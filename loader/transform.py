import math
import re
import string

from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch


class UniformSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            sampled = np.stack(frames)
        else:
            indices = np.linspace(0, n_frames - 1, self.n_sample, dtype=int)
            sampled = np.stack([frames[i] for i in indices])
        return torch.from_numpy(sampled)


class RandomSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            sampled = np.stack(frames)
        else:
            block_len = n_frames // self.n_sample
            start = n_frames - block_len - 1
            base_idx = np.linspace(0, start, self.n_sample, dtype=int)
            noise = np.random.choice(block_len, self.n_sample, replace=True)
            indices = base_idx + noise
            sampled = np.stack([frames[i] for i in indices])
        return torch.from_numpy(sampled)


class TrimIfLongerThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        if len(frames) > self.n:
            frames = frames[:self.n]
        return frames


class ZeroPadIfLessThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        while len(frames) < self.n:
            frames = np.vstack([ frames, np.zeros_like(frames[0]) ])
        return frames


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, array):
        np_array = np.asarray(array)
        t = torch.from_numpy(np_array)
        if self.dtype:
            t = t.type(self.dtype)
        return t


class NLTKWordpunctTokenizer:

    def __call__(self, sentence):
        return wordpunct_tokenize(sentence)


class TrimExceptAscii:
    def __init__(self, corpus):
        self.corpus = corpus

    def __call__(self, sentence):
        if self.corpus == "MSVD":
            if isinstance(sentence, list):
                return sentence
            else:
                # s = sentence.encode('ascii', 'ignore').encode('ascii')
                s = sentence.encode('ascii', 'ignore')
        elif self.corpus == "MSR-VTT":
            if isinstance(sentence, list):
                return sentence
            else:
                s = sentence.encode('ascii', 'ignore')
                # s = sentence
        return s


class RemovePunctuation:
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, sentence):
        return self.regex.sub('', sentence.decode('ascii'))


class Lowercase:

    def __call__(self, sentence):
        return sentence.lower()


class SplitWithWhiteSpace:

    def __call__(self, sentence):
        return sentence.split()


class Truncate:
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]


class PadFirst:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return [ self.token ] + words


class PadLast:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return words + [ self.token ]


class PadToLength:
    def __init__(self, token, length):
        self.token = token
        self.length = length

    def __call__(self, words):
        n_pads = self.length - len(words)
        return words + [ self.token ] * n_pads


class ToIndex:
    def __init__(self, word2idx):
        self.word2idx = word2idx

    def __call__(self, words):  # Ignore unknown (or trimmed) words.
        # word_to_idx = []
        # for word in words:
        #     if word in self.word2idx:
        #         word_to_idx.append(self.word2idx[word])
        #     else:
        #         word_to_idx.append(self.word2idx['<UNK>'])
        # return word_to_idx
        return [ self.word2idx[word] for word in words if word in self.word2idx ]

