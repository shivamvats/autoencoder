import nltk.tokenize
import codecs
import logging
import numpy as np

_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'[\w\$]+|[^\w\s]')


def get_logger(file_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)

    return logger


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time


def tokenize(text):
    tokens = _tokenizer.tokenize(text.lower())
    return tokens


class IterableSentences(object):
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        for line in codecs.open(self._filename, 'r', 'utf-8'):
            yield line.strip()

def one_hot(row, vocab_size):
    max_len = len(row)
    array = np.zeros((max_len, vocab_size+1))
    for i, ele in enumerate(row):
        array[i][ele] = 1
    return array

def de_one_hot(row):
    return np.nonzero(row)[0][0]

