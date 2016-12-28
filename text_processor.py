import codecs
import json
import os
from collections import Counter
from itertools import tee

from configs.config import VOCAB_MAX_SIZE
from utils import IterableSentences, tokenize, get_logger

EOS_SYMBOL = '$$$'
EMPTY_TOKEN = '###'

_logger = get_logger(__name__)


def get_tokens_voc(tokenized_sentences):
    """
    :param tokenized_sentences: generator for the efficient use of RAM
    """
    token_counter = Counter()

    for line in tokenized_sentences:
        for token in line:
            token_counter.update([token])

    token_voc = [token for token, _ in token_counter.most_common()[:VOCAB_MAX_SIZE]]
    token_voc.append(EMPTY_TOKEN)

    return set(token_voc)


def get_transformed_sentences(tokenized_sentences, tokens_voc):
    for line in tokenized_sentences:
        transformed_line = []

        for token in line:
            if token not in tokens_voc:
                token = EMPTY_TOKEN

            transformed_line.append(token)
        yield transformed_line


def get_tokenized_sentences(iterable_sentences):
    for line in iterable_sentences:
        tokenized_sentence= tokenize(line)
        tokenized_sentence.append(EOS_SYMBOL)
        yield tokenized_sentence


def get_tokenized_sentences_from_processed_corpus(iterable_sentences):
    for line in iterable_sentences:
        tokenized_sentence = line.strip().split()
        yield tokenized_sentence


def process_corpus(corpus_path):
    iterable_sentences = IterableSentences(corpus_path)

    tokenized_sentences = get_tokenized_sentences(iterable_sentences)
    tokenized_sentences_for_voc, tokenized_sentences_for_transform = tee(tokenized_sentences)

    tokens_voc = get_tokens_voc(tokenized_sentences_for_voc)
    transformed_sentences = get_transformed_sentences(tokenized_sentences_for_transform, tokens_voc)

    _logger.info('Token voc size = ' + str(len(tokens_voc)))
    index_to_token = dict(enumerate(tokens_voc))

    return transformed_sentences, index_to_token


def save_corpus(tokenized_dialog, processed_dialog_path):
    with codecs.open(processed_dialog_path, 'w', 'utf-8') as dialogs_fh:
        for tokenized_sentence in tokenized_dialog:
            sentence = ' '.join(tokenized_sentence)
            dialogs_fh.write(sentence + '\n')


def save_index_to_tokens(index_to_token, token_index_path):
    with codecs.open(token_index_path, 'w', 'utf-8') as token_index_fh:
        json.dump(index_to_token, token_index_fh, ensure_ascii=False)


def get_index_to_token(token_index_path):
    with codecs.open(token_index_path, 'r', 'utf-8') as token_index_fh:
        index_to_token = json.load(token_index_fh)
        index_to_token = {int(k): v for k, v in index_to_token.items()}

    return index_to_token


def get_processed_sentences_and_index_to_token(corpus_path, processed_corpus_path, token_index_path):
    _logger.info('Loading corpus data...')

    if os.path.isfile(processed_corpus_path) and os.path.isfile(token_index_path):
        _logger.info(processed_corpus_path + ' and ' + token_index_path + ' exist, loading files from disk')
        processed_sentences = IterableSentences(processed_corpus_path)
        processed_sentences = get_tokenized_sentences_from_processed_corpus(processed_sentences)
        index_to_token = get_index_to_token(token_index_path)
        return processed_sentences, index_to_token

    # continue here if processed corpus and token index are not stored on the disk
    _logger.info(processed_corpus_path + ' and ' + token_index_path + " don't exist, compute and save it")
    processed_sentences, index_to_token = process_corpus(corpus_path)
    processed_sentences, processed_sentences_for_save = tee(processed_sentences)

    save_index_to_tokens(index_to_token, token_index_path)
    save_corpus(processed_sentences_for_save, processed_corpus_path)

    return processed_sentences, index_to_token
