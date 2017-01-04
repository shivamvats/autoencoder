import gensim
import numpy as np
import pickle
import nltk
import copy

from config import UNKNOWN_TOKEN, ENDLINE_TOKEN
from config import TOKEN_REPRESENTATION_SIZE, PADDING_TOKEN, MAX_SEQ_LEN, MAX_VOCAB_SIZE, VALIDATION_SPLIT

import itertools

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk.tokenize import RegexpTokenizer

def load_data(filename):
    sentences = pickle.load(open(filename, "rb"))
    print("Loaded data")
    return sentences

def get_word_to_index_dic(w2v_model, sentences):
    token_list = []
    token_to_index_dic = {}
    for sentence in sentences:
        for word in sentence:
            if word in w2v_model.vocab:
                token_list.append(word)
    token_list.append(UNKNOWN_TOKEN)

    for ind, token in enumerate(token_list):
        token_to_index_dic[token] = ind
    token_to_index_dic[UNKNOWN_TOKEN] = len(token_list)
    token_to_index_dic[ENDLINE_TOKEN] = len(token_list) + 1

    return token_to_index_dic

def get_index_to_word_dic(word_to_index_dic):
    index_to_word_dic = {}
    for i, j in word_to_index_dic.items():
        index_to_word_dic[j] = i
    index_to_word_dic[0] = PADDING_TOKEN
    return index_to_word_dic

def get_token_vector(token, model):
    if token in model.vocab:
        return np.array(model[token])
    print("%s is not in vocab" % token)

    # return a zero vector for the words that are not presented in the model
    return np.zeros(TOKEN_REPRESENTATION_SIZE)


def get_vectorized_token_sequence(sequence, w2v_model, max_sequence_length, reverse=True):
    vectorized_token_sequence = np.zeros((max_sequence_length, TOKEN_REPRESENTATION_SIZE), dtype=np.float)

    for i, word in enumerate(sequence):
        vectorized_token_sequence[i] = get_token_vector(word, w2v_model)

    if reverse:
        # reverse token vectors order in input sequence as suggested here
        # http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
        vectorized_token_sequence = vectorized_token_sequence[::-1]

    return vectorized_token_sequence

def convert_word_to_embedding(model, word):
    embedding = model[word]
    return embedding

def convert_words_to_embedding(model, words):
    embedding = []
    for word in words:
        try:
            word_embedding = convert_word_to_embedding(word)
        except KeyError:
            word_embedding = np.zeros(model.vector_size, dtype=float)
        embedding.appen(word_embedding)
    return embedding

def tokenize_and_pad_sentences(sentences):
    tokenizer = Tokenizer(nb_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = np.asarray([np.asarray(sequence[:MAX_SEQ_LEN]) for sequence in sequences])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = np.asarray(pad_sequences(sequences, maxlen=MAX_SEQ_LEN))
    output= copy.copy(data)
    print('Shape of data tensor:', data.shape)

    return [data, output, word_index]

def get_train_val_test_data(input_data, output_data):
    # split the data into a training set and a validation set
    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    input_data = input_data[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * input_data.shape[0])

    x_train = input_data[:-nb_validation_samples]
    y_train = output_data[:-nb_validation_samples]
    x_test = input_data[-nb_validation_samples:]
    y_test = output_data[-nb_validation_samples:]

    x_val = x_test[:-20]
    y_val = y_test[:-20]
    x_test = x_test[-20:]
    y_test = y_test[-20:]

    return map(np.asarray, [x_train, y_train, x_val, y_val, x_test, y_test])

def preprocess_text(sentences):
    # Lower case
    lower_sequences = [ele.lower() for ele in sentences]

    tokenizer = RegexpTokenizer(r'\w+')
    processed_seqs = []
    for seq in lower_sequences:
        processed_seqs.append(" ".join(tokenizer.tokenize(seq)))

    return np.asarray(processed_seqs)

#def get_embeddings(w2v_model, sequences):
#    # At the point, all the sequences are of the same length.
#    # So we can input then to a fixes sized NN.
#    embedding = [get_vectorized_token_sequence(sequence, w2v_model,
#        TOKEN_REPRESENTATION_SIZE, False) for sequence in preprocessed_sequences]
#    return embedding
