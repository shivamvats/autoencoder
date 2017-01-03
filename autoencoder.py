import numpy as np
import gensim

from keras.layers import Input, LSTM, RepeatVector, Activation, Dense, TimeDistributed, Embedding
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from seq2seq import Seq2Seq, AttentionSeq2Seq

from config import ACTOR_BATCH_SIZE, TIME_STEPS, MAX_SEQ_LEN, TOKEN_REPRESENTATION_SIZE, ACTOR_NUM_EPOCHS

from text_preprocessing import get_word_to_index_dic, get_index_to_word_dic, load_data, tokenize_sentences, get_train_val_test_data, preprocess_text

class Autoencoder(object):
    def __init__(self, w2v_model, token_to_index_dic):
        self.w2v_model = w2v_model
        self.token_to_index_dic = token_to_index_dic
        self.data_vocab_size = len(token_to_index_dic)
        self.embedding_matrix = None
        print("Data vocab size= %d" % self.data_vocab_size)

    def create_embedding(self):
        """Crates an embedding matrix whose ith row corresponds to the
        embedding of the ith token in dictionary."""
        self.embedding_matrix = np.zeros((len(self.token_to_index_dic) + 1, TOKEN_REPRESENTATION_SIZE))
        for word, i in self.token_to_index_dic.items():
            if word in self.w2v_model.vocab:
                embedding_vector = np.array(self.w2v_model[word])
            else:
                embedding_vector = np.zeros(TOKEN_REPRESENTATION_SIZE)

            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def create_nn_model(self):
        self.create_embedding()
        #input_seq = Input(shape=(MAX_SEQ_LEN,))

        # This converts the positive indices(integers) into a dense multi-dim representation.
        model = Sequential()
        model.add(Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
                            weights=[self.embedding_matrix], trainable=False, input_shape=(MAX_SEQ_LEN,)))
        model.add(AttentionSeq2Seq(batch_input_shape=(None, MAX_SEQ_LEN, TOKEN_REPRESENTATION_SIZE),
                            #hidden_dim=200, output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE, 
                            #depth=MAX_SEQ_LEN, peek=True))#, return_sequences=True))
                            hidden_dim=100, output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE, 
                            depth=1))
        model.add(TimeDistributed(Dense(self.data_vocab_size+1, activation="softmax")))
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        self.autoencoder = model


        #embedding = Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
        #       weights=[self.embedding_matrix], trainable=False)(input_seq)

        #encoder = LSTM(200)(embedding)
        #repeated = RepeatVector(MAX_SEQ_LEN)(encoder)
        #decoder = LSTM(200, return_sequences=True)(repeated)

        #time_dist = TimeDistributed(Dense(self.data_vocab_size+1, activation="softmax"))(decoder)

        #seq_autoencoder = Model(input_seq, time_dist)

        #seq_autoencoder.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        #seq_autoencoder.summary()

        #self.autoencoder = seq_autoencoder
        return self.autoencoder

    def train(self, train_x, train_y, test_x, test_y):
        print("one hot shape ", train_y.shape)
        self.autoencoder.fit(np.asarray(train_x), train_y,
                        nb_epoch=ACTOR_NUM_EPOCHS,
                        batch_size=ACTOR_BATCH_SIZE,
                        shuffle=True,
                        validation_data=(test_x, test_y),
                        verbose=1)

    def save(self, filename):
        self.autoencoder.save(filename)

    def load_weights(self, filename):
        self.autoencoder.load_weights(filename)

    def predict(self, data):
        return self.autoencoder.predict(data)


def get_w2v_model(filename):
    #Can't use load as file is in C text format.
    return gensim.models.Word2Vec.load_word2vec_format(filename)

def train_w2v_model(sentences):
    w2v_model = gensim.models.Word2Vec(sentences)
    #w2v_model.train(sentences)
    return w2v_model


#def predict_output()
