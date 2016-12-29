import numpy as np
import gensim

from keras.layers import Input, LSTM, RepeatVector, Activation, Dense, TimeDistributed, Embedding
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from seq2seq import Seq2Seq, AttentionSeq2Seq

from config import BATCH_SIZE, TIME_STEPS, MAX_SEQ_LEN, TOKEN_REPRESENTATION_SIZE, NUM_EPOCHS

from text_preprocessing import get_word_to_index_dic, get_index_to_word_dic, load_data, tokenize_sentences, get_train_val_test_data, preprocess_text

from autoencoder import Autoencoder

class ActorCriticAutoEncoder(Autoencoder):
    def __init__(self, w2v_model, token_to_index_dic):
        self.w2v_model = w2v_model
        self.token_to_index_dic = token_to_index_dic
        self.data_vocab_size = len(token_to_index_dic)
        print("Data vocab size= %d" % self.data_vocab_size)

    def create_nn_model(self):
        self.create_embedding()
        self.create_actor_model()
        self.create_critic_model()

    def create_actor_model(self):
        # This converts the positive indices(integers) into a dense multi-dim representation.
        model = Sequential()
        model.add(Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
            weights=[self.embedding_matrix], trainable=False,
            input_shape=(MAX_SEQ_LEN,)))
        model.add(AttentionSeq2Seq(batch_input_shape=(None, MAX_SEQ_LEN,
            TOKEN_REPRESENTATION_SIZE), hidden_dim=100,
            output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE,
            depth=1))
        model.add(TimeDistributed(Dense(self.data_vocab_size+1, activation="softmax")))
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        self.actor = model

    def create_critic_model(self):
        model = Sequential()
        model.add(Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
            weights=[self.embedding_matrix], trainable=False,
            input_shape=(MAX_SEQ_LEN,)))
        model.add(AttentionSeq2Seq(batch_input_shape=(None, MAX_SEQ_LEN,
            TOKEN_REPRESENTATION_SIZE), hidden_dim=100,
            output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE,
            depth=1))
        # The output is expected to be one scalar approximating the value of a state.
        model.add(TimeDistributed(Dense(1, activation="linear")))
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        self.critic = model

    def train_crtic(self, train_x, train_y, test_x, test_y):
        pass

    def distance_metric(self, X, Y):
        """x and y are two sequences of embeddings. This function returns the
        sum of differences of the embeddings"""

        return sum(sum([((x_i - y_i)**2)**(.5) for x_i, y_i in zip(x, y)]) for x, y in zip(X, Y))

    def reward(self, predicted_seq, ground_truth_seq):
        """
        The reward is calculated as follows:

        Let X be the input sequence to the actor and Y be the output from the
        actor. We first calculate score for each prefix of X and Y, i.e,
        R(X_1:1, Y_1:1), R(X_1:2, Y_1:2)..., R(X_1:T, Y_1:T), assuming size of
        the sequences is T.

        Then the reward for each word r_t(y_t; Y_1:t-1) = R(X_1:t-1, Y_1:t-1).

        action_timestep: t, Corresponds to the t in y_t.
        predicted_seq: Y = Y_1:T, The whole sequence generated by the autoencoder.
        ground_truth_seq: X = X_1:T
        """

        # Calculate scores for each prefix
        R = []
        # Score for 0th prefix is 0.
        R.append(0)

        T = len(predicted_seq)
        assert(T == len(ground_truth_seq))
        for t in range(1, T+1):
            X_1_t = ground_truth_seq[:t]
            Y_1_t = predicted_seq[:t]
            score = distance_metric(X_1_t, Y_1_t)
            R.append(score)
        print(R)
        # Difference of consecutive scores.
        # To be used as reward for tth prediction.
        return [j - i for j, i in zip(R[1:], R)]







