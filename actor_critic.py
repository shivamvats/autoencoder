import numpy as np
import gensim
import copy

from keras.layers import Input, LSTM, RepeatVector, Activation, Dense, TimeDistributed, Embedding
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from seq2seq import Seq2Seq, AttentionSeq2Seq

from config import CRITIC_BATCH_SIZE, CRITIC_NUM_EPOCHS, TIME_STEPS, MAX_SEQ_LEN, TOKEN_REPRESENTATION_SIZE, ACTOR_BATCH_SIZE

from text_preprocessing import get_word_to_index_dic, get_index_to_word_dic, load_data, tokenize_and_pad_sentences, get_train_val_test_data, preprocess_text

from utils import one_hot

from autoencoder import Autoencoder

class ActorCriticAutoEncoder(Autoencoder):
    def __init__(self, w2v_model, token_to_index_dic, actor=None, critic=None):
        super(ActorCriticAutoEncoder, self).__init__(w2v_model, token_to_index_dic)

        self.actor = actor
        self.critic = critic

    def create_nn_model(self):
        self.create_embedding()
        self.create_actor_model()

    def maximum_layer(X):
        X = T.max(X, 0)

    def maximum_output_shape(input_shape):
        return (input_shape[0],)

    def evaluation_function(layers):
        X = layers[0]
        X = T.max(X, 0)

        Y = layers[1]
        return T.sum(T.dot(X, Y))

    def create_actor_model(self):
        # This converts the positive indices(integers) into a dense multi-dim representation.

        self.create_critic_model()

        model = Sequential()
        model.add(Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
            weights=[self.get_embedding_matrix()], trainable=False,
            input_shape=(MAX_SEQ_LEN,)))
        model.add(AttentionSeq2Seq(batch_input_shape=(None, MAX_SEQ_LEN,
            TOKEN_REPRESENTATION_SIZE), hidden_dim=100,
            output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE,
            depth=1))
        model.add(TimeDistributed(Dense(self.data_vocab_size+1, activation="softmax")))

        evaluation_model = Sequential()
        evaluation_model.add(Merge([model, self.critic], mode=evaluation_function))

        evaluation_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

        self.actor = actor
        self.evaluation_of_actor = evaluation_model

    def create_critic_model(self):
        model = Sequential()
        model.add(Embedding(self.data_vocab_size+1, TOKEN_REPRESENTATION_SIZE,
            weights=[self.get_embedding_matrix()], trainable=False,
            input_shape=(MAX_SEQ_LEN,)))
        model.add(AttentionSeq2Seq(batch_input_shape=(None, MAX_SEQ_LEN,
            TOKEN_REPRESENTATION_SIZE), hidden_dim=100,
            output_length=MAX_SEQ_LEN, output_dim=TOKEN_REPRESENTATION_SIZE,
            depth=1))
        # The output is expected to be one scalar approximating the value of a state.
        model.add(TimeDistributed(Dense(1, activation="linear")))
        model.summary()
        model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

        self.critic = model

    def train_actor(self, input_sequences, target):
        output = self.actor.predict(input_sequences)

        # Q_pi will contain Q value of all prefixes also since the last layer
        # of the critic is timeDistributed.
        Q_pi = self.critic.predict(output)

        output_p = [[np.max(word) for word in sent] for sent in output]
        actor_target = []

        # dJ = d( Sum(Pr(a | Y) * Q(a | Y)) )
        for sent, q in zip(output_p, Q_pi):
            actor_target.append(np.sum(np.dot(sent, q)))

        self.evaluation_of_actor.fit(input_sequences, actor_target, batch_size=ACTOR_BATCH_SIZE


    def train_critic(self, predicted_seqs_prob, ground_truth_seqs):
        # List of lists; list of rewards for each sequence.

        P = np.asarray(copy.copy(predicted_seqs_prob))
        X = [[np.argmax(ele) for ele in seq] for seq in predicted_seqs_prob]
        Y = ground_truth_seqs
        # Length of sequences.
        T = len(ground_truth_seqs[0])
        A = self.data_vocab_size

        train_x = []
        train_y = []

        # Note that our indexing starts from 0, whereas in the paper, it starts
        # from 1.
        print("Predicting rewards of subsequences")
        V_subse_predicted = np.asarray(self.critic.predict(X))
        #V_subse_predicted = V_subse_predicted.flatten()
        print("Rewards predicted using critic")

        # Getting the critic to predict one by one is too slow.
        # Let's collect all the subsequences in a list and predict beforehand.
        X_subse = []
        for x in X:
            subse_sent = []
            for i in range(len(x)):
                subse_sent.append(x[:i+1])
            X_subse = X_subse + subse_sent
        X_subse = np.asarray(pad_sequences(X_subse, maxlen=MAX_SEQ_LEN))


        i = 0
        len_x = len(X)
        print("\n\n")
        for p, y in zip(P, Y):
            print("Calculating rewards")
            rewards = self.reward(y, p)
            print("Rewards calculated: %d / %d " % (i, len_x))

            print("Calculating target")
            y_seq = []
            for t in range(T):
                #p_t = p[t]
                # y[:t+1] as numpy slices from 0 to t-1.
                #XXX#
                # In the original paper, they sum over all the actions/words.
                # This means, I need V_subse_predicted assuming the last word
                # is word_i (word_i is every word from vocab). 
                V = max(P[i][t] * V_subse_predicted[i][t])

                y_seq.append(rewards[t] + V)
            train_y.append(y_seq)
            i+=1

        train_x = np.asarray(X)
        train_y = np.asarray(train_y)
        train_y = train_y.reshape(train_y.shape + (1,))
        print(np.asarray(train_x).shape, np.asarray(train_y).shape)
        self.critic.fit(train_x, train_y,
            nb_epoch=CRITIC_NUM_EPOCHS, batch_size=CRITIC_BATCH_SIZE,
            shuffle=True, #validation_data=(test_x, test_y),
            verbose=1)

    def similarity(self, X, Y):
        """X and Y are two sequences of embeddings. This function returns the
        sum of differences of the embeddings"""

        return np.mean([np.dot(x, y) / (max(1e-7, np.linalg.norm(x)) * max(1e-7, np.linalg.norm(y))) for x, y in zip(X, Y)])

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
        #print(predicted_seq)
        for t in range(1, T+1):
            X_1_t = ground_truth_seq[:t]
            Y_1_t = predicted_seq[:t]
            # Higher the similarity, higher should be the reward. Hence using
            # cosine similarity.
            score = self.similarity(X_1_t, Y_1_t)
            R.append(score)
        # Difference of consecutive scores.
        # To be used as reward for tth prediction.
        return [j - i for j, i in zip(R[1:], R)]

    def save(self, filename):
        #self.actor.save(actor_filename)
        self.critic.save(critic_filename)

    def get_embedding_matrix(self):
        if self.embedding_matrix is None:
            self.create_embedding()
        return self.embedding_matrix




