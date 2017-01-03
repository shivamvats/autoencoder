import numpy as np
import gensim

from autoencoder import Autoencoder, get_w2v_model
from utils import one_hot, de_one_hot
from text_preprocessing import get_word_to_index_dic, get_index_to_word_dic, load_data, tokenize_sentences, get_train_val_test_data, preprocess_text
from actor_critic import ActorCriticAutoEncoder

from config import (WEIGHTS_FILE, SAVE_WEIGHTS, LOAD_WEIGHTS, TRAIN, DATA_FILE,
    USE_SAVED_PREPROCESSED_INPUT, PREPROCESSED_INPUT_FILE)


DEBUG = 1

def main_actorCritic():

    #print("Loading data")
    #sentences = load_data(DATA_FILE)
    #print("Data loaded")

    if USE_SAVED_PREPROCESSED_INPUT:
        sentences = pickle.load(open(PREPROCESSED_INPUT_FILE, 'r'))[:5000]
    #else:
    #    sentences = preprocess_text(sentences)[:5000]
    print("shape of sentences", sentences.shape)

    print("Training w2v model")
    w2v_model = train_w2v_model(sentences)
    print("w2v model trained")

    token_sequences, output_sequences, token_to_index_dic = tokenize_sentences(sentences)
    index_to_word_dic = get_index_to_word_dic(token_to_index_dic)
    token_sequences = np.asarray(token_sequences)
    output_sequences = np.asarray(output_sequences)
    print("input shape", token_sequences.shape)

    #token_sequences = token_sequences[:1000, :]
    #output_sequences[:1000, :]

    output_sequences = [one_hot(seq, len(token_to_index_dic)) for seq in output_sequences]
    print("Tokenization done. %d sequences" % len(token_sequences), "shape ", token_sequences.shape)
    #token_to_index_dic = get_word_to_index_dic(w2v_model, token_sequences)
    print("preprocessing done")
    train_x, train_y, val_x, val_y, test_x, test_y  = get_train_val_test_data(token_sequences, output_sequences)

    autoencoder = Autoencoder(w2v_model, token_to_index_dic)


    print("Creating NN model")
    autoencoder.create_nn_model()
    print("NN model created")

    if LOAD_WEIGHTS:
        print("Loading saved weights from %s" % WEIGHTS_FILE)
        autoencoder.load_weights(WEIGHTS_FILE)
    print("Training autoencoder")

    if TRAIN:
        autoencoder.train(train_x, train_y,  val_x, val_y)

    if SAVE_WEIGHTS:
        autoencoder.save(WEIGHTS_FILE)

    output = autoencoder.predict(test_x)

    actor_critic = ActorCriticAutoEncoder(w2v_model=w2v_model,
            token_to_index_dic=token_to_index_dic,
            actor=autoencoder.autoencoder)
    actor_critic.create_critic_model()
    actor_critic.train_critic(train_x, output)



def main_auto():
    print("Loading w2v model")
    w2v_model = get_w2v_model("/home/aries/Documents/Learning/DL/autoencoder/data/glove.6B/glove.6B.200d_gensim.txt")
    print("w2v model loaded")

    #print("Loading data")
    #sentences = load_data(DATA_FILE)
    #print("Data loaded")

    if USE_SAVED_PREPROCESSED_INPUT:
        sentences = pickle.load(open(PREPROCESSED_INPUT_FILE, 'r'))[:5000]
    #else:
    #    sentences = preprocess_text(sentences)[:5000]
    print("shape of sentences", sentences.shape)

    token_sequences, output_sequences, token_to_index_dic = tokenize_sentences(sentences)
    index_to_word_dic = get_index_to_word_dic(token_to_index_dic)
    token_sequences = np.asarray(token_sequences)
    output_sequences = np.asarray(output_sequences)
    print("input shape", token_sequences.shape)

    #token_sequences = token_sequences[:1000, :]
    #output_sequences[:1000, :]

    output_sequences = [one_hot(seq, len(token_to_index_dic)) for seq in output_sequences]
    print("Tokenization done. %d sequences" % len(token_sequences), "shape ", token_sequences.shape)
    #token_to_index_dic = get_word_to_index_dic(w2v_model, token_sequences)
    print("preprocessing done")
    train_x, train_y, val_x, val_y, test_x, test_y  = get_train_val_test_data(token_sequences, output_sequences)

    autoencoder = Autoencoder(w2v_model, token_to_index_dic)


    print("Creating NN model")
    autoencoder.create_nn_model()
    print("NN model created")

    if LOAD_WEIGHTS:
        print("Loading saved weights from %s" % WEIGHTS_FILE)
        autoencoder.load_weights(WEIGHTS_FILE)
    print("Training autoencoder")

    if TRAIN:
        autoencoder.train(train_x, train_y,  val_x, val_y)

    if SAVE_WEIGHTS:
        autoencoder.save(WEIGHTS_FILE)

    output = autoencoder.predict(test_x)

    #print(test_x)
    #test_x = [de_one_hot(ele) for ele in test_x]

    print("Input")
    for seq in test_x:
        print(index_to_sentence(index_to_word_dic, seq))
    print("\n")
    print("########Generated##########")
    print("\n")
    for seq in output:
        print(index_to_sentence(index_to_word_dic, [np.argmax(ele) for ele in seq]))


def index_to_sentence(index_to_word_dic, seq):
    return " ".join([index_to_word_dic[ele] for ele in seq])

main_actorCritic()







