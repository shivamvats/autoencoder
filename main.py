import numpy as np
import gensim

from autoencoder import Autoencoder, get_w2v_model
from utils import one_hot, de_one_hot
from text_preprocessing import get_word_to_index_dic, get_index_to_word_dic, load_data, tokenize_sentences, get_train_val_test_data, preprocess_text

from config import WEIGHTS_FILE, SAVE_WEIGHTS, LOAD_WEIGHTS, TRAIN

DEBUG = 1

def main():
    print("Loading w2v model")
    w2v_model = get_w2v_model("/home/aries/Documents/Learning/DL/autoencoder/data/glove.6B/glove.6B.200d_gensim.txt")
    print("w2v model loaded")

    print("Loading data")
    sentences = load_data("/home/aries/Documents/Learning/DL/autoencoder/data/all_sentences.pkl")
    print("Data loaded")

    sentences = preprocess_text(sentences)[:5000]
    print("shape of sentences", sentences.shape)

    token_sequences, output_sequences, token_to_index_dic = tokenize_sentences(sentences)
    index_to_word_dic = get_index_to_word_dic(token_to_index_dic)
    token_sequences = np.asarray(token_sequences)
    output_sequences = np.asarray(output_sequences)
    print("input shape", token_sequences.shape)

    #token_sequences = token_sequences[:1000, :]
    #output_sequences[:1000, :]

    output_sequences = [one_hot(seq, 10, len(token_to_index_dic)) for seq in output_sequences]
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

main()







