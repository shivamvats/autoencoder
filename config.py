MOTHER_INPUT_FILE = "./data/all_sentences.pkl"

PRETRAINING_DATA_FILE = "/home/aries/Documents/Learning/DL/autoencoder/data/all_sentences_6000.pkl"
USE_SAVED_PREPROCESSED_INPUT = True
PRETRAINING_PREPROCESSED_INPUT_FILE = "/home/aries/Documents/Learning/DL/autoencoder/data/preprocessed_sentences_6000.pkl"

DATA_FILE = "./data/all_sentences_8000.pkl"
PREPROCESSED_DATA_FILE = "./data/preprocessed_sentences_8000.pkl"

TOKEN_REPRESENTATION_SIZE = 200
PADDING_TOKEN = "#"

ACTOR_BATCH_SIZE = 32
ACTOR_NUM_EPOCHS = 300
CRITIC_BATCH_SIZE = 32
CRITIC_NUM_EPOCHS = 20

MAX_SEQ_LEN = 10
TIME_STEPS = 5
UNKNOWN_TOKEN = "$$$"
ENDLINE_TOKEN = "###"

MAX_VOCAB_SIZE=20000
VALIDATION_SPLIT = 0.2

LOAD_WEIGHTS=True
SAVE_WEIGHTS=True
PRETRAINING_ACTOR_WEIGHTS_FILE="./models/simple_lstm.h5"
PRETRAINING_CRITIC_MODEL_NAME = "./models/pretraining_critic_model"

TRAIN_ACTOR=False
TRAIN_CRITIC=True
