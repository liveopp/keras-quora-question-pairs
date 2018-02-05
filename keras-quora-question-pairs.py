from __future__ import print_function

import numpy as np
import csv, datetime, time
from os.path import expanduser, exists
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, CuDNNLSTM, Dense,\
    Bidirectional, Lambda, Multiply, Subtract, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split

# Initialize global variables
KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
TEST_QUESTION_PAIRS_FILE = 'test.csv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
Q1_TEST_DATA_FILE = 'q1_test.npy'
Q2_TEST_DATA_FILE = 'q2_test.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 2196018
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
SENTENCE_DIM = 100
MODEL_JSON_FILE = 'question_pairs_model.json'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
MODEL_JSON_FILE = 'question_pairs.json'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 10
DROPOUT = 0.1
BATCH_SIZE = 60
OPTIMIZER = 'adam'
USE_CUDA = True


def get_embedding_matrix(fname):
    # Download and process GloVe embeddings
    print("Processing", fname)
    word_index = {}
    with open(fname, encoding='utf-8') as f:
        # zero index for out-of-vocabulary words
        embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM),
                                    dtype='float32')
        idx = 0
        for line in f:
            idx += 1
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_matrix[idx] = vector
            word_index[word] = idx
    return word_index, embedding_matrix


def texts_to_sequences(texts, word_index):
    """Transforms each text in texts in a sequence of integers.
    # Arguments
        texts: A list of texts (strings).
    # Returns
        A list of sequences.
    """
    res = []
    for text in texts:
        seq = text_to_word_sequence(text)
        vect = []
        for w in seq:
            i = word_index.get(w)
            if i is None:
                i = 0
            vect.append(i)
        res.append(vect)
    return res


def read_test_csv(fname):
    q1, q2 = [], []
    with open(fname, 'r') as f:
        f.readline()
        whole_line = ''
        num = 1
        for line in f:
            line = line.strip()
            if line.find(',') > 0 and line.split(',')[0] == str(num):
                if whole_line:
                    q1_idx = whole_line.find('"')
                    q2_idx = whole_line.find('","')
                    while whole_line[q2_idx-1] == '"':
                        if whole_line.find('","', q2_idx+2) > 0:
                            q2_idx = whole_line.find('","', q2_idx+2)
                            print(len(q1)+2)
                        else:
                            break
                    q1.append(whole_line[q1_idx+1:q2_idx])
                    q2.append(whole_line[q2_idx+3:-1])
                    num += 1
                whole_line = line
            else:
                whole_line += line
        q1_idx = whole_line.find('"')
        q2_idx = whole_line.find('","')
        while whole_line[q2_idx-1] == '"':
            if whole_line.find('","', q2_idx+2) > 0:
                q2_idx = whole_line.find('","', q2_idx+2)
                print(len(q1)+2)
            else:
                break
        q1.append(whole_line[q1_idx+1:q2_idx])
        q2.append(whole_line[q2_idx+3:-1])
        assert num == 2345796
    return q1, q2


def build_model(is_load=False):
    if not is_load:
        # Define the model
        question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
        question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

        embedding_layer = Embedding(MAX_NB_WORDS,
                                    EMBEDDING_DIM,
                                    weights=[word_embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        q1 = embedding_layer(question1)
        if USE_CUDA:
            q1 = Bidirectional(CuDNNLSTM(SENTENCE_DIM, return_sequences=True), merge_mode='concat')(q1)
        else:
            q1 = Bidirectional(LSTM(SENTENCE_DIM, return_sequences=True), merge_mode='concat')(q1)
        q1 = Lambda(lambda x: K.mean(x, axis=1), output_shape=(2*SENTENCE_DIM, ))(q1)

        q2 = embedding_layer(question2)
        if USE_CUDA:
            q2 = Bidirectional(CuDNNLSTM(SENTENCE_DIM, return_sequences=True), merge_mode='concat')(q2)
        else:
            q2 = Bidirectional(LSTM(SENTENCE_DIM, return_sequences=True), merge_mode='concat')(q2)
        q2 = Lambda(lambda x: K.mean(x, axis=1), output_shape=(2*SENTENCE_DIM, ))(q2)

        distance = Lambda(lambda x: K.abs(x))(Subtract()([q1, q2]))
        angle = Multiply()([q1, q2])

        merged = concatenate([distance, angle])
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(DROPOUT)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[question1, question2], outputs=is_duplicate)
        model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    else:
        # load json and create model
        json_file = open()
        loaded_model_json = json_file.read(MODEL_JSON_FILE)
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(MODEL_WEIGHTS_FILE)
        print("Loaded model from disk")
    return model


def train(model, X_train, y_train):
    Q1_train = X_train[:,0]
    Q2_train = X_train[:,1]
    # Train the model, checkpointing weights with best validation accuracy
    print("Starting training at", datetime.datetime.now())
    t0 = time.time()
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
    history = model.fit([Q1_train, Q2_train],
                        y_train,
                        epochs=NB_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=2,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)
    t1 = time.time()
    print("Training ended at", datetime.datetime.now())
    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

    # Print best validation accuracy and epoch
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))
    model.load_weights(MODEL_WEIGHTS_FILE)
    return model


if __name__ == '__main__':
    # If the dataset, embedding matrix and word count exist in the local directory
    if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) \
            and exists(LABEL_TRAINING_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
        # Then load them
        q1_data = np.load(Q1_TRAINING_DATA_FILE)
        q2_data = np.load(Q2_TRAINING_DATA_FILE)
        labels = np.load(LABEL_TRAINING_DATA_FILE)
        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_FILE)
    else:
        # Else download and extract questions pairs data
        if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
            get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

        print("Processing", QUESTION_PAIRS_FILE)

        question1 = []
        question2 = []
        is_duplicate = []
        with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                question1.append(row['question1'])
                question2.append(row['question2'])
                is_duplicate.append(row['is_duplicate'])

        print('Train question pairs: %d' % len(question1))

        print("Processing", TEST_QUESTION_PAIRS_FILE)

        # Build tokenized word index
        word_index, word_embedding_matrix = get_embedding_matrix(GLOVE_FILE)
        question1_word_sequences = texts_to_sequences(question1, word_index)
        question2_word_sequences = texts_to_sequences(question2, word_index)

        print("Words in index: %d" % len(word_index))

        # Prepare training data tensors
        q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        labels = np.array(is_duplicate, dtype=int)
        print('Shape of question1 data tensor:', q1_data.shape)

        print('Shape of label tensor:', labels.shape)

        # Persist training and configuration data to files
        np.save(Q1_TRAINING_DATA_FILE, q1_data)
        np.save(Q2_TRAINING_DATA_FILE, q2_data)
        np.save(LABEL_TRAINING_DATA_FILE, labels)
        np.save(WORD_EMBEDDING_MATRIX_FILE, word_embedding_matrix)

    # Partition the dataset into train and test sets
    X = np.stack((q1_data, q2_data), axis=1)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
    Q1_test = X_test[:, 0]
    Q2_test = X_test[:, 1]

    model = build_model()
    model = train(model)

    # Evaluate the model with best validation accuracy on the test partition
    loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
    print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))
