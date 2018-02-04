from __future__ import print_function
import numpy as np
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, TimeDistributed, Dense,\
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
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
SENTENCE_DIM = 128
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25
DROPOUT = 0.1
BATCH_SIZE = 32
OPTIMIZER = 'adam'


def read_test_csv(fname):
    q1, q2 = [], []
    with open(fname, 'r') as f:
        f.readline()
        while f.readable():
            line = f.readline()
            q1_idx = line.find('"')
            q2_idx = line.find('","')
            while line[q2_idx-1] == '"':
                q2_idx = line.find('","', q2_idx+2)
            q1.append(line[q1_idx+1:q2_idx])
            q2.append(line[q2_idx+3:-1])
    return q1, q2


# If the dataset, embedding matrix and word count exist in the local directory
if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) \
        and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) \
        and exists(WORD_EMBEDDING_MATRIX_FILE) and exists(Q1_TEST_DATA_FILE) \
        and exists(Q2_TEST_DATA_FILE):
    # Then load them
    q1_data = np.load(Q1_TRAINING_DATA_FILE)
    q2_data = np.load(Q2_TRAINING_DATA_FILE)
    q1_test = np.load(Q1_TEST_DATA_FILE)
    q2_test = np.load(Q2_TEST_DATA_FILE)
    labels = np.load(LABEL_TRAINING_DATA_FILE)
    word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_FILE)
    with open(NB_WORDS_DATA_FILE, 'r') as f:
        nb_words = json.load(f)['nb_words']
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

    test_q1, test_q2 = read_test_csv('test.csv')

    print('Test question pairs: %d' % len(test_q1))

    # Build tokenized word index
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    question1_test_sequences = tokenizer.texts_to_sequences(test_q1)
    question2_test_sequences = tokenizer.texts_to_sequences(test_q2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q1_test = pad_sequences(question1_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_test = pad_sequences(question2_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)

    print('Shape of label tensor:', labels.shape)

    # Persist training and configuration data to files
    np.save(Q1_TRAINING_DATA_FILE, q1_data)
    np.save(Q2_TRAINING_DATA_FILE, q2_data)
    np.save(Q1_TEST_DATA_FILE, q1_test)
    np.save(Q2_TEST_DATA_FILE, q2_test)
    np.save(LABEL_TRAINING_DATA_FILE, labels)
    np.save(WORD_EMBEDDING_MATRIX_FILE, word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)

# Partition the dataset into train and test sets
X_train = np.stack((q1_data, q2_data), axis=1)
y_train = labels
X_test = np.stack((q1_test, q2_test), axis=1)
Q1_train = X_train[:, 0]
Q2_train = X_train[:, 1]
Q1_test = X_test[:, 0]
Q2_test = X_test[:, 1]

# Define the model
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)
#q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q1 = Bidirectional(LSTM(SENTENCE_DIM, return_sequences=True), merge_mode='sum')(q1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(SENTENCE_DIM, ))(q1)

q2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)
#q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q2 = Bidirectional(LSTM(SENTENCE_DIM, return_sequences=True), merge_mode='sum')(q2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(SENTENCE_DIM, ))(q2)

distance = Subtract()([q1, q2])
angle = Multiply()([q1, q2])

merged = concatenate([distance, angle])
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

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

# Evaluate the model with best validation accuracy on the test partition
model.load_weights(MODEL_WEIGHTS_FILE)
y_test = model.predict(X_test)
with open('submission.csv', 'w') as f:
    f.write('test_id,is_duplicate\n')
    for i, _y in enumerate(y_test):
        f.write('{},{}\n'.format(i, _y))
#loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
#print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))

