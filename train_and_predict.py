import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input, optimizers
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# reproducibility
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)

def get_pad_train_test_val(data_group, tag2idx, data, n_vocab, eval_split="dev", ):
    """
    The function get_pad_train_test_val pads the token and tag sequences with zeros and generates one-hot encoded tags
    for the input data. It then splits the dataset into train and evaluation sets based on the split ratio provided.

    Args:

    - data_group (pandas.DataFrame): A dataframe containing token and tag sequences, split information, and precomputed token and tag indices.
    - data (pandas.DataFrame): A dataframe containing token and tag sequences, split information, and precomputed token and tag indices.
    - eval_split (str): A string indicating the split to use for evaluation data (default is "dev").
    - n_vocab (int): The number of unique tokens in the token vocabulary (default is None).

    Returns:
    - train_tokens (numpy.ndarray): containing the padded and indexed train tokens.
    - eval_tokens (numpy.ndarray): containing the padded and indexed evaluation tokens.
    - train_tags (numpy.ndarray): containing the one-hot encoded tags for the train data.
    - eval_tags (numpy.ndarray): containing the one-hot encoded tags for the evaluation data.
    """

    # Pad tokens
    tokens = data_group['WORD_IDX'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value=n_vocab)

    # Pad Tags and convert it into one hot encoding
    tags = data_group['TAG_IDX'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int64', padding='post', value=tag2idx["_"])
    n_tags = len(tag2idx)
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]

    # Split train, test and validation set
    train_tokens, eval_tokens, train_tags, eval_tags, train_preds, eval_preds = [], [], [], [], [], []
    for i, row in data_group.iterrows():
        if 'train' in row['SPLIT']:
            train_tokens.append(pad_tokens[i])
            train_tags.append(pad_tags[i])
        elif eval_split in row['SPLIT']:
            eval_tokens.append(pad_tokens[i])
            eval_tags.append(pad_tags[i])

    print(
        'evaluation based on:', eval_split,
        '\ntrain_tokens length:', len(train_tokens),
        '\ntrain_tags length:', len(train_tags),
        '\neval_tokens:', len(eval_tokens),
        '\neval_tags:', len(eval_tags),
    )

    return np.array(train_tokens), np.array(eval_tokens), np.array(train_tags), np.array(eval_tags)

def build_model(data, emb_dim, data_group,tag2idx):
    input_dim = len(list(set(data['TOKEN'].to_list()))) + 1
    output_dim = emb_dim  # number of dimensions
    input_length = max([len(s) for s in data_group['WORD_IDX'].tolist()])
    n_tags = len(tag2idx)

    return input_dim, output_dim, input_length, n_tags


def get_bilstm_lstm_model(embedding_matrix, embedding_dim, input_length, output_dim, n_tags, token2idx):
    """
    Creates a BiLSTM model with an embedding layer using the provided embedding matrix and dimensions.

    Parameters:
    embedding_matrix (numpy array): Pre-trained embedding matrix
    embedding_dim (int): Dimensions of the embedding layer

    Returns:
    model: A compiled BiLSTM model with TimeDistributed Dense layer for tagging
    """
    # Create a sequential model
    model = Sequential()

    # Add an embedding layer with pre-trained embedding matrix
    embedding_layer = Embedding(len(token2idx) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)
    model.add(embedding_layer)

    # Add a BiLSTM layer
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                            merge_mode='concat'))

    # Add a TimeDistributed Dense layer for tagging
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    model.summary()

    return model


def train_model(X, y, model):
    """Train a given Keras model on input-output pairs (X, y).

    Args:
    - X: the input data
    - y: the output data
    - model: the model to train.

    Returns:
    - loss: the training loss at the end of each epoch.
    """

    loss = list()
    # set epochs to 3 (from 25) (you can change this)
    for i in range(3):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=200, verbose=1, epochs=3, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss

def evaluate(y_pred, eval_tags):
    # Convert the predictions and true labels to their original tag forms (not one-hot encoded)
    predicted_tags = np.argmax(y_pred, axis=-1)
    true_tags = np.argmax(np.array(eval_tags), axis=-1)

    # Create a reverse mapping from tag indices to tag names
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    predicted_tags_names = []
    true_tags_names = []

    for true_seq, pred_seq in zip(true_tags, predicted_tags):
        for true_tag, pred_tag in zip(true_seq.ravel(), pred_seq.ravel()):
            # Ignore padding values when both true_tag and pred_tag are padding tags
            if not (true_tag == tag2idx["C-ARGM-GOL"] and pred_tag == tag2idx["C-ARGM-GOL"]):
                predicted_tags_names.append(idx2tag[pred_tag])
                true_tags_names.append(idx2tag[true_tag])

    predicted_tags_names_filtered, true_tags_names_filtered = [], []

    for predicted_tag_name, true_tag_name in zip(predicted_tags_names, true_tags_names):
        if predicted_tag_name is not None and true_tag_name is not None:
            predicted_tags_names_filtered.append(predicted_tag_name)
            true_tags_names_filtered.append(true_tag_name)

    # Generate the classification report
    report = classification_report(true_tags_names_filtered, predicted_tags_names_filtered, zero_division=0)

    return report