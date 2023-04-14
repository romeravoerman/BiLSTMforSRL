from gensim.models import KeyedVectors
import numpy as np

def get_dict_map(data, token_or_tag):
    """
    Create two dictionaries to map tokens or tags to indices and vice versa.

    Args:
    - data (pandas.DataFrame): DataFrame containing the token and label columns.
    - token_or_tag
    """

    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'TOKEN':
        vocab = list(set(data['TOKEN'].to_list()))
    else:
        vocab = list(set(data['LABEL'].to_list()))

    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}

    return tok2idx, idx2tok

def integrate_emb(path_emb, token2idx):
    w2v_model = KeyedVectors.load_word2vec_format(path_emb, binary=True)
    emb_dim = 300
    embedding_matrix = np.zeros((len(token2idx) + 1, emb_dim))
    print(embedding_matrix.shape)
    for word, i in token2idx.items():
        if word in w2v_model.key_to_index:
            embedding_vector = w2v_model[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return w2v_model, emb_dim, embedding_matrix


def transform_to_seq(data):
    data_fillna = data.fillna(method='ffill', axis=0)
    data_group = data_fillna.groupby(['# SENTENCE'], as_index=False)[
        'ID', 'TOKEN', 'LEMMA', 'POS-UNIV', 'POS', 'MORPH', 'HEAD', 'BASIC-DEP',
        'ENH-DEP', 'SPACE', 'PREDICATE', 'LABEL', 'SPLIT', '# SENTENCE', 'WORD_IDX', 'TAG_IDX'].agg(lambda x: list(x))

    return data_group