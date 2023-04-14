from preprocessing import *
from train_and_predict import *
from utils import *

import sys

# Paths to the CONLLU data
path_train ='Data/en_ewt-up-train.conllu'
path_eval = 'Data/en_ewt-up-dev.conllu'
path_test = 'Data/en_ewt-up-test.conllu'
paths = [path_train, path_eval, path_test]

# Change to test when you are evaluating on test-set:
eval_split = 'test'

# Embedding model
path_emb = 'Models/GoogleNews-vectors-negative300.bin'

def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        None

    data = convert_conll_to_df(paths)
    token2idx, idx2token = get_dict_map(data, 'TOKEN')
    tag2idx, idx2tag = get_dict_map(data, 'LABEL')
    n_vocab, n_tags = len(token2idx), len(tag2idx)

    # Adds the index information to the dataframe
    data['WORD_IDX'] = data['TOKEN'].map(token2idx)
    data['TAG_IDX'] = data['LABEL'].map(tag2idx)
    data['PRED_IDX'] = data['LABEL'].apply(lambda x: 1 if x == 'V' else 0)

    w2v_model, emb_dim, emb_matrix = integrate_emb(path_emb, token2idx)
    data_grouped = transform_to_seq(data)
    train_tokens, eval_tokens, train_tags, eval_tags = get_pad_train_test_val(data_grouped, tag2idx, data, n_vocab, eval_split)
    input_dim, output_dim, input_length, n_tags = build_model(data, emb_dim, data_grouped, tag2idx)
    model_bilstm = get_bilstm_lstm_model(emb_matrix, emb_dim, input_length, output_dim, n_tags, token2idx)
    plot_model(model_bilstm)
    results['with_add_lstm_3epochs'] = train_model(train_tokens, train_tags, model_bilstm)
    y_pred = model_bilstm.predict(eval_tokens)

    evaluate(y_pred, eval_tags)

if __name__ == '__main__':
    main(sys.argv[1:])