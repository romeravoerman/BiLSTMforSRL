import pandas as pd

def read_conllu_file(file_path):
    """
    Reads a CoNLL-U formatted file and returns a list of sentences, where each sentence is a list of token information.

    Args:
        file_path (str): The file path of the CoNLL-U file to be read.

    Returns:
        list: A list of sentences, where each sentence is represented as a list of token information.
    """

    sentences = []
    with open(file_path, "r", encoding="utf-8") as file:
        sentence = []
        for line in file:
            line = line.strip()

            if line.startswith("#"):
                continue

            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []

            else:
                sentence.append(line.split("\t"))

        if sentence:
            sentences.append(sentence)
    return sentences


def copy_predicates(file_path):
    """
    Reads a CoNLL-U formatted file, finds sentences with multiple predicates, and creates new sentences by copying the original
    sentence and changing the predicate values.

    Args:
        file_path (str): The file path of the CoNLL-U file to be read.

    Returns:
        list: A list of modified sentences, where each modified sentence is represented as a list of token information.
    """

    sentences = read_conllu_file(file_path)
    new_sentences = []

    for sentence in sentences:
        # print(sentence)
        predicate_values = list([columns[10] for columns in sentence if len(columns) >= 11])
        predicate_values = [value for value in predicate_values if value != "_"]

        if len(predicate_values) <= 1:
            new_sentences.append(sentence)
        elif len(predicate_values) > 1:
            for i, pred in enumerate(predicate_values):
                b = i + 1
                df = pd.DataFrame(sentence)
                df_2 = df.iloc[:, :11].copy()
                new_col = df.iloc[:, (10 + b)]
                df_2[11] = new_col
                new_sentence = df_2.values.tolist()
                new_sentences.append(new_sentence)
    return new_sentences


def create_adapted_conll_files(file_paths):
    """
    Creates new CoNLL-U formatted files by adapting the original files with the 'copy_predicates' function.

    Args:
        file_paths (list): A list of file paths to CoNLL-U formatted files to be adapted.

    Returns:
        list: A list of file paths to the newly created adapted CoNLL-U files.
    """

    adapted_paths = []
    for file_path in file_paths:
        sentences = copy_predicates(file_path)

        output_file = file_path.replace(".conllu", "_adapted.conllu")

        with open(output_file, "w", encoding="utf-8") as outfile:
            for sentence in sentences:
                for columns in sentence:
                    if None in columns:
                        continue
                    else:
                        outfile.write("\t".join(columns) + "\n")
        adapted_paths.append(output_file)

    return adapted_paths


def convert_conll_to_df(paths):
    """
    Converts a list of CoNLL files to a pandas dataframe with additional columns for sentence ID and split ID.

    Args:
        paths (list): A list of file paths to CoNLL files.

    Returns:
        final_dataframe (pandas.DataFrame): A pandas dataframe containing the concatenated data from all input files with
                                             two additional columns, "# SENTENCE" and "SPLIT".
    """

    complete_dataframe = []
    sentence_id = 0
    adapted_paths = create_adapted_conll_files(paths)

    for path in paths:
        suffix = path.split('.')[0]
        split = (suffix.split('-')[-1]).split('_')[0]
        print(split)

        df = pd.read_csv(path, delimiter='\t', header=None, on_bad_lines="skip", engine="python",
                         names=['ID', 'TOKEN', 'LEMMA', 'POS-UNIV', 'POS', 'MORPH', 'HEAD', 'BASIC-DEP', 'ENH-DEP',
                                'SPACE', 'PREDICATE', 'LABEL'])
        df.dropna()
        # Adds the SPLIT column to the dataframe
        df['SPLIT'] = split

        # Adds the # SENTENCE column to the dataframe
        df['# SENTENCE'] = 0
        for index, row in df.iterrows():
            if row['ID'] == 1:
                sentence_id += 1
            df.at[index, '# SENTENCE'] = sentence_id

        # print(len(df))
        complete_dataframe.append(df)

    final_dataframe = pd.concat(complete_dataframe, ignore_index=True)

    return final_dataframe