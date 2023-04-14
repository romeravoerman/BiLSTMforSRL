# BiLSTM tutorial for Semantic Role Labeling
This GitHub Repository contains the files for the class Advanced NLP at VU Amsterdam by Romera Voerman (2686014). 

This repository contains the following directories:
- Data: directory holding the original .conllu files and the adapted files obtained from: https://github.com/UniversalPropositions/UP-1.0
- Models: the wordembedding should be added here, obtained from: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

This repository contains the following files:
- requirements.txt: a file listing all the requirements to run the code
- main.py: executes all functions to run the model
- preprocessing.py: preprocesses the conllu data into an adapted version of the conllu data
- train_and_predict.py: contains the code to pad the tokens and tag sequences, build the bilstm model, trains and evaluates the model
- utils.py: contains the code for get_dict_map, the integration of the embeddings and the transformation to sequential data
- R_bilstm_for_srl.ipynb: elaborates on all pieces of code in a notebook format which runs everything and shows outputs


To run the main script, call the main script in the command line in the following format:
[path to the current directory] python main.py [epochs:int] [batch_size:int] [eval_split:str]
- epochs: number of epochs you want to train for
- batch_size: the number of sentences per batch during training and evaluation
- eval_split: choose on which split to evaluate, default is set to "dev"

For instance (on Windows):     C:\\Users\user\\Documents\\AdvNLP\\main.py 3 200 test
