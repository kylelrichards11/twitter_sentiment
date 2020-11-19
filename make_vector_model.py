import time
import argparse

import numpy as np
import pandas as pd
from multiprocessing import cpu_count

from data_preparations.cleaner import Cleaner
from data_preparations.loader import load_data, load_indiv_data, load_cleaned
from data_preparations.text_features import text_feature_maker
from classifiers.vector_classifier import VectorModel
from utils import print_hist, print_time

def remove_empty(data):
    """ Removes tweets with no words

    Parameters
    ----------
    data : pd.DataFrame - data containing tweets

    Returns
    -------
    pd.DataFrame data with empty tweets removed

    """
    lengths = data['tokens'].apply(lambda x : len(x))
    return data.drop(lengths[lengths == 0]).reset_index(drop=True)

def save_probabilities(probs, folder, index=True):
    """ Saves the probabilities to a csv file in the specified folder

    Parameters
    ----------
    probs : pd.DataFrame - positive and negative probabilities

    folder : str - the folder to save the csv file in

    index : bool (default=True) - whether or not to save the index

    Returns
    -------
    None
    """
    probs = pd.DataFrame(probs, columns=['neg_prob', 'pos_prob'])
    folder += '.csv'
    probs.to_csv(folder, index=index)

def save_predictions(preds, folder):
    """ Saves the predictions in the required format to a csv file in the specified folder

    Parameters
    ----------
    preds : np.ndarray - array of predictions

    folder : str - the folder to save the csv file in

    Returns
    -------
    None
    """
    preds = pd.Series(preds.tolist(), dtype=np.int8)
    ids = pd.Series(range(1, preds.shape[0]+1), dtype=np.int16)
    final_df = pd.concat((ids, preds), axis=1)
    final_df.columns = ['Id', 'Prediction']
    final_df.to_csv(f'{folder}/test_predictions.csv', index=False)

def remove_underscores(data):
    """ Removes the underscores from the correlated words data

    Parameters
    ----------
    data : pd.DataFrame - the data with the tweets

    Returns
    -------
    pd.DataFrame - the data with underscores removed

    """
    data['tweets'] = data['tweets'].apply(lambda tweet : tweet.replace('_', ''))
    return data

def main():
    """ Trains a vector embedding model to make probabilities for sentiment

    Options
    -------
    --cleaned : int - the number of the cleaned dataset. If none, full data is loaded and cleaned 
    using the default cleaning function

    --corr (default=False) - whether or not to use the correlation datasets

    --dim : int - the dimension of the vector embedding to create. The default is 30.

    --epochs : int - the number of epochs to train for. The defualt is 3.

    --lstm (default=False) - whether or not to use an lstm after the cnn in the nueral network

    --max_len : int - the maximum tweet length allowed when vectorizing

    --sample : float = random sample of full data to use

    --save : str - save the embeddings and tokenizer in the specified folder

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned', type=int, help="Number for cleaned data file")
    parser.add_argument('--corr', action='store_const', const=True, help="Whether or not to use correlation datasets")
    parser.add_argument('--dim', type=int, help="Dimension of embeddings to create")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('--lstm', action='store_const', const=True, help="whether or not to use an lstm after the cnn in the nueral network")
    parser.add_argument('--max_len', type=int, help="The maximum tweet length allowed when vectorizing")
    parser.add_argument('--save', type=str, help="Folder to save everything in")
    parser.add_argument('--sample', type=float, help="random sample of full data to use")
    args = parser.parse_args()

    all_stime = time.perf_counter()

    threads_avail = cpu_count()
    print(f"\n{threads_avail} Threads Available\n")

    ## Load Data
    if args.cleaned is not None:
        full_data, test_data = load_cleaned(args.cleaned)
    elif args.corr:
        full_data = load_data(False, corr=True)
        full_data = full_data.reset_index(drop=True)
        test_data = load_indiv_data("corr_test")
        full_data = remove_underscores(full_data)
        test_data = remove_underscores(test_data)
    else:
        full_data = load_data(False)
        full_data = full_data.reset_index(drop=True)
        test_data = load_indiv_data("test")

    sample_size = 1 if args.sample is None else args.sample
    train_data = full_data.sample(frac=sample_size).reset_index(drop=True)
    train_data = Cleaner().remove_duplicates(train_data).reset_index(drop=True)

    ## Tokenize
    tfm = text_feature_maker(num_threads=threads_avail)
    train_data = tfm.tokenize(train_data)
    full_data = tfm.tokenize(full_data)
    test_data = tfm.tokenize(test_data)

    train_data = remove_empty(train_data)
    train_data = train_data.reset_index(drop=True)

    ## Train Custom Model
    stime = time.perf_counter()
    dim = 30 if args.dim is None else args.dim
    epochs = 3 if args.epochs is None else args.epochs
    max_len = 50 if args.max_len is None else args.max_len

    custom_model = VectorModel(train_data, embedding_dim=dim, lstm=args.lstm, max_sequence_length=max_len)
    train_args = {'epochs':epochs, 'shuffle':True, 'validation_split':0.1, 'batch_size':32, 'verbose':False}
    history = custom_model.fit(train_args)
    etime = time.perf_counter()
    print_hist(history)
    print_time(stime, etime, prefix='Custom Training: ')

    # Generate Probabilities and Predictions
    stime = time.perf_counter()
    full_probabilities_cust = custom_model.get_probabilities(full_data)
    test_probabilities_cust = custom_model.get_probabilities(test_data)
    test_predictions = custom_model.predict(test_data)
    etime = time.perf_counter()
    print_time(stime, etime, prefix='Custom Probabilities: ')

    # Save
    if args.save is not None:
        print(f"Saving to vector_models/{args.save}")
        save_probabilities(full_probabilities_cust, f'vector_models/{args.save}/custom_{dim}d_full_probs')
        save_probabilities(test_probabilities_cust, f'vector_models/{args.save}/custom_{dim}d_test_probs')
        save_predictions(test_predictions, f'vector_models/{args.save}')
        custom_model.save(args.save)

    all_etime = time.perf_counter()
    print_time(all_stime, all_etime, prefix='Total Time: ')

if __name__ == "__main__":
    main()
