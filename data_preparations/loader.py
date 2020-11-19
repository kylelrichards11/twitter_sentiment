import numpy as np
import pandas as pd
from gensim import models

####################################################################################################
# LOAD DATA HELPERS
####################################################################################################

def load_indiv_data(dataset):
    """ Loads the specified dataset from one file

    Parameters
    ----------
    dataset: string - Specifies which dataset to load. Must be one of "pos", "neg", "pos_full", "neg_full", "corr_pos", "corr_neg", "test", or "corr_test".

    Returns
    -------
    pd.DataFrame - The specified data where each row contains one line and its label        
    """

    # Check for valid datasets
    assert(dataset == "pos" or dataset == "neg" or dataset == "pos_full" or dataset == "neg_full" or dataset=="test" or dataset == "corr_pos" or dataset == "corr_neg" or dataset == "corr_test")
    
    # Test data is handled separately
    path = "twitter-datasets"
    
    if dataset == "test":
        path = f"{path}/test_data.txt"

    elif dataset == "corr_pos":
        path = f"{path}/corr_train_pos.txt"

    elif dataset == "corr_neg":
        path = f"{path}/corr_train_neg.txt"

    elif dataset == "corr_test":
        path = f"{path}/corr_test.txt"
    
    else:
        path = f"{path}/train_{dataset}.txt"
    
    # Read text
    lines = pd.Series(list(open(path, "r", encoding="utf8")), name='tweets')
    
    # Test has no labels so return
    if dataset == "test" or dataset=="corr_test":
        return lines.to_frame()

    # Assign label
    label = 1
    if dataset == "neg" or dataset == "neg_full" or dataset == "corr_neg":
        label = 0

    lbl = pd.Series(label*np.ones(lines.shape), name='type')
    return pd.concat((lines, lbl), axis = 1)

def load_data(full, corr=False):
    """ Loads either the small dataset or the full dataset as a Pandas DataFrame
    
    Parameters
    ----------
    full : bool - True if it should load the full dataset, False otherwise

    corr : bool (default=False) - True if it should load the correlation dataset, False otherwise

    Returns
    -------
    pd.DataFrame - The labeled dataset where each row is a line and its label. The column
    name for the tweets is "tweets" and the column name for the label is "type".
    """

    if full:
        pos_df = load_indiv_data("pos_full")
        neg_df = load_indiv_data("neg_full")
    elif corr:
        pos_df = load_indiv_data("corr_pos")
        neg_df = load_indiv_data("corr_neg")
    else:
        pos_df = load_indiv_data("pos")
        neg_df = load_indiv_data("neg")

    return pd.concat((pos_df, neg_df))

def load_cleaned(file_num):
    """ Loads a dataset from the cleaned_datasets folder

    Parameters
    ----------
    file_num : int - the dataset number desired. I.e. `0` loads full_cleaned_0 and test_cleaned_0

    Returns
    -------
    tuple - the full dataset as a DataFrame and the test dataset as a DataFrame

    """
    full_df = pd.read_csv(f'cleaned_datasets/full_cleaned_{file_num}.csv', encoding='utf-8', na_filter=False, dtype={'tweets':str, 'type':float})
    test_df = pd.read_csv(f'cleaned_datasets/test_cleaned_{file_num}.csv', encoding='utf-8', na_filter=False, dtype={'tweets':str})
    return full_df, test_df

####################################################################################################
# LOAD VECTOR HELPERS
####################################################################################################

def load_word2vec_vectors():
    """ Helper function to load the word2vec model from disk. If not on disk, download
    from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    
    Parameters
    ----------
    None

    Returns
    -------
    object - the word2vec model
    
    """
    print("Loading pretrained word2vec vectors (takes ~5 minutes)")
    word2vec_path = 'embeddings/GoogleNews-vectors-negative300.bin.gz'
    return models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def load_glove(filename, dimension):
    """ Helper function to load glove vectors from disk. If not on disk, download from
    https://github.com/stanfordnlp/GloVe

    Parameters
    ----------
    filename : string - the name of the file containing the vectors. Folder is assumed to be
    "embeddings", do NOT include the folder name in the file name

    dimension : int - the dimension of the vectors in the file chosen

    Returns
    -------
    dict : the words mapped to their vectors
    """

    emb_dict = {}
    with open(f'embeddings/{filename}',encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = ' '.join(values[:-dimension])
            vector = np.asarray(values[-dimension:], dtype='float32')
            emb_dict[word] = vector
    return emb_dict