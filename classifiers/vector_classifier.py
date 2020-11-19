import re
import string
from multiprocessing import Pool
import pickle
import json

import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model, load_model

####################################################################################################
## DATA HANDLING HELPERS
####################################################################################################

def split_label(data, label='type'):
    """ Splits the labels column into two columns, one for positive and one for negative.
    Assumes negative is 0 and positive is 1
    
    Parameters
    ----------
    data : pd.DataFrame - the data with the label
    
    label : string, default='label' - the name of the column with the label
    
    Returns
    -------
    pd.DataFrame - the input data with the label column removed and the two new label columns
    added
    
    """
    pos = data[label]
    neg = np.invert(pos.astype(bool)).astype(int)
    
    pos.name = f"{label}_pos"
    neg.name = f"{label}_neg"
    data = data.drop(columns=label)
    data = pd.concat((data, neg, pos), axis=1)
    return data

def split_x_y(data, label='type'):
    """ Splits the input dataframe into x and y based on the given y columns
    
    Parameters
    ----------
    data : pd.DataFrame - the data with x and y
    
    label : string or [string], default="type" - the name(s) of the y column(s)
    
    Returns
    -------
    x : pd.DataFrame - the data with y columns removed
    y : pd.Series or pd.DataFrame - the y columns
    
    """
    x = data.drop(columns=label)
    y = data[label]
    return x, y

####################################################################################################
# VECTOR PROCESSOR
####################################################################################################

class VectorProcessor():
    """ A class that takes strings and extracts useful information out using vectors

    Attributes
    ----------
    tokens : str, default='tokens' - the name of the column containing the words to get features for. The values
    in the column must be lists of strings.

    max_sequence_length : int, default=50 - the maximum number of words per input to consider when getting word 
    embeddings.

    embedding_dim : int, default=300 - the length of the vector to use

    vectors : np.array, defaule=None - previously trained vectors to use
    """

    def __init__(self, tokens='tokens', max_sequence_length=50, embedding_dim=300, vectors=None, tokenizer=None):
        self.tokens = tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.vectors = vectors
        self.tokenizer = tokenizer

    def _parallelize(self, func, args):
        """ Private: helper function to parallelize a given function on the given DataFrame

        Parameters
        ----------
        func : function - the function to parallelize

        args : tuple - the arguments to the function. The first argument must be a pd.DataFrame

        Returns
        -------
        pd.DataFrame - the output of the given function

        """
        data, *args = args
        data_split = np.array_split(data, self.num_threads)
        thread_args = [[ds] + args for ds in data_split]
        
        pool = Pool(self.num_threads)
        data = pd.concat(pool.starmap(func, thread_args))
        pool.close()
        pool.join()
        return data

            
    def get_vocab(self, data):
        """ Gets all unique words from data

        Parameters
        ----------
        data : pd.DataFrame - the data to extract vocabulary from

        token_col : string, default='tokens' - the column containing the tokenized string

        Returns
        -------
        [string] - All unique words

        """
        all_words = [word for tokens in data[self.tokens] for word in tokens]
        vocab = sorted(list(set(all_words)))
        return vocab

    def train_tokenizer(self, train_data):
        """ Trains the tokenizer to convert strings of space separated words into lists of ints where each int
        represents a unique word. This method should only be called using training data as the model will not
        know how to tokenize words it does not know. This method is called by get_features() if it was not 
        previously called.

        Parameters
        ----------
        train_data : pd.DataFrame - the data to train the tokenizer on

        Returns
        -------
        int - the number of unique words tokenized 
        """
        vocab = self.get_vocab(train_data)
        num_words = len(vocab)
        tokenizer = Tokenizer(num_words=num_words, lower=True, char_level=False)
        tokenizer.fit_on_texts(train_data[self.tokens])
        self.tokenizer = tokenizer
        return num_words
        
    def get_average_vector(self, tokens_list, generate_missing=False):
        """ Gets the average vector for all tokens in the given list

        Parameters
        ----------
        tokens_list : [string] - a list of individual words to get vectors for

        generate_missing : bool, default=False - whether or not to generate a random
        vector if no vector can be obtained for the word. If False and no vector can
        be obtained the word is given a vector of all 0

        Returns
        -------
        numpy.ndarray : the average vector for the list of tokens

        """
        if self.vectors is None:
            print("No vectors provided to get_average_vector")
            return
        
        if len(tokens_list) < 1:
            return np.zeros(self.embedding_dim)
        if generate_missing:
            vectorized = [self.vectors[word] if word in self.vectors else np.random.rand(self.embedding_dim) for word in tokens_list]
        else:
            vectorized = [self.vectors[word] if word in self.vectors else np.zeros(self.embedding_dim) for word in tokens_list] #TODO: Skip this word all together?
        length = len(vectorized)
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
        return averaged
    
    def get_vector(self, tokens_list, generate_missing=False):
        """ Gets the average vector for all tokens in the given list

        Parameters
        ----------
        tokens_list : [string] - a list of individual words to get vectors for

        generate_missing : bool, default=False - whether or not to generate a random
        vector if no vector can be obtained for the word. If False and no vector can
        be obtained the word is given a vector of all 0

        Returns
        -------
        numpy.ndarray : the average vector for the list of tokens

        """
        if self.vectors is None:
            print("No vectors provided to get_average_vector")
            return
        
        if len(tokens_list) < 1:
            return np.zeros(self.embedding_dim)
        if generate_missing:
            vectorized = [self.vectors[word] if word in self.vectors else np.random.rand(self.embedding_dim) for word in tokens_list]
        else:
            vectorized = [self.vectors[word] if word in self.vectors else np.zeros(self.embedding_dim) for word in tokens_list]
        return vectorized
        
    def get_vector_embeddings(self, data, generate_missing=False):
        """ Gets the vector embeddings for each word. Each word is mapped to a vector by the model

        Parameters
        ----------
        data : pd.DataFrame - the data to get the word embeddings for

        generate_missing : bool, default=False - whether or not to generate a random
        vector if no vector can be obtained for the word. If False and no vector can
        be obtained the word is skipped

        Returns
        -------

        """
        if self.vectors is None:
            print("No vectors provided to get_vector_embeddings")
            return
        
        def helper(data, generate_missing):
            embeddings = data['tokens'].apply(lambda x: self.get_vector(x, generate_missing=generate_missing))
            return embeddings

        #return self._parallelize(helper, (data, generate_missing))
        return helper(data, generate_missing)
    
    def get_sequences(self, data):
        """ Translates the tokens into unique ints for each token. If no tokenizer for this object has 
        been trained, then this function will call train_tokenizer() on the input data. This should only 
        be done with the training data.
        
        Parameters
        ----------
        data : pd.DataFrame - the data with the tokens
        
        Returns
        -------
        pd.DataFrame - the sequences with each word as a column
        
        """
        if self.tokenizer is None:
            self.train_tokenizer(data)
            
        sequences = self.tokenizer.texts_to_sequences(data[self.tokens].tolist())
        sequences = pad_sequences(sequences, maxlen = self.max_sequence_length)
        
        sequences = pd.Series(sequences.tolist(), name='sequences')
        return pd.concat((sequences, data), axis=1)
        
    def get_embedding_weights(self, data):
        """ Gets the weights for each tokenized word. These weights are used as an input layer to a neural
        network in order to combine the word vectors.
        
        Parameters
        ----------
        data : pd.DataFrame - the data to get the features for
        
        """
        if self.vectors is None:
            return None
        
        print("Getting embedding weights")
        word_index = self.tokenizer.word_index
        weights = np.zeros((len(word_index)+1, self.embedding_dim))
        for word, index in word_index.items():
            weights[index, :] = self.vectors[word] if word in self.vectors else np.random.rand(self.embedding_dim)
        return weights

    def get_avg_features(self, data, keep_cols=True):
        """ Gets the avg vector features of the given data. The vectors for each word in the tokenized
        string are averaged to produce a vector for the entire string. If no tokenizer for this object has 
        been trained, then this function will call train_tokenizer() on the input data. This should only 
        be done with the training data.

        Parameters
        ----------
        data : pd.DataFrame - the data to get the features for

        keep_cols : bool, default True - True to keep the columns of the original DataFrame in the output,
        False to return a DataFrame with only the new features

        Returns
        -------
        pd.DataFrame - the average vector features

        """
        if self.tokenizer is None:
            self.train_tokenizer(data)

        # Get embeddings for each word
        print("Getting vector embeddings")
        embeddings = self.get_vector_embeddings(data, generate_missing=True)

        # Change features to DataFrame
        cols = []
        for i in range(self.embedding_dim):
            cols.append(f"vec_{i}")
        features = pd.DataFrame(embeddings, columns=cols)

        if keep_cols:
            return pd.concat((data, features), axis=1)
        return features

    def save_tokenizer(self, folder):
        """ Saves the trained tokenizer to the specified file

        Parameters
        ----------
        folder : str - the folder and filename to save the object at

        Returns
        -------
        None
        """
        with open(f'vector_models/{folder}/tokenizer.pickle', 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


####################################################################################################
# VECTOR NN
####################################################################################################

class VectorNN():
    """
    A class to create a convolutional neural network using the vector embeddings
    Based on: https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7
    
    Attributes
    ----------

    max_sequence_length : int, default=50 - the maximum number of words per input to consider when getting word 
    embeddings.

    embedding_dim : int, default=300 - the length of the vector to use
    """
    def __init__(self, embeddings, num_words, lstm=False, max_sequence_length=50, embedding_dim=300, nn=None):
        self.lstm = lstm
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.model = nn
        if nn is None:
            self._define_nn(embeddings, num_words)
    
    def _define_nn(self, embeddings, num_words):
        """ Creates the neural network model with the provided embeddings
        
        Parameters
        ----------
        embeddings : numpy.ndarray or None - a mapping of tokenized word ids to vectors. If none, then embeddings are trained
        with the rest of the network

        num_words : the number of unique words the model will be trained on
    
        Returns
        -------
        None
        
        """
        if embeddings is None:
            embedding_layer = Embedding(num_words + 1, self.embedding_dim, input_length=self.max_sequence_length)
        else:
            embedding_layer = Embedding(num_words + 1, self.embedding_dim, weights=[embeddings], input_length=self.max_sequence_length, trainable=False)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        convs = []

        for filter_size in range(2,7):
            conv_layer = Conv1D(filters=250, kernel_size=filter_size, activation='relu')(embedded_sequences)
            if self.lstm:
                convs.append(conv_layer)
            else:
                pool_layer = GlobalMaxPooling1D()(conv_layer)
                convs.append(pool_layer)

        cnn_out = concatenate(convs, axis=1)

        if self.lstm:
            cnn_out = LSTM(256)(cnn_out)
            print("Using lstm")

        x = Dense(256, activation='relu')(cnn_out)
        x = Dense(64, activation='relu')(x)
        preds = Dense(2, activation='sigmoid')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        self.model = model
        
    def train(self, train_data, sequences='sequences', label='type', kwargs={}):
        """ Trains the neural network model
        
        Parameters
        ----------
        train_data : pd.DataFrame - the training data with sequences and labels
        
        sequences : string, default='sequences' - the name of the column containing the sequences
        
        label : string, default='type' - the name of the column containing the labels
        
        kwargs : dict, default={} - arguments to pass to fit()
        
        Returns
        -------
        keras.callbacks.callbacks.History : training history 
        """
        train_data = split_label(train_data, label)
        _, train_y = split_x_y(train_data, [f"{label}_pos", f"{label}_neg"])
        seq = train_data[sequences]
        seq = np.array(pd.DataFrame.from_dict(dict(zip(seq.index, seq.values)))).T
        return self.model.fit(seq, train_y, **kwargs)
        
    def get_embeddings(self):
        """ Gets the word embeddings used in the NN model. This is useful only if you are 
        training your own embeddings

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray : array of embedding weights
        """
        for layer in self.model.layers:
            if type(layer) == Embedding:
                return layer.get_weights()[0]   # There should only be one embedding layer and one numpy array in the list

    def get_probabilities(self, data, sequences='sequences'):
        """ Gets the probabilities for each sample
        
        Parameters
        ----------
        data : pd.DataFrame - the input data to use the model on
        
        sequences : string, default='sequences' - the name of the column containing the sequences
        
        Returns
        -------
        np.array - the probabilities of each class, where each column is a class
        """
        seq = data[sequences]
        seq = np.array(pd.DataFrame.from_dict(dict(zip(seq.index, seq.values)))).T
        return self.model.predict(seq)

    def save(self, folder):
        """ Saves the entire keras model to the specified folder. This includes the embeddings layer 
        
        Parameters
        ----------
        folder: str - the folder to save the model to

        Returns
        -------

        None
        """
        self.model.save(f"vector_models/{folder}/NN")

####################################################################################################
# VECTOR MODEL
####################################################################################################

class VectorModel():
    """ This class is used to get vector features or to use the vectors to make predictions
    
    There are 3 ways to get information using this class
    
    1. Get average vectors for each sample
    2. Get probabilities to use as features in future models
    3. Directly make predictions
    
    Attributes
    ----------
    train_data : pd.DataFrame - the data to train the model on
    
    label : string, default='type' - the name of the column containing the y values
    
    tokens : string, default='tokens' - the name of the column containing the tokenized input

    lstm : bool, default=False - whether or not to use an lstm after the cnn in the nueral network
    
    vectors : object, default=None - the vectors
    
    max_sequence_length : int, default=50 - the maximum number of words per input to consider when getting word 
    embeddings.

    embedding_dim : int, default=300 - the length of the vector to use
    """
    
    def __init__(self, train_data, label='type', tokens='tokens', lstm=False, vectors=None, embedding_dim=300, max_sequence_length=50, tokenizer=None, num_words=None, nn=None):
        
        self.train_data = train_data
        self.label = label
        self.tokens = tokens
        self.lstm = lstm
        self.vectors = vectors
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_words = num_words

        self.vec_pr = VectorProcessor(tokens=tokens, max_sequence_length=self.max_sequence_length, embedding_dim=self.embedding_dim, vectors=self.vectors, tokenizer=tokenizer)

        if tokenizer is None:
            self.num_words = self.vec_pr.train_tokenizer(train_data)
        
        if nn is None:
            self.vec_nn = None
        else:
            self.vec_nn = VectorNN(None, None, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim, nn=nn, lstm=lstm)

    @classmethod
    def load(cls, folder, vectors=None):
        """ Loads a model from the specified folder

        Parameters
        ----------
        folder : str - the folder where the model was saved

        vectors : obj (default=None) - the pretrained vectors used to create the model. If no pretrained vectors were used, leave as None. If vectors were used this must not be None

        Returns
        -------
        VectorModel : the model ready to go

        """
        path = f"vector_models/{folder}"
        with open(f"{path}/info.json", 'r') as f:
            info = json.load(f)

        if info['vectors'] and vectors is None:
            print("Must pass vectors when loading model from memory")
            exit()

        with open(f"{path}/tokenizer.pickle", 'rb') as f:
            tokenizer = pickle.load(f)

        nn = load_model(f"{path}/NN")

        vec_model = cls(None, label=info['label'], tokens=info['tokens'], vectors=vectors, embedding_dim=info['embedding_dim'], max_sequence_length=info['max_sequence_length'], tokenizer=tokenizer, num_words=info['num_words'], nn=nn)

        return vec_model
        
    def _probs_to_predictions(self, probabilities):
        """ Converts probabilities of classes to an array of classes. Class 0 is converted to -1

        Parameters
        ----------
        probabilities : array-like - the probabilities of classes where each column is a different class

        """

        classes = np.argmax(probabilities, axis=1)
        return np.where(classes==0, -1, classes)
        
    def get_average_vectors(self, data):
        """ Gets the avg vector features of the given data. The vectors for each word in the tokenized
        string are averaged to produce a vector for the entire string.

        Parameters
        ----------
        data : pd.DataFrame - the data to get the features for

        Returns
        -------
        pd.DataFrame - the average vector features

        """
        return self.vec_pr.get_avg_features(data, keep_cols=False)
        
    def fit(self, kwargs={}):
        """ Trains the neural network using the already provided training data
        
        Parameters
        ----------
        kwargs : dict, default={} - arguments to pass to the keras fit function
        
        Returns
        -------
        keras.callbacks.callbacks.History : training history
        """
        if self.train_data is None:
            print("Training Data was not provided when object was initialized, cannot train nn. If this object was loaded from a file, training is locked.")
            return
        nn_train_data = self.vec_pr.get_sequences(self.train_data)
        embedding_weights = self.vec_pr.get_embedding_weights(self.train_data)
        self.vec_nn = VectorNN(embedding_weights, self.num_words, max_sequence_length=self.max_sequence_length, embedding_dim=self.embedding_dim, lstm=self.lstm)
        return self.vec_nn.train(nn_train_data, label=self.label, kwargs=kwargs)

    def get_embeddings(self):
        """ Gets the word embeddings used in the NN model. This is mostly useful only if you are 
        training your own embeddings

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray : array of embedding weights
        """
        return self.vec_nn.get_embeddings()
    
    def get_probabilities(self, data):
        """ Gets the probabilities for each sample
        
        Parameters
        ----------
        data : pd.DataFrame - the input data to use the model on
        
        Returns
        -------
        np.array - the probabilities of each class, where each column is a class
        
        """
        if self.vec_nn is None:
            print("Must train NN first by calling vector_model.train_nn()")
            return
        nn_data = self.vec_pr.get_sequences(data)
        return self.vec_nn.get_probabilities(nn_data)
    
    def predict(self, data):
        """ Gets the final prediction for each sample
        
        Parameters
        ----------
        data : pd.DataFrame - the input data to use the model on
        
        Returns
        -------
        np.array - the prediction for each sample
        
        """
        probs = self.get_probabilities(data)
        return self._probs_to_predictions(probs)

    def save(self, folder):
        """ Saves the model to the specified folder 
        
        Parameters
        ----------
        folder : str - the folder to save the model in

        Returns
        -------
        None
        """
        # Save embeddings
        self.save_embeddings(folder)

        # Save NN weights
        if self.vec_nn is not None:
            self.vec_nn.save(folder)

        # Save tokenizer
        self.vec_pr.save_tokenizer(folder)
        
        # Save object
        info = {
            'vectors' : self.vectors is not None,
            'label' : self.label,
            'tokens' : self.tokens,
            'embedding_dim' : self.embedding_dim,
            'max_sequence_length' : self.max_sequence_length,
            'num_words': self.num_words
        }
        with open(f"vector_models/{folder}/info.json", 'w') as f:
            json.dump(info, f)

    def save_embeddings(self, folder):
        """ Saves the word embeddings array as a file called embeddings.npy in the specified folder in the vector_models folder

        Parameters
        ----------
        folder : str - the folder to store the embeddings file in

        Returns
        -------
        None
        """
        np.save(f"vector_models/{folder}/embeddings", self.get_embeddings())