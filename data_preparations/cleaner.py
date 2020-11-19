import re
import string
import glob
import time
import math
from multiprocessing import Pool

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import emot

from utils import print_time

##################################################################################################################
## BASE CLEANER
##################################################################################################################

class _BaseCleaner():
    """ Class to clean tweets before feature extraction. Private, should only be called by Cleaner.
    Specifically it can:

    1. Remove urls
    2. Remove digits
    3. Remove newline characters
    4. Remove punctuation
    5. Reduce lengthening of letters
    6. Perform stemming
    7. Perform lemmatization
    8. Replace emoticons
    9. Replace abbreviations
    10. Remove stopwords
    11. Fix misspellings

    Attributes
    ----------
    external_resources_path : string - the path to the external resources folder

    tweets_col : string, default="tweets" - the name of the column containing the tweets
    """

    def __init__(self, tweets_col="tweets"):
        """ Init a BaseCleaner object

        Parameters
        ----------
        tweets_col : string, default="tweets" - the name of the column containing the tweets
        """
        self.tweets_col = "tweets"
        self.path = "external_resources"

    def remove_digits_alone(self, data):
        """ Removes numbers and digits on their own (not those that are part of words containing other characters)"""
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub('\b[0-9]*\b', '', tweet))
        return data

    def remove_words_with_digits(self, data):
        """ Removes words that contain a digit (ex: 2morrow is completely removed (not only 2), same for 4th)"""
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub('\S*\d+\S*', '', tweet))
        return data

    def remove_repeated_letters(self, data):
        """ Removes words that are composed of a single letter or a repetition of the same letter"""
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub('\b(\w)\1*\b', '', tweet))
        return data

    def remove_spaces(self, data):
        """ Removes unecessary spaces"""
        # Remove sequences of several spaces to only one
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : ' '.join([w for w in tweet.split() if len(w)>1]))

        # Remove some spaces, leave only one space between each word
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub(' {2,}', ' ', tweet))

        # Remove spaces at the beginning and the end of the tweet
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub('(^ +)|( +$)', '', tweet))
        return data

    def remove_empty_tweets(self, data):
        """ Removes empty tweets"""
        data = data[data.tweets != ""]
        return data

    def remove_urls(self, data):
        """ Removes any urls from the given data"""
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub(r'https?://\S+', '', tweet))
        return data

    def remove_punctuation(self, data):
        """ Removes punctuation from the tweets in the given data. Specifically
        !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ are removed
        """
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub(f"[{string.punctuation}]", '', tweet))
        return data

    def remove_digits(self, data):
        """ Removes digits from the tweets in the given data."""
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : re.sub('\d+', '', tweet))
        return data

    def remove_newline(self, data):
        """ Removes the newline character from a string """
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet: tweet.replace('\n', ''))
        return data

    def remove_words(self, data, remove):
        """ Removes words from the tweets in the given data."""
        # Helper function
        def rw(tweet):
            words = tweet.split()
            filtered = [w for w in words if not w in remove]
            return " ".join(filtered)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet: rw(tweet))
        return data

    def _stem(self, data, stemmer):
        """ Stems the words in the given data's tweets"""
        # Helper function
        def stm(tweet):
            words = tweet.split()
            stemmed_words = [stemmer.stem(word) for word in words]
            return " ".join(stemmed_words)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet: stm(tweet))
        return data

    def porter_stem(self, data, porter):
        return self._stem(data, porter)

    def lancaster_stem(self, data, lancaster):
        return self._stem(data, lancaster)

    def snowball_stem(self, data, snowball):
        return self._stem(data, snowball)

    def lemmatize(self, data, lemmatizer):
        """ Lemmatizes the words in the given data's tweets"""
        # Helper function
        def lmt(tweet):
            words = tweet.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized_words)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet: lmt(tweet))
        return data

    def reduce_lengthening(self, data, pattern):
        """ Changes any alphabetical character that repeats itself 3 or more
        times in a row to repeating itself 2 times in a row. For example,
        "helllllloo" becomes "helloo"
        """
        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet: pattern.sub(r"\1\1", tweet))
        return data

    def fix_spelling(self, data, spell):
        """ Attempts to fix any misspelled words. Note: This takes ~100 seconds for 250 tweets, which is too long.
        When I ran on the old version it took ~220 seconds for 250 tweets.
        """
        # Helper function
        def fix(tweet):
            words = tweet.split()
            reconstructed = []
            for word in words :
                if (spell.unknown([word]) != set()) :
                    reconstructed += [spell.correction(word)]
                else :
                    reconstructed += [word]
            return " ".join(reconstructed)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : fix(tweet))
        return data

    def replace_abbreviations(self, data, slang_df):
        """ Replaces abbreviations with their long form. For example,
        "brb" becomes "be right back"
        """
        def abb(tweet):
            words = tweet.split()
            reconstructed = []
            for word in words :
                if (word in list(slang_df.abbrev)) :
                    reconstructed += [list(slang_df.expr)[list(slang_df.abbrev).index(word)]]
                else :
                    reconstructed += [word]
            return " ".join(reconstructed)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : abb(tweet))
        return data

    def replace_emoticon(self, data, emote_df):
        """ Replaces emoticons with their meaning. For example ":)" becomes "happy" """
        def emote(tweet):
            words = tweet.split()
            for i in range (len(words)) :
                if words[i] in [":d", ";d", ":p", ";p"]:
                    words[i] = "happy"
                elif words[i] == "<3":
                    words[i] = "love"
                elif words[i] in ["</3", ":/"]:
                    words[i] = "sad"
                elif words[i] == ":" and len(words) > i+1:
                    if words[i+1] == "*":
                        words[i] = "kiss"
                    elif words[i+1] == "-":
                        words[i] = "sad"
                    elif words[i+1] == "o":
                        words[i] = "surprised"
                else :
                    res = emot.emoticons(words[i])
                    if type(res) is dict and res["flag"] :
                        word = res["mean"][0]
                        if (word in list(emote_df.meaning)) :
                            word = emote_df.word[list(emote_df.meaning).index(word)]
                        words[i] = word
            return " ".join(words)

        data.loc[:, self.tweets_col] = data.loc[:, self.tweets_col].apply(lambda tweet : emote(tweet))
        return data

##################################################################################################################
## CLEANER
##################################################################################################################

class Cleaner():
    """ Class to clean tweets before feature extraction. Specifically it can

    1. Remove duplicate tweets
    2. Remove urls
    3. Remove digits
    4. Remove punctuation
    5. Reduce lengthening of letters
    6. Perform stemming
    7. Perform lemmatization
    8. Replace emoticons
    9. Replace abbreviations
    10. Remove stopwords
    11. Fix misspellings

    Attributes
    ----------
    external_resources_path : string - the path to the external resources folder

    base : object - the base class containing the single thread versions of the functions

    tweets_col : string, default="tweets" - the name of the column containing the tweets

    num_threads : int, default=8 - the number of parallel threads to use. Even if this amount
    of threads is not available, the data will be split up into this many DataFrames
    """
    def __init__(self, tweets_col="tweets", num_threads=8):
        """ Init a Cleaner object

        Parameters
        ----------
        tweets_col : string, default="tweets" - the name of the column containing the tweets

        num_threads : int, default=8 - the number of parallel threads to use. Even if this amount
        of threads is not available, the data will be split up into this many DataFrames
        """
        self.path = "external_resources"
        self.tweets_col = tweets_col
        self.num_threads = num_threads
        self._base = _BaseCleaner(tweets_col)

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

    def remove_digits_alone(self, data, verbose=False):
        """ Removes numbers and digits on their own (not those that are part of words containing other characters

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with digits removed
        """

        if verbose:
            print("Removing digits alone... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_digits_alone, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_words_with_digits(self, data, verbose=False):
        """ Removes words that contain a digit (ex: 2morrow is completely removed (not only 2), same for 4th)

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with words with digits removed
        """

        if verbose:
            print("Removing words with digits... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_words_with_digits, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_repeated_letters(self, data, verbose=False):
        """ Removes words that are composed of a single letter or a repetition of the same letter

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with repeated letters removed
        """

        if verbose:
            print("Removing repeated letters... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_repeated_letters, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_spaces(self, data, verbose=False):
        """ Removes unnecessary spaces

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with unnecessary spaces removed
        """

        if verbose:
            print("Removing spaces... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_spaces, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_empty_tweet(self, data, verbose=False):
        """ Removes empty tweets

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with empty tweets removed
        """

        if verbose:
            print("Removing empty tweets... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_empty_tweets, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_duplicates(self, data, verbose=False):
        """ Removes any subsequent duplicates of rows where the tweets are the same. All
        columns of the row are deleted. NOTE: Should not be parallelized since duplicates
        could be in two different data splits (also its only ~2 seconds anyways).

        Parameters
        ----------
        data : pd.DataFrame - the data potentially with duplicate rows

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the data with duplicates removed
        """
        if verbose:
            print("Removing Duplicates... ", end="")
        stime = time.perf_counter()
        data = data.drop_duplicates(subset=self.tweets_col)
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_urls(self, data, verbose=False):
        """ Removes any urls from the given data

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with urls removed
        """
        if verbose:
            print("Removing Urls... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_urls, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_punctuation(self, data, verbose=False):
        """ Removes punctuation from the tweets in the given data. Specifically
        !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ are removed

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with punctuation removed

        """
        if verbose:
            print("Removing Punctuation... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_punctuation, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_digits(self, data, verbose=False):
        """ Removes digits from the tweets in the given data.

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with digits removed

        """
        if verbose:
            print("Removing Digits... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_digits, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_newline(self, data, verbose=False):
        """ Removes the newline character from a string

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - The input data without newline character
        """
        if verbose:
            print("Removing Newlines... ", end="")
        stime = time.perf_counter()
        data = self._parallelize(self._base.remove_newline, (data,))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def remove_words(self, data, words=None, verbose=False):
        """ Removes words from the tweets in the given data.

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        words : list, default=None - the list of words to remove. If None, default
        stopwords are used

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with punctuation removed

        """
        if verbose:
            print("Removing Words... ", end="")
        stime = time.perf_counter()
        if words is None:
            my_stop = stopwords.words('english')
            my_stop = [word for word in my_stop if word not in ["n't", "not", "no"]]
            my_stop += ["user", "<url>", "url", "<user>"]
        else:
            my_stop = words
        data = self._parallelize(self._base.remove_words, (data, my_stop))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def stem(self, data, verbose=False):
        """ Stems the words in the given data's tweets using the Porter Stemmer

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with Porter stemming applied

        """
        if verbose:
            print("Porter Stemming... ", end="")
        stime = time.perf_counter()
        porter = PorterStemmer()
        data = self._parallelize(self._base.porter_stem, (data, porter))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def lancaster_stem(self, data, verbose=False):
        """ Stems the words in the given data's tweets using the Lancaster Stemmer

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with Lancaster stemming applied

        """
        if verbose:
            print("Lancaster Stemming... ", end="")
        stime = time.perf_counter()
        lancaster = LancasterStemmer()
        data = self._parallelize(self._base.lancaster_stem, (data, lancaster))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def snowball_stem(self, data, verbose=False):
        """ Stems the words in the given data's tweets using the Snowball Stemmer

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with Snowball stemming applied

        """
        if verbose:
            print("Snowball Stemming... ", end="")
        stime = time.perf_counter()
        snowball = SnowballStemmer("english")
        data = self._parallelize(self._base.snowball_stem, (data, snowball))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def lemmatize(self, data, verbose=False):
        """ Lemmatizes the words in the given data's tweets

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with lemmatization applied

        """
        if verbose:
            print("Lemmatizing... ", end="")
        stime = time.perf_counter()
        lemmatizer = WordNetLemmatizer()
        data = self._parallelize(self._base.lemmatize, (data, lemmatizer))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def reduce_lengthening(self, data, verbose=False):
        """ Changes any alphabetical character that repeats itself 3 or more
        times in a row to repeating itself 2 times in a row. For example,
        "helllllloo" becomes "helloo"

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with triples fixed
        """
        if verbose:
            print("Reducing Lengthenings... ", end="")
        stime = time.perf_counter()
        pattern = re.compile(r"([a-z])\1{2,}")
        data = self._parallelize(self._base.reduce_lengthening, (data, pattern))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def fix_spelling(self, data, verbose=False):
        """ Attempts to fix any misspelled words. Note: This takes ~100 seconds for 250 tweets, which is too long.
        When I ran on the old version it took ~220 seconds for 250 tweets. Became ~50 seconds for 250 tweets
        with parallelization.

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with misspellings fixed

        """
        if verbose:
            print("Fixing Spelling... ", end="")
        stime = time.perf_counter()
        spell = SpellChecker()
        data = self._parallelize(self._base.fix_spelling, (data, spell))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def replace_abbreviations(self, data, verbose=False):
        """ Replaces abbreviations with their long form. For example,
        "brb" becomes "be right back"

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with abbreviations replaced

        """
        if verbose:
            print("Replacing Abbreviations... ", end="")
        stime = time.perf_counter()
        slang_df = pd.read_csv(f"{self.path}/slang.csv")
        data = self._parallelize(self._base.replace_abbreviations, (data, slang_df))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def replace_emoticon(self, data, verbose=False):
        """ Replaces emoticons with their meaning. For example ":)" becomes "happy"

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print when the function is running

        Returns
        -------
        pd.DataFrame - the input data with emoticons replaced

        """
        if verbose:
            print("Replacing Emoticons... ", end="")
        stime = time.perf_counter()
        emote_df = pd.read_csv(f"{self.path}/trad_emoticon.csv")
        data = self._parallelize(self._base.replace_emoticon, (data, emote_df))
        etime = time.perf_counter()
        if verbose:
            print_time(stime, etime)
        return data

    def run_all(self, data, verbose=False):
        """ Runs all cleaning functions in the following order

        1. Removes duplicate tweets
        2. Removes urls
        3. Removes digits
        4. Removes newline characters
        5. Removes punctuation
        6. Reduces lengthening of letters
        7. Performs stemming
        8. Performs lemmatization
        9. Replaces emoticons
        10. Replaces abbreviations
        11. Removes stopwords

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print which function is running

        Returns
        -------
        pd.DataFrame - the cleaned data

        """

        data = self.remove_duplicates(data, verbose=verbose)
        data = self.remove_urls(data, verbose=verbose)
        data = self.remove_digits(data, verbose=verbose)
        data = self.remove_newline(data, verbose=verbose)
        data = self.remove_punctuation(data, verbose=verbose)
        data = self.reduce_lengthening(data, verbose=verbose)
        data = self.stem(data, verbose=verbose)
        data = self.lemmatize(data, verbose=verbose)
        data = self.replace_emoticon(data, verbose=verbose)
        data = self.replace_abbreviations(data, verbose=verbose)
        data = self.remove_words(data, verbose=verbose)
        data = self.fix_spelling(data, verbose=verbose)
        return data

    def glove_clean(self, data, listClean, verbose=False):
        """ Run the clean function listed in listClean

        Parameters
        ----------
        data : pd.DataFrame - the data containing tweets

        verbose : bool, default=False - whether or not to print which function is running

        Returns
        -------
        pd.DataFrame - the cleaned data

        """
        list_all_clean = ["remove_duplicates","remove_urls","remove_digits","remove_newline","remove_punctuation",
                          "reduce_lengthening","stem","lemmatize","replace_emoticon","replace_abbreviations","remove_words"]
        for el in listClean:
            if el not in list_all_clean:
                print(f"Wrong name of function {el}")
                return data

        if "remove_duplicates" in listClean:
            data = self.remove_duplicates(data, verbose=verbose)

        if "remove_urls" in listClean:
            data = self.remove_urls(data, verbose=verbose)

        if "remove_digits" in listClean:
            data = self.remove_digits(data, verbose=verbose)

        if "remove_newline" in listClean:
            data = self.remove_newline(data, verbose=verbose)

        if "remove_punctuation" in listClean:
            data = self.remove_punctuation(data, verbose=verbose)

        if "reduce_lengthening" in listClean:
            data = self.reduce_lengthening(data, verbose=verbose)

        if "stem" in listClean:
            data = self.stem(data, verbose=verbose)

        if "lemmatize" in listClean:
            data = self.lemmatize(data, verbose=verbose)

        if "replace_emoticon" in listClean:
            data = self.replace_emoticon(data, verbose=verbose)

        if "replace_abbreviations" in listClean:
            data = self.replace_abbreviations(data, verbose=verbose)

        if "remove_words" in listClean:
            data = self.remove_words(data, verbose=verbose)

        return data