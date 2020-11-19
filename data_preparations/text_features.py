import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import emot
from textblob import TextBlob

class _base_tfm():
    """ A class to make custom features based on the tweets
        
    Attributes
    ----------
    tweet_col : string, default="tweets" - the name of the column containing the tweets
    """
    def __init__(self, tweet_col="tweets"):
        self.tweet_col = tweet_col

    def __init__(self, tweet_col="tweets", num_threads=8):
        self.tweet_col = tweet_col
        self.num_threads = num_threads

    def word_count(self, data, word, word_esc):
        """ Adds the number of the given word in the tweet as a feature"""
        data[f"{word}_cnt"] = data[self.tweet_col].apply(lambda s: len(re.findall(word_esc, s)))
        return data

    def count_to_bool(self, data, col_name, threshold):
        """ Adds a boolean feature to the data which is True if the character count is greater
        or equal to the threshold and False otherwise
        """
        data[f"{col_name}≥{threshold}"] = data[col_name].apply(lambda cnt : cnt >= threshold)
        return data

    def count_all_words(self, data, pattern):
        """ Adds a feature which is the number of words in the tweet """
        data["all_words_cnt"] = data[self.tweet_col].apply(lambda tweet : len(re.findall(pattern, tweet)))
        return data

    def count_emoticons(self, data):
        """ Adds a feature with the total number of emoticons in the tweet """
        def count_emot(tweet) :
            res = emot.emoticons(tweet)
            if (type(res) is dict and res["flag"]) :
                return len(res["value"])
            return 0
        data["emot_cnt"] = data[self.tweet_col].apply(lambda tweet : count_emot(tweet))
        return data

    def extract_emoji(self, data, dico_emotes):
        """ Adds a feature which is the sum of positive and negative emojis. """
        emotes = ['<3',':)',';)',':`)','.)',':d',':p',':`d',';d',';p',';`d', 'xx', 'x',':(','</3',':/',':`)',':*',':o',':-']
        def get_num(tweet):
            numberPositif = 0
            sentence = tweet.split()
            for emoji in emotes:
                numberPositif += dico_emotes[emoji]*sentence.count(emoji)
            for i in range(len(sentence)):
                if sentence[i] == ":" and len(sentence) > i+1:
                    if sentence[i+1] == "*":
                        numberPositif += dico_emotes[':*']
                    elif sentence[i+1] == "-":
                        numberPositif += dico_emotes[':-']
                    elif sentence[i+1] == "o":
                        numberPositif += dico_emotes[':o']
                        
            return numberPositif
        
        data["pos_emoji"] = data[self.tweet_col].apply(lambda tweet : get_num(tweet))
        return data
    

    def count_hashtag(self, data, dico_hash):
        """
        This function assign a value for each tweet in function of the hashtag in the tweet and dictionnary of hashtags
        """
        key = dico_hash.keys()

        def count_hash_mean(tweet) :
            res = 0
            for i in tweet.split():
                if i.startswith("#"):
                    if i in key and (dico_hash[i][1] - dico_hash[i][0] != 0):
                        res += (dico_hash[i][1])/(dico_hash[i][1] - dico_hash[i][0])*100.0 + (dico_hash[i][0])/(dico_hash[i][1] - dico_hash[i][0])*100.0
            return res
        data["hash_tag_mean"] = data[self.tweet_col].apply(lambda tweet : count_hash_mean(tweet))
        return data

    def tweet_polarity(self, data, sia):
        """ Returns the net polarity of the tweet. Based on nltk'ssentiment intensity analyzer scores """
        def polarity_of_tweet(tweet):
            listWord = tweet.split()
            balance = 0
            for word in listWord:
                if sia.polarity_scores(word)['compound'] >= 0.5:
                    balance += 1
                elif sia.polarity_scores(word)['compound'] <= -0.5:
                    balance -= 1
            return balance

        data["polarity"] = data[self.tweet_col].apply(lambda tweet : polarity_of_tweet(tweet))
        return data

    def tokenize(self, data):
        """ Splits the tweets into a list of words in the tweet """
        data["tokens"] = data[self.tweet_col].apply(lambda tweet : tweet.split())
        return data

    def blob_sentiment(self, data):
        """Computes TextBlob sentiment and subjectivity for each tweet """
        return data[self.tweet_col].apply(lambda tweet : pd.Series(TextBlob(tweet).sentiment))

class text_feature_maker():
    """ A class to make custom features based on the tweets
        
    Attributes
    ----------
    tweet_col : string, default="tweets" - the name of the column containing the tweets

    num_threads: int, default=8 - the maximum number of threads to use. Even if this many threads 
    are not available, the data will be split into this many chunks

    _base : object - the base text feature maker containing the unparallelized versions
    """

    def __init__(self, tweet_col="tweets", num_threads=8):
        self.tweet_col = tweet_col
        self.num_threads = num_threads
        self._base = _base_tfm(tweet_col)

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

    def word_count(self, data, word):
        """ Adds the number of the given word in the tweet as a feature

        Parameters
        ----------
        data : pd.DataFrame - the data with the tweets to count words for

        character : string - the word to get count for

        Returns
        -------
        pd.DataFrame - the original data with the count for the tweet appended
        """
        word_esc = re.escape(word)
        return self._parallelize(self._base.word_count, (data, word, word_esc))

    def count_to_bool(self, data, col_name, threshold):
        """ Adds a boolean feature to the data which is True if the character count is greater
        or equal to the threshold and False otherwise

        Parameters
        ----------
        data : pd.DataFrame - the data with the tweets to count characters for

        col_name : string - the name of the column containing the count. Assumes the col_name ends in "_cnt"

        threshold : int - the number of character occurances needed for True
        
        Returns
        -------
        pd.DataFrame - the original data with the boolean feature appended
        """
        return self._parallelize(self._base.count_to_bool, (data, col_name, threshold))

    def count_all_words(self, data):
        """ Adds a feature which is the number of words in the tweet

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the new feature appended
        """
        pattern = re.compile(r'\w+')
        return self._parallelize(self._base.count_all_words, (data, pattern))


    def count_emoticons(self, data):
        """ Adds a feature with the total number of emoticons in the tweet

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the new feature appended
        """
        return self._parallelize(self._base.count_emoticons, (data,))
    
    
    def dico_emotes(self,df):
        """
        This function create a dictionnary of the different emoticon present in the data. For each emoticon, a value is 
        assigned in function of polarisation of the emoticon.
        The list of the different emoticons is : 
            '<3',':)',';)',':`)','.)',':d',':p',':`d',';d',';p',';`d', 'xx', 'x',':(','</3',':/',':`)',':*',':o',':-'

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        dict - the dictionnary with the polarilized value for each emoticons
        """
    
        emotes = ['<3',':)',';)',':`)','.)',':d',':p',':`d',';d',';p',';`d', 'xx', 'x',':(','</3',':/',':`)',':*',':o',':-']
        emotes_enum = {}
        emotes_pos = {}
        for el in emotes:
            emotes_enum[el] = 0
            emotes_pos[el] = 0
        for l in range(df.shape[0]):
            t = df.tweets[l] 
            words = t.split()
            for w in words :
                if w in emotes:
                    emotes_enum[w] += 1
                    emotes_pos[w] += 2*df.type[l] - 1
            for i in range(len(words)) :
                if words[i] == ":" and len(words) > i+1:
                    if words[i+1] == "*":
                        emotes_enum[':*'] += 1
                        emotes_pos[':*'] += 2*df.type[l] - 1
                    elif words[i+1] == "-":
                        emotes_enum[':-'] += 1
                        emotes_pos[':-'] += 2*df.type[l] - 1
                    elif words[i+1] == "o":
                        emotes_enum[':o'] += 1
                        emotes_pos[':o'] += 2*df.type[l] - 1
        print("create dictionnary")
        emotes_mean = {}
        for i in range(len(emotes)):
            if emotes_enum[emotes[i]] != 0:
                emotes_mean[emotes[i]] = emotes_pos[emotes[i]]/emotes_enum[emotes[i]]
            else:
                emotes_mean[emotes[i]] = 0
        return emotes_mean
    

    def extract_emoji(self, data,dico_emotes):
        """ Adds a feature which is the sum of positive and negative emoticons.

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the new feature appended

        """
        #Don't need to put emote with bracket because we count them after
        listPositif = ['<3',':)',';)',':`)','.)',':d',':p',':`d',';d',';p',';`d', 'xx', 'x'] 
        listNegatif = [':(','</3',':/',':`)']
        return self._parallelize(self._base.extract_emoji, (data, dico_emotes))
    
    
    def dico_hashtag(self, data): 
        """ Create a dictionnary with the polarilize value for each hashtags

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        dict - the dictionnary with the polarilized value for each hashtags
        """
        
        hashtag = {'#'}
        for tweet in data.tweets:
            for i in tweet.split():
                if i.startswith("#"):
                    hashtag.add(i)
        dico_hash = {'#' : [0,0]}
        for hash in hashtag: 
            dico_hash[hash] = [0,0]
        for l in range(0,data.shape[0]):
            for i in data.iloc[l,0].split():
                if i.startswith("#"):
                    if data.iloc[l,1] == 1:
                        dico_hash[i][1] += 1
                    else :
                        dico_hash[i][0] -= 1
        print("finish to contablilize hashtag")
        return dico_hash

    def count_hashtag(self, data,dico_hashtag):
        """ Adds a feature which is the sum of positive and negative hashtags.

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the new feature appended

        """

        return self._parallelize(self._base.count_hashtag, (data, dico_hashtag))

    def tweet_polarity(self, data):
        """ Returns the net polarity of the tweet. Based on nltk's sentiment intensity analyzer scores

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the new feature appended

        """
        sia = SentimentIntensityAnalyzer()
        return self._parallelize(self._base.tweet_polarity, (data, sia))

    def tokenize(self, data):
        """ Splits the tweets into a list of words in the tweet

        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with a new column for the list of tokens

        """
        return self._parallelize(self._base.tokenize, (data,))

    def blob_sentiment(self, data):
        """Computes TextBlob sentiment and subjectivity for each tweet 
        
        Parameters
        ----------
        data : pd.DataFrame - the data containing the tweets

        Returns
        -------
        pd.DataFrame - the data with the sentiment and subjectivity
        
        """
        fts = self._parallelize(self._base.blob_sentiment, (data,))
        fts.columns = ['tb_sent', 'tb_subj']
        return fts