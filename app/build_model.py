'''heavily borrowed from https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk and Galvanize DSI'''

import argparse
import pickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# from sklearn.externals import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
import string, re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier


from nltk.tag import pos_tag

import sys
sys.path.append('../src/pipelines/data_ingestion/')
from read_in_data import *

import numpy as np
from io import StringIO
import os
import string
import math
from string import digits 

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import random

#########################################################


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


#######################################



class TextClassifier(object):

    def __init__(self, model):
        self._vectorizer = TfidfVectorizer()
        self._classifier = MultinomialNB()
        self.data = data
        self.model = MultinomialNB()


    
    
        # model = MultinomialNB()
        # model.fit(test_data, train_data)
        # joblib.dump(model, fitted_data.joblib)
    

    
    def clean_description(self, data):
        data = str(data)
        # print("string data: ", data)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in data:  
            if ele in punc:  
                data = data.replace(ele, "")
        # print("cleaned data: ", data)
        return data
    
    
    def tokenize_data(self, data):
        data = str(data)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in data:  
            if ele in punc:  
                data = data.replace(ele, "")
        data = data.split()
        # print("tokenized data: ", data)
        return data


    def lemmatize_data(self, data):
        data = str(data)
        

    def sentim(self, data):
        stop_words = ['the', 'an', 'the', 'i', 'a', 'and', 'to'] #, 'none'] #, 'heartworm', ' distemper/parvo'] #stopwords.words('english')

        path_csv = '../data/csv/tf_idf_adoptable_csv.csv'
        df = read_df_csv(path_csv)
        X_negative = df["description"] #data
        corpus_dirty = []
        for doc in range(len(X_negative)):
            str_corpus = str(X_negative[doc])
            corpus_dirty.append(str_corpus)

        negative_documents = []
        for doc in range(len(X_negative)):
            record = X_negative[doc]
            record = (record.lower())
            replaced = record.replace(", '...'", "").replace("...", '').replace('\d+', '') 
            remove_digits = str.maketrans('', '', digits) 
            replaced = replaced.translate(remove_digits) 
            clean = replaced.replace(", '...'", "").replace("...", '')
            negative_documents.append(clean)
        # print(documents)
    # #     # 2. Create a set of tokenized documents.
        negative_descriptions = [word_tokenize(content) for content in negative_documents]

        negative_cleaned_tokens_list = []
        for tokens in negative_descriptions:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        all_neg_words = get_all_words(negative_cleaned_tokens_list)
        
        
        freq_dist_neg = FreqDist(all_neg_words)
        print("most common ADOPTABLE words: ", freq_dist_neg.most_common(10))

        ##################################################################
        ##################################################################
        ##################################################################

        
        path_csv = '../data/csv/tf_idf_adopted_csv.csv'
        df = read_df_csv(path_csv)
        X_positive = df["description"] #data
        corpus_dirty = []
        for doc in range(len(X_positive)):
            str_corpus = str(X_positive[doc])
            corpus_dirty.append(str_corpus)

        positive_documents = []
        for doc in range(len(X_positive)):
            record = X_positive[doc]
            record = (record.lower())
            replaced = record.replace(", '...'", "").replace("...", '').replace('\d+', '') 
            remove_digits = str.maketrans('', '', digits) 
            replaced = replaced.translate(remove_digits) 
            clean = replaced.replace(", '...'", "").replace("...", '')
            positive_documents.append(clean)
        # print(documents)
    # #     # 2. Create a set of tokenized documents.
        positive_descriptions = [word_tokenize(content) for content in positive_documents]
        # print("\n\nPositive Descriptions Tokenized: ", positive_descriptions)
        # ['dora', 'female', 'shep', 'mix', 'brindle', 'dhpp', 'kc', '//', 'no', 'puppy', 'hi', 'cathleen', ',', 'she', 'is', 'doing', 'great', 'and', 'really', 'starting'], ['meet', 'nova', '!', 'now', 'that', 'she', 'is', 'done', 'raising', 'her', 'pups', 'she', 'is', 'looking', 'for', 'a', 'home', 'of', 'her', 'own', 'where']]
        
        positive_cleaned_tokens_list = []
        for tokens in positive_descriptions:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


        
        
        all_pos_words = get_all_words(positive_cleaned_tokens_list)
        
        # save_documents = open("pickled_algos/all_pos_words.pickle","wb")
        # pickle.dump(positive_cleaned_tokens_list, save_documents)
        # save_documents.close()
        

        freq_dist_pos = FreqDist(all_pos_words)
        print("most common ADOPTED words: ", freq_dist_pos.most_common(10))

        ##################################################################
        ##################################################################
        ##################################################################
        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        # positive_tokens_for_model = all_pos_words.pickle
        
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
        

        
        
        
        positive_dataset = [(description_dict, "Positive")
                        for description_dict in positive_tokens_for_model]

        negative_dataset = [(description_dict, "Negative")
                            for description_dict in negative_tokens_for_model]
        
        # print("positive_dataset: ", positive_dataset)
        # print("negative_dataset: ", negative_dataset)


        dataset = positive_dataset + negative_dataset
        seventy_percent_of_data = int(len(dataset) * .7)
        thirty_percent_of_data = int(len(dataset) * .3)
        # print(thirty_percent_of_data) #361

        random.shuffle(dataset) #to avoid bias

        train_data = dataset[:seventy_percent_of_data]
        test_data = dataset[thirty_percent_of_data:]

        classifier = NaiveBayesClassifier.train(train_data)
        # classifier = MultinomialNB.fit(train_data)
        save_classifier = open("naivebayes_pet.pickle","wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

        print("%%%%%%%%%%%%%%%%%%%Accuracy is:", classify.accuracy(classifier, test_data))

        print(classifier.show_most_informative_features(10))
        
        # from nltk.corpus import twitter_samples
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(twitter_samples)
        data = str(data)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in data:  
            if ele in punc:  
                data = data.replace(ele, "")
        data = data.split()
        # print("tokenized data: ", data)
        
        #breakdown parts of speech
        parts_of_speech = [] 
        parts_of_speech.append(nltk.pos_tag(data))
        print("parts of speech tagging: ", parts_of_speech) 
        #lemmatized data:
        stop_words = [] #left here in case I want to add words in the future
        cleaned_tokens = []


        for token, tag in nltk.pos_tag(data):
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos) 



            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        
        custom_tokens = remove_noise(word_tokenize(str(data)))

        print(str(data), classifier.classify(dict([token, True] for token in custom_tokens)))

        sentiment_result = [classifier.classify(dict([token, True] for token in custom_tokens))]

        print("sentiment_result: ", type(sentiment_result), sentiment_result)

        data = sentiment_result
        return data



    def sentim_twitter(self, data):
        '''heavily borrowed from https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
        to show functioning model'''
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')
        text = twitter_samples.strings('tweets.20150430-223406.json')
        tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

        stop_words = stopwords.words('english')

        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
        
        

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        all_pos_words = get_all_words(positive_cleaned_tokens_list)

        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(10))

        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        train_data = dataset[:700]
        test_data = dataset[700:]

        classifier = NaiveBayesClassifier.train(train_data)
        print("twitter data **********************************")

        print("%%%%%%%%%%%%%%%%%%% Twitter Accuracy is:", classify.accuracy(classifier, test_data))
        print("twitter data **********************************")

        print(classifier.show_most_informative_features(10))

        # data = (data)

        # custom_tweet = str(data) 
        print("twitter data **********************************")
        print("twitter data **********************************")
        print("is this reading data correctly???: ", type(str(data)))
        custom_tweet = str(data)
        # this gives negative
        
        
        
        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        print("twitter data **********************************")
        print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
        twitter =  classifier.classify(dict([token, True] for token in custom_tokens))
        return twitter


if __name__ == '__main__':



