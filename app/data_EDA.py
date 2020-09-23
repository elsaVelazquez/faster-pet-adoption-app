import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
# import emoji
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)

import pickle 

#randomize data
df = pd.read_csv('data/Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
df = df[['description', 'sentiment']]

df_dogs = pd.read_csv('../data/csv/sentiment_status_csv.csv')
df_dogs = df_dogs.reindex(np.random.permutation(df_dogs.index))
df_dogs = df_dogs[['description', 'sentiment']]

# EDA
# TODO print this!!
sns.factorplot(x="sentiment", data=df, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")#.set_title("Tweet Descriptions Sentiment Data")
plt.show();
plt.close()

sns.factorplot(x="sentiment", data=df_dogs, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")#.set_title("Dog Descriptions Sentiment Data")
plt.show();
plt.close()


# compute basic statistics on the description variable   
class TextDescriptionCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        #used only on training data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x))        
        df = pd.DataFrame({'count_words': count_words })        
        return df  

class CleanTextDescriptions(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X
    
    def grid_vect(clf, parameters_clf, X_train, X_test, y_train, y_test, dataset, parameters_text, vect, max_iter=4000, is_w2v=False): #=None, vect=None, is_w2v=False):
        '''Based on http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
        Heavily borrowed from https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27'''
        print(f"*~*~*~*~*~*~*~*~*\nBegin dataset: {dataset} using \nmodel: {clf}  \nparameters: {parameters_clf}\n")
        textcountscols = ['count_words']
        
        if is_w2v:
            w2vcols = []
            for i in range(SIZE):
                w2vcols.append(i)
            features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                    , ('w2v', ColumnExtractor(cols=w2vcols))]
                                    , n_jobs=-1)
        else:
            features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                    , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                    , n_jobs=-1)
        
        pipeline = Pipeline([
            ('features', features)
            , ('clf', clf)
        ])
        
        # Join the parameters dictionaries together
        parameters = dict()
        if parameters_text:
            parameters.update(parameters_text)
        parameters.update(parameters_clf)
        # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
        
        print(f"Performing grid search on {dataset}...")
        print(f"{dataset} pipeline:", [name for name, _ in pipeline.steps])
        print(f" {dataset} parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(X_train, y_train)
        print(f"{dataset} done in %0.3fs" % (time() - t0))
        print()
        print(f"{dataset} Best CV score: %0.3f" % grid_search.best_score_)
        print(f"{dataset} Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f" {dataset} \t%s: %r" % (param_name, best_parameters[param_name]))
            
        print(f"{dataset} Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
        print("\n")
        print(f"{dataset} Classification Report Test Data")
        print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
        print(f"*~*~*~*~*~*~*~*~*\nEND dataset: {dataset} using \nmodel: {clf}  \nparameters: {parameters_clf}*~*~*~*~*~*~*~*~*\n")
                            
        return grid_search

    
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X, **transform_params):
        return X[self.cols]
    def fit(self, X, y=None, **fit_params):
        return self

  
def show_dist(df, col, plot_title):
    print('Descriptive stats for {}'.format(col))
    print('-'*(len(col)+22))
    print(df.groupby('sentiment')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='sentiment', size=5, hue='sentiment', palette="PuBuGn_d") #.set_title(plot_title)
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    # TODO print this!!
    plt.show();
 
def display_word_freq(giant_string, plot_title):
    bow = cv.fit_transform(giant_string)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(10), columns = ['word', 'freq'])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax).set_title(plot_title)
    # TODO print this!!
    plt.show();



def grid_vect(clf, parameters_clf, X_train, X_test, y_train, y_test, dataset, parameters_text, vect, max_iter=4000, is_w2v=False): #=None, vect=None, is_w2v=False):
    '''Based on http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
       Heavily borrowed from https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27'''
    print(f"*~*~*~*~*~*~*~*~*\nBegin dataset: {dataset} using \nmodel: {clf}  \nparameters: {parameters_clf}\n")
    textcountscols = ['count_words']
    
    if is_w2v:
        w2vcols = []
        for i in range(SIZE):
            w2vcols.append(i)
        features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('w2v', ColumnExtractor(cols=w2vcols))]
                                , n_jobs=-1)
    else:
        features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                , n_jobs=-1)
    
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)
    ])
    
    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)
    # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print(f"Performing grid search on {dataset}...")
    print(f"{dataset} pipeline:", [name for name, _ in pipeline.steps])
    print(f" {dataset} parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print(f"{dataset} done in %0.3fs" % (time() - t0))
    print()
    print(f"{dataset} Best CV score: %0.3f" % grid_search.best_score_)
    print(f"{dataset} Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print(f" {dataset} \t%s: %r" % (param_name, best_parameters[param_name]))
        
    print(f"{dataset} Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print(f"{dataset} Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
    print(f"*~*~*~*~*~*~*~*~*\nEND dataset: {dataset} using \nmodel: {clf}  \nparameters: {parameters_clf}*~*~*~*~*~*~*~*~*\n")
                        
    return grid_search

   
if __name__ == "__main__":
    #EDA to show counts of words
    tc = TextDescriptionCounts()
    df_eda_tweets = tc.fit_transform(df.description)
    df_eda_tweets['sentiment'] = df.sentiment
    #TODO print this!!
    # show_dist(df_eda_tweets, 'count_words', 'Tweet Word Counts')
    
    tc_dogs = TextDescriptionCounts()
    df_eda_dogs = tc_dogs.fit_transform(df_dogs.description)
    df_eda_dogs['sentiment'] = df_dogs.sentiment
    #TODO print this!!
    # show_dist(df_eda_dogs, 'count_words', 'Dog Descriptions Word Counts')

    #clean data
    ct = CleanTextDescriptions()
    sr_clean = ct.fit_transform(df.description)
    # print(sr_clean.sample(5))
    
    ct_dogs = CleanTextDescriptions()
    sr_clean_dogs = ct_dogs.fit_transform(df_dogs.description)
    # print(sr_clean_dogs.sample(5))
    

    #impute into empty tweets rows
    #the dog data already was imputed with 'None'
    empty_clean = sr_clean == ''
    print('{} Tweets records have no words left after text cleaning and were imputed with the word "None"'.format(sr_clean[empty_clean].count()))
    sr_clean.loc[empty_clean] = '[None]'
    
    imputed_clean_dogs = (sr_clean_dogs == 'None')
    print('64 Dog records were imputed with the word "None"') #.format(sr_clean_dogs[imputed_clean_dogs].count()))


    #compare word freq by sentiment
    df_dogs_positive = df_dogs[df_dogs['sentiment'] == 'positive']
    sr_clean_dogs_positive = ct.fit_transform(df_dogs_positive.description)
    
    df_dogs_negative = df_dogs[df_dogs['sentiment'] == 'negative']
    sr_clean_dogs_negative = ct.fit_transform(df_dogs_negative.description)
    
    df_tweets_positive = df[df['sentiment'] == 'positive']
    sr_clean_tweets_positive = ct.fit_transform(df_tweets_positive.description)
    
    df_tweets_negative = df[df['sentiment'] == 'negative']
    sr_clean_tweets_negative = ct.fit_transform(df_tweets_negative.description)

    #Display word frequency
    cv = CountVectorizer()
    #TODO print this!!
    display_word_freq(sr_clean, 'All Tweeted Words Frequencies')
    display_word_freq(sr_clean_tweets_positive, 'All Positive Sentiment Tweeted Words Frequencies')
    display_word_freq(sr_clean_tweets_negative, 'All Negative Sentiment Tweeted Words Frequencies')
    display_word_freq(sr_clean_dogs, 'All Dog Descriptions Words Frequencies')
    display_word_freq(sr_clean_dogs_positive, 'All Positive Sentiment Dog Descriptions Word Frequencies')
    display_word_freq(sr_clean_dogs_negative, 'All Negative Sentiment Dog Descriptions Word Frequencies')


    #create test data
    df_model_tweets = df_eda_tweets
    df_model_tweets['clean_text'] = sr_clean
    df_model_tweets.columns.tolist()
    
    df_model_dogs = df_eda_dogs
    df_model_dogs['clean_text'] = sr_clean_dogs
    df_model_dogs.columns.tolist()
    
    X_train_tweets, X_test_tweets, y_train_tweets, y_test_tweets = train_test_split(df_model_tweets.drop('sentiment', axis=1), df_model_tweets.sentiment, test_size=0.1, random_state=37)
    X_train_dogs, X_test_dogs, y_train_dogs, y_test_dogs = train_test_split(df_model_dogs.drop('sentiment', axis=1), df_model_dogs.sentiment, test_size=0.1, random_state=37)
    
    
    #get metrics Precision and Recall
    # Parameter grid settings for the vectorizers (Count and TFIDF)
    parameters_vect = {
        'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
        'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
        'features__pipe__vect__min_df': (1,2)
    }
    # Parameter grid settings for MultinomialNB
    parameters_mnb = {
        'clf__alpha': (0.25, 0.5, 0.75)
    }
    # Parameter grid settings for LogisticRegression
    parameters_logreg = {
        'clf__C': (0.25, 0.5, 1.0),
        'clf__penalty': ('l1', 'l2')
    }
    print("METRICS REPORTS")
    # model == clf_RF
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    grid_vect(clf, parameters, X_train=X_train_tweets, X_test=X_test_tweets, y_train=y_train_tweets, y_test=y_test_tweets, dataset='TWEETS', parameters_text=None, vect=None, is_w2v=False)
    grid_vect(clf, parameters, X_train=X_train_dogs, X_test=X_test_dogs, y_train=y_train_dogs, y_test=y_test_dogs, dataset='DOGS', parameters_text=None, vect=None, is_w2v=False)
    
    
    
    ## BEGIN THESE HAVE ALREADY BEEN DONE
    mnb = MultinomialNB()
    logreg = LogisticRegression(max_iter=4000)
    
    countvect = CountVectorizer()
    # MultinomialNB
    best_mnb_countvect_tweets = CleanTextDescriptions.grid_vect(mnb, parameters_mnb, X_train=X_train_tweets, X_test=X_test_tweets, y_train=y_train_tweets, y_test=y_test_tweets, dataset='TWEETS',  parameters_text=parameters_vect, vect=countvect)
    joblib.dump(best_mnb_countvect_tweets, 'pickled_algos/tweets_best_mnb_countvect.pkl')
    
    mnb_dogs_pickled = mnb.fit(CleanTextDescriptions.X_train_dogs, CleanTextDescriptions.y_train_dogs, clf__alpha=0.5)
    pickle.dump(mnb_dogs_pickled, 'pickled_algos/mnb_dogs_pickled.pkl')
    
    
    # LogisticRegression
    best_logreg_countvect_tweets = CleanTextDescriptions.grid_vect(logreg, parameters_logreg, X_train=X_train_tweets, X_test=X_test_tweets, y_train=y_train_tweets, y_test=y_test_tweets, dataset='TWEETS', parameters_text=parameters_vect, vect=countvect)
    joblib.dump(best_logreg_countvect_tweets, 'pickled_algos/tweets_best_logreg_countvect.pkl')
    
    countvect = CountVectorizer()
    # MultinomialNB
    best_mnb_countvect = CleanTextDescriptions.grid_vect(mnb, parameters_mnb, X_train=X_train_dogs, X_test=X_test_dogs, y_train=y_train_dogs, y_test=y_test_dogs, dataset='DOGS',  parameters_text=parameters_vect, vect=countvect)
    joblib.dump(best_mnb_countvect, 'pickled_algos/dogs_best_mnb_countvect.pkl')
    # LogisticRegression
    best_logreg_countvect = CleanTextDescriptions.grid_vect(logreg, parameters_logreg, X_train=X_train_dogs, X_test=X_test_dogs, y_train=y_train_dogs, y_test=y_test_dogs, dataset='DOGS', parameters_text=parameters_vect, vect=countvect)
    joblib.dump(best_logreg_countvect, 'pickled_algos/dogs_best_logreg_countvect.pkl')
    ## END THESE HAVE ALREADY BEEN DONE
    
    
    #shows unpickling works
filename = 'dogs_best_mnb_countvect.pkl'
model = joblib.load(os.path.join('pickled_algos/', filename))


print('\nPredicting!')
string = 'this sentence to show positive'
y_pred_dogs = model.predict(string)

print(f'\nPredicted classes: \n{y_pred_dogs}')


    # #PICKLE FOR DOGS USING MNB
    # mnb_dogs_pickled = mnb.fit(CleanTextDescriptions.X_train_dogs, CleanTextDescriptions.y_train_dogs, clf__alpha=0.5)
    # pickle.dump(mnb_dogs_pickled, 'pickled_algos/mnb_dogs_pickled.pkl')
    
    # best_mnb_countvect = CleanTextDescriptions.grid_vect(mnb, parameters_mnb, X_train=X_train_dogs, X_test=X_test_dogs, y_train=y_train_dogs, y_test=y_test_dogs, dataset='DOGS',  parameters_text=parameters_vect, vect=countvect)
    # joblib.dump(best_mnb_countvect, 'pickled_algos/dogs_countvect.pkl')
    
    
    
    
    
    # #PICKLE FOR TWEETS USING LOGREG
    # logreg_tweets_pickled = logreg.fit(CleanTextDescriptions.X_train_tweets, CleanTextDescriptions.y_train_tweets, clf__alpha=0.5)
    # pickle.dump(logreg_tweets_pickled, 'pickled_algos/logreg_tweets_pickled')
    
    # best_logreg_countvect = CleanTextDescriptions.grid_vect(mnb, parameters_mnb, X_train=X_train_tweets, X_test=X_test_tweets, y_train=y_train_tweets, y_test=y_test_tweets, dataset='TWEETS',  parameters_text=parameters_vect, vect=countvect)
    # joblib.dump(best_logreg_countvect, 'pickled_algos/tweets_countvect.pkl')
    
    