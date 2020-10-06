import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances 
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.spatial import distance 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances 
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.spatial import distance 
import pandas as pd 
import numpy as np 
# from tfidf_helpers import arr_convert_1d, cosine, manhatten_distance, euclidean_function

'''heavily inspired by https://www.geeksforgeeks.org/sklearn-feature-extraction-with-tf-idf/'''

txt_adopted = '../app/data/adopted_csv.csv'
scraped_data_adopted = open('../app/data/adopted_corpus.txt', 'r').read()
txt_adoptable = '../app/data/adoptable_csv.csv'
scraped_data_adoptable = open('../app/data/adoptable_corpus.txt', 'r').read()

## Converting 3D array of array into 1D array 
def arr_convert_1d(arr):
    arr = np.array(arr)
    arr = np.concatenate( arr, axis=0 )
    arr = np.concatenate( arr, axis=0 )
    print(arr)
    return arr 

def arr_convert_1d_adopted(arr):
    arr = np.array(arr)
    arr = np.concatenate( arr, axis=0 )
    arr = np.concatenate( arr, axis=0 )
    print(arr)
    return arr 

## Cosine Similarity 
cos = [] 
def cosine(trans):
    cos.append(cosine_similarity(trans[0], trans[1]))

# Manhatten Distance 
manhatten = [] 
def manhatten_distance(trans): 
	manhatten.append(pairwise_distances(trans[0], trans[1], 
										metric = 'manhattan')) 

## Euclidean Distance 
euclidean = [] 
def euclidean_function(vectors): 
	euc=euclidean_distances(vectors[0], vectors[1]) 
	euclidean.append(euc) 


def convert_adoptable(): 
    dataf_adoptable = pd.DataFrame() 
    lis2_adoptable = arr_convert_1d(manhatten) 
    dataf_adoptable['manhatten'] = lis2_adoptable 
    lis2_adoptable = arr_convert_1d(cos) 
    dataf_adoptable['cos_sim'] = lis2_adoptable 
    lis2_adoptable = (arr_convert_1d(euclidean) ) - 1
    dataf_adoptable['euclidean'] = lis2_adoptable 
    return dataf_adoptable['cos_sim'][::-1][1]

def convert_adopted(): 
    dataf_adopted = pd.DataFrame() 
    lis2_adopted = arr_convert_1d_adopted(manhatten) 
    dataf_adopted['manhatten'] = lis2_adopted 
    lis2_adopted = arr_convert_1d_adopted(cos) 
    dataf_adopted['cos_sim'] = lis2_adopted 
    lis2_adopted = (arr_convert_1d_adopted(euclidean) ) - 1
    dataf_adopted['euclidean'] = lis2_adopted 
    # print(dataf_adopted['cos_sim'][::-1][0])
    return dataf_adopted['cos_sim'][::-1][0]


class TextClassifier(object):
    def __init__(self):
        with open('pickled_algos/pickled_nb.pickle', 'rb') as f:
            self.model = pickle.load(f)
        with open('pickled_algos/tfidf_transformer.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)
        with open('pickled_algos/count_vect.pickle', 'rb') as f:
            self.cv = pickle.load(f)  

    def predict_one(self, data):
        cv_transformed = self.cv.transform(data) #counts how many words
        tfidf_transformed = self.tfidf.transform(cv_transformed)  #tf == cv . 
        string_predicted = self.model.predict(tfidf_transformed) 
        length = str(len((str(data))))
        if length == '4':
            error = 'Error.  You did not input a description.  Please try agian.'
            return error
        res_mnb = str(string_predicted[0])
        if res_mnb == '0':
            res_mnb = ('Less Likely to be Adopted')
        else:
            res_mnb = ("More Likely Than Not to be Adopted")
        return res_mnb 
    
    def tfidf_adopted(self, data): 
        txt=txt_adopted
        scraped_data=scraped_data_adopted
        df1 = pd.read_csv(str(txt))
        df = df1.fillna("None")  #impute empty records
        # df.drop(['index', 'status_adopted'], axis = 1, inplace= True)
        df_str = df[["description"]].copy()
        document = ' '.join(df_str['description'].tolist())
        document =[] 
        # Iterate over each row 
        for index, rows in df_str.iterrows(): 
            document.append(rows.description) 
        vect = TfidfVectorizer() 
        vect.fit(document) 
        corpus = [scraped_data, (str(data))] #the scraped data compared to the input string
        trans = vect.transform(corpus) 
        euclidean_function(trans) 
        cosine(trans) 
        manhatten_distance(trans) 
        return convert_adopted() 

    def tfidf_adoptable(self, data): 
        txt=txt_adoptable
        scraped_data=scraped_data_adoptable
        df1 = pd.read_csv(str(txt))
        df = df1.fillna("None")  #impute empty records
        # df.drop(['index', 'status_adopted'], axis = 1, inplace= True)
        df_str = df[["description"]].copy()
        document = ' '.join(df_str['description'].tolist())
        document =[] 
        # Iterate over each row 
        for index, rows in df_str.iterrows(): 
            document.append(rows.description) 
        vect = TfidfVectorizer() 
        vect.fit(document) 
        corpus = [scraped_data, (str(data))] #the scraped data compared to the input string
        trans = vect.transform(corpus) 
        euclidean_function(trans) 
        cosine(trans) 
        manhatten_distance(trans) 
        return convert_adoptable() 


    def recommend(tfidf_adoptable, tfidf_adopted):
        print(tfidf_adoptable)
        a = tfidf_adoptable
        print("recommed passed: ", a)

class SentimAnalysis(object):
    def __init__(self):
        with open('pickled_algos/pickled_nb_sentiment140.pickle', 'rb') as f:
            self.model = pickle.load(f)
        with open('pickled_algos/tfidf_transformer_sentiment140.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)
        with open('pickled_algos/count_vect_sentiment140.pickle', 'rb') as f:
            self.cv = pickle.load(f)
            
    def sentiment_(self, data): 
        cv_transformed = self.cv.transform(data) #counts how many words
        tfidf_transformed = self.tfidf.transform(cv_transformed)  #tf == cv . 
        string_predicted = self.model.predict(tfidf_transformed) 
        length = str(len((str(data))))
        if length == '4':
            error = 'Error.  You did not input a description.  Please try agian.'
            return error
        res_sent = str(string_predicted[0])
        if res_sent == '0':
            res_sent = ('Negative Sentiment')
        else:
            res_sent = ("Positive Sentiment")
        return res_sent
    
if __name__ == '__main__':
    pass

    ### BEGIN test scripts 
    
    # # instantiate object
    my_classifier = TextClassifier()
    my_sentim = SentimAnalysis()
    # #test negative sentiment
    test_string_pred = ['this girl is a foster pit and has none of her teeth']
    # test_string_pred = ['bad doggo']
    
    # # #test positive sentimnet
    # test_string_pred = ['fill out an online form to find this male puppy a forever home']
    
    # # #test empty string
    # # test_string_pred = ['bad']

    # # res_mnb = my_classifier.predict_one(test_string_pred)
    # # print("Your MNB description yields: ", res_mnb)


    res_tfidf_adopted = my_classifier.tfidf_adopted(str(test_string_pred)); 
    # print(res_tfidf_adopted)
    
    res_tfidf_adoptable = my_classifier.tfidf_adoptable(str(test_string_pred)); 
    # print(res_tfidf_adoptable); 
    # # my_classifier.recommend(my_classifier.tfidf_adopted) #(my_classifier.tfidf_adoptable, my_classifier.tfidf_adopted)
    # # test_string_pred = ['bad, bad dog']
    # res_sentiment = my_sentim.sentiment_((test_string_pred)); 
    # print("Your input invokes the following sentiment: ", res_sentiment); 
    
    
    #### END test scripts