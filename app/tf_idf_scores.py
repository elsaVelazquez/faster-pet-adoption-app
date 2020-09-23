import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv 
from nltk.corpus import stopwords
stopwords.words('english')


#ADOPTABLE
df_adoptable = pd.read_csv('../app/data/adoptable_csv.csv')
df = df_adoptable.fillna("None")  #impute empty records
# df.drop(['index', 'status_adoptable'], axis = 1, inplace= True)
df_str_adoptable = df[["description"]].copy()
document_adoptable = ' '.join(df_str_adoptable['description'].tolist())
# print(str_adoptable)

df_adopted = pd.read_csv('../app/data/adopted_csv.csv')
df = df_adopted.fillna("None")  #impute empty records
# df.drop(['index', 'status_adopted'], axis = 1, inplace= True)
df_str_adopted = df[["description"]].copy()
document_adopted = ' '.join(df_str_adopted['description'].tolist())
# print(document_adopted)


'''heavily borrowed from https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76'''

#CREATE BAG OF WORDS
bag_of_words_adoptable = document_adoptable.split(' ')
bag_of_words_adopted = document_adopted.split(' ')


#UNIQUE WORDS
unique_words = set(bag_of_words_adoptable).union(set(bag_of_words_adopted))
# print(unique_words)



#a dictionary of words and their occurence for each document in the corpus 
#this still has lots of noise
num_words_adoptable = dict.fromkeys(unique_words, 0)
for word in bag_of_words_adoptable:
    num_words_adoptable[word] += 1
    print("ADOPTABLE ", word + ":", num_words_adoptable[word])
num_words_adopted = dict.fromkeys(unique_words, 0)
for word in bag_of_words_adopted:
    num_words_adopted[word] += 1
    print("ADOPTED ", word + ":", num_words_adopted[word])
    
    

# Term Frequency (TF)
# The number of times a word appears in a document divded by the total number of words in the document.
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


tf_adopted = computeTF(num_words_adopted, bag_of_words_adopted)
tf_adoptable = computeTF(num_words_adoptable, bag_of_words_adoptable)

idfs = computeIDF([num_words_adopted, num_words_adoptable])


tfidfA = computeTFIDF(tf_adopted, idfs)
tfidfB = computeTFIDF(tf_adoptable, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
df_sort = df[::-1]
df_top = df_sort[:-4]
# print(df.head(5))
# print(df_top.head())

print([tfidfA])

# d = {}
# sorted_desc = (sorted(d.items(), key = lambda kv:(kv[1], kv[0])))  
# desc = (sorted_desc[::-1]) #sort
# print("TOP 10 ADOPTABLE TOKENS: ", desc[:10]) #top 10 only