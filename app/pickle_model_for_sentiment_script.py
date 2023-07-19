# %%
''' heavily borrowed from https://github.com/GalvanizeDataScience/lectures/blob/Denver/text-classification/frank-burkholder/naive_bayes_sklearn.ipynb'''
''' data is from http://help.sentiment140.com/for-students .  The Sentiment140 is used for brand management, polling, and planning a purchase. Sentiment140 is used to discover the sentiment of a brand or product or even a topic on the social media platform Twitter. Rather than working on keywords-based approach, which leverages high precision for lower recall, Sentiment140 works with classifiers built from machine learning algorithms. The Sentiment140 uses classification results for individual tweets along with the traditional surface that aggregated metrics. '''


# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# %%
categories = [0, 1]

# %%

df = pd.read_csv('../data/sentiment140.csv',header = None)

# drop emtpy records
df = df.dropna()

df.drop([1,2,3,4], axis=1, inplace=True)
# df.head()


X = df[5] #data
X.to_numpy()
# print(X)

y= df[0].replace(4, 1) #turn the 4 into a 1 to show positive sentiment
y.to_numpy()
print(y)
                                 

# %%
print(len(X)) #11541
print(len(y)) #11541

# %%
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
new_stopwords = ['a']
new_stopwords_list = list(stop_words.union(new_stopwords))

count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words = new_stopwords_list,  analyzer='word', max_df=1.0, min_df=1,  max_features=None) 

count_vect.fit(X)

target_names = y

# %%
print(count_vect)

# %%

X_counts = count_vect.transform(X)
print("The type of X_counts is {0}.".format(type(X_counts)))
print("The X matrix has {0} rows (documents) and {1} columns (words).".format(
        X_counts.shape[0], X_counts.shape[1]))

# %%
tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_transformer.fit(X_counts)
X_tfidf = tfidf_transformer.transform(X_counts)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y.values, test_size=0.2, random_state=42)

# %%
print(X_train.shape)
print(y_train.shape)

# %%
print(X_test.shape)
print(y_test.shape)

# %%
nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb_model.fit(X_train, y_train);


# %%
# feature_words = count_vect.get_feature_names()
# n = 20 #number of top words associated with the category that we wish to see
# target_names[categories] = 1
# for cat in range(len(categories)): #categories == ['negative', 'positive']
#     # print(f"\nTarget: {cat}")
#     # print(f"\nname: ", target_names[cat])
#     print(f"\nTarget: {cat}, name:", target_names[cat])
#     log_prob = nb_model.feature_log_prob_[cat]
#     i_topn = np.argsort(log_prob)[::-1][:n]
#     features_topn = [feature_words[i] for i in i_topn]
#     print(f"Top {n} tokens: ", features_topn)


# %%
y_predicted = nb_model.predict(X_test)
# X_test[0]

# %%
# y_predicted

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# %%
y_pred_proba = nb_model.predict_proba(X_test)[:,1]
y_pred_proba

# %%
thresh=0.7
y_pred = (y_pred_proba>=thresh).astype(int)
# y_pred

# %%
y_pred

# %%


# %%
# pickle and save the model 
import pickle
save_documents = open("../pickled_algos/pickled_nb_sentiment140.pickle","wb")
pickle.dump(nb_model, save_documents)
save_documents.close()

tfidf = tfidf_transformer
save_documents = open("../pickled_algos/tfidf_transformer_sentiment140.pickle", "wb")
pickle.dump(tfidf, save_documents)
save_documents.close()

count_vect = count_vect
save_documents = open("../pickled_algos/count_vect_sentiment140.pickle", "wb")
pickle.dump(count_vect, save_documents)
save_documents.close()



# %%
def predict_one_twitter(input):
    with open('../pickled_algos/pickled_nb_sentiment140.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('../pickled_algos/tfidf_transformer_sentiment140.pickle', 'rb') as f:
        tfidf = pickle.load(f)

    with open('../pickled_algos/count_vect_sentiment140.pickle', 'rb') as f:
        cv = pickle.load(f)

  
    cv_transformed = cv.transform(string_pred) #counts how many words
    tfidf_transformed = tfidf.transform(cv_transformed)  #tf == cv . 
    string_predicted = model.predict(tfidf_transformed) 
    res = str(string_predicted[0])
    if res == '0':
        res = ('Negative Sentiment')
    else:
        res = ("Positive Sentiment")
    return res
# test Positive Sentiment
# string_pred = ['this girl is a foster pit and has none of her teeth']
# test Negative Sentiment:
string_pred = ['fill out an online form to find this male puppy a forever home']
# string_pred = ['this dog is great i love him']
res = predict_one_twitter(string_pred)
print(res)

# %%


# %%



