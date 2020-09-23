from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances 
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.spatial import distance 
import pandas as pd 
import numpy as np 
'''heavily inspired by https://www.geeksforgeeks.org/sklearn-feature-extraction-with-tf-idf/'''

## Converting 3D array of array into 1D array 
def arr_convert_1d(arr): 
	arr = np.array(arr) 
	arr = np.concatenate( arr, axis=0 ) 
	arr = np.concatenate( arr, axis=0 ) 
	return arr 

## Cosine Similarity 
cos = [] 
def cosine(trans): 
	cos.append(cosine_similarity(trans[0], trans[1])) 

## Manhatten Distance 
manhatten = [] 
def manhatten_distance(trans): 
	manhatten.append(pairwise_distances(trans[0], trans[1], 
										metric = 'manhattan')) 

## Euclidean Distance 
euclidean = [] 
def euclidean_function(vectors): 
	euc=euclidean_distances(vectors[0], vectors[1]) 
	euclidean.append(euc) 

# find the similarity between user input and the existing data using the above functions

## TF - IDF 
def tfidf(txt, scraped_data, user_input): 
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

	corpus = [scraped_data,user_input] #the scraped data compared to the input string
	trans = vect.transform(corpus) 
	print("TRANS!!!!!!!: ", trans)

	euclidean_function(trans) 
	cosine(trans) 
	manhatten_distance(trans) 
	return convert() 

def convert(): 
	dataf = pd.DataFrame() 
	lis2 = arr_convert_1d(manhatten) 
	dataf['manhatten'] = lis2 
	lis2 = arr_convert_1d(cos) 
	dataf['cos_sim'] = lis2 
	lis2 = (arr_convert_1d(euclidean) ) - 1
	if lis2 < 0.005:
		lis2 = 0
	dataf['euclidean'] = lis2 
	return dataf 


if __name__ == "__main__":
    
	txt_adopted = '../app/data/adopted_csv.csv'
	scraped_data_adopted = open('../app/data/adopted_corpus.txt', 'r').read()
	user_input = 'this girl is a foster pit and has none of her teeth'

	newData = tfidf(txt=txt_adopted, scraped_data=scraped_data_adopted, user_input=user_input); 
	print(newData); 
