import pandas as pd
import numpy as np
import string 
import csv 
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords




def make_df(path_csv):
    df1 = pd.read_csv(path_csv)
    df = df1.fillna("None")  #impute empty records
    # df.drop(['index', 'status_adoptable'], axis = 1, inplace= True)
    df_content = df[["description"]].copy()
    return df_content
    # txt_file = '../../../data/txt/adoptable_txt.txt'
 

 
    
class CleanTextDescriptions(object):
    def __init__(self, df):
        self.df = df

    def remove_stray(self, input_text):
        sub = re.sub(r'@\w+', '', input_text)
        return sub

    def remove_urls(self, input_text):
        http = re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
        return http 

    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        replace = input_text.replace('_','')
        return replace

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        translated = input_text.translate(trantab)
        return translated 
        
    def remove_digits(self, input_text):
        no_nums = re.sub('\d+', '', input_text)
        return no_nums

    def to_lower(self, input_text):
        lower = input_text.lower()
        return lower 

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        stopw = " ".join(clean_words) 
        return stopw 

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        stem =  " ".join(stemmed_words)   
        return stem 

if __name__ == "__main__":
    ##############################################
    ############# . BEGIN ADOPTABLE . ############# 
    path_csv = '../app/data/adoptable_csv.csv'
    df_adoptable = make_df(path_csv)
    giant_string = ' '.join(df_adoptable['description'].tolist())
    # print(giant_string)

    adoptable = CleanTextDescriptions(giant_string)
    stray = adoptable.remove_stray(giant_string)
    url = adoptable.remove_urls(stray)
    emoji = adoptable.emoji_oneword(url)
    punct = adoptable.remove_punctuation(emoji)
    dig = adoptable.remove_digits(punct)
    low = adoptable.to_lower(dig)
    stopw = adoptable.remove_stopwords(low)
    stem = adoptable.stemming(stopw)

    token_lst = stem.split(' ')
    tokens = np.array(token_lst)
    denom = (len(tokens))
    print("total tokens in adoptable: ", denom)
    d = {} #make a dictionary
    # Loop through each word of the string 
    for word in tokens: 
        if word in d: # Check if the word is already in dictionary 
            d[word] = d[word] + 1 # Increment count of word by 1 
        else: 
            d[word] = 1  # Initialize the word to dictionary with count 1 

    sorted_desc = (sorted(d.items(), key = lambda kv:(kv[1], kv[0])))  
    desc = (sorted_desc[::-1]) #sort
    print("TOP 10 ADOPTABLE TOKENS: ", desc[:10]) #top 10 only
    ############# . END ADOPTABLE . ############# 
    ##############################################
    
    
    
    ##############################################
    ############# . BEGIN ADOPTED . ############# 
    path_csv = '../app/data/adopted_csv.csv'
    df_adopted = make_df(path_csv)
    giant_string = ' '.join(df_adopted['description'].tolist())
    # print(giant_string)

    adopted = CleanTextDescriptions(giant_string)
    stray = adopted.remove_stray(giant_string)
    url = adopted.remove_urls(stray)
    emoji = adopted.emoji_oneword(url)
    punct = adopted.remove_punctuation(emoji)
    dig = adopted.remove_digits(punct)
    low = adopted.to_lower(dig)
    stopw = adopted.remove_stopwords(low)
    stem = adopted.stemming(stopw)

    token_lst = stem.split(' ')
    tokens = np.array(token_lst)
    
    denom = (len(tokens))
    print("total tokens in adoptable: ", denom)
    d = {} #make a dictionary
    # Loop through each word of the string 
    for word in tokens: 
        if word in d: # Check if the word is already in dictionary 
            d[word] = d[word] + 1 # Increment count of word by 1 
        else: 
            d[word] = 1  # Initialize the word to dictionary with count 1 

    sorted_desc = (sorted(d.items(), key = lambda kv:(kv[1], kv[0])))  
    desc = (sorted_desc[::-1]) #sort
    print("TOP 10 ADOPTED TOKENS: ", desc[:10]) #top 10 only
    ############# . END ADOPTED . ############# 
    ##############################################