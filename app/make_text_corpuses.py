''' insired by https://stackoverflow.com/questions/47339698/how-to-convert-csv-file-to-text-file-using-python'''
import pandas as pd
import csv 
from pandas import DataFrame


#ADOPTABLE
df_adoptable = pd.read_csv('../app/data/adoptable_csv.csv')
df = df_adoptable.fillna("None")  #impute empty records
df_str_adoptable = df[["description"]].copy()
document_adoptable = ' '.join(df_str_adoptable['description'].tolist())
str_adoptable = str(document_adoptable )
text_file = open("../app/data/adoptable_corpus.txt", "w")
n = text_file.write(str_adoptable)
text_file.close()



#ADOPTABLE
df_adopted = pd.read_csv('../app/data/adopted_csv.csv')
df = df_adopted.fillna("None")  #impute empty records
df_str_adopted = df[["description"]].copy()
document_adopted = ' '.join(df_str_adopted['description'].tolist())
str_adopted = str(document_adopted )
text_file = open("../app/data/adopted_corpus.txt", "w")
n = text_file.write(str_adopted)
text_file.close()