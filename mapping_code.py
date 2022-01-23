# please pip install google translator
# pip install googletrans==3.1.0a0

import pandas as pd
import numpy as np
from googletrans import Translator
from unidecode import unidecode
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("max_colwidth", 300)

#####Please change it here and run your function##
input_loc = '/home/aksve/hotel_hackathon/mappinghotelsdataset.xlsx' #change this directory and keep rest of the information as same

p1cols = ['p1.hotel_name','p1.city_name','p1.hotel_address', 'p1.postal_code','p1.country_code']
p2cols = ['p2.hotel_name','p2.city_name','p2.hotel_address', 'p2.postal_code','p2.country_code']
total_cols = p1cols + p2cols

def data_cleaning(df,key_var,cols):
    
    op = {}
    
    for i in range(len(cols)):
        
        print(cols[i])
        df[cols[i]] = df[cols[i]].astype(str)#convert the data type to string
        df[cols[i]] = df[cols[i]].fillna('miss')#replace missing with a string
        
        df[cols[i]] = df[cols[i]].str.lower()#lowercase the string
        df[cols[i]] = df[cols[i]].str.replace('[^\w\s]','')#remove punctuation marks
        
        non_eng = df[~df[cols[i]].map(lambda x: x.isascii())]#fetch non english data
        print(non_eng.shape,non_eng[cols[i]].head(5))
        translator = Translator()
        non_eng[cols[i]] = non_eng[cols[i]].apply(lambda x: translator.translate(x, dest='en').text)#convert non english data
        
        non_eng[cols[i]] = non_eng[cols[i]].apply(unidecode)#unidecode the data with extra symbols
        
        #Clean non_eng_database
        non_eng[cols[i]] = non_eng[cols[i]].str.lower()#again need to lower
        non_eng[cols[i]] = non_eng[cols[i]].str.replace('[^\w\s]','')#again need to punctuate
        
        f = non_eng.append(df[~df[key_var].isin(non_eng[key_var])])#append the dataset
        f[cols[i]] = f[cols[i]].str.replace(' ', '')
        
        op[cols[i]] = f[cols[i]]
    
    df = df.drop(cols,axis=1) #Drop the new columns
    df.reset_index(drop=True,inplace=True)
    
    df1 = pd.concat([df,pd.DataFrame(op)], axis=1)
    df1['combined_str'] = df1[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    return df1  
        

def tf_idf_vectorizer(df,vectorizer,star):
    
    X1 = vectorizer.transform(df['combined_str']) #convert the data to matrix form and create grams
    s1 = (X1.toarray())#Pass it from sparse array to full array for concatenation
    a = np.array(df[star]/1000)#generate the star rating feature
    a = a.reshape(s1.shape[0],1)#Reshape for stacking it 
    s1 = np.hstack([s1,a])#Stack the two arrays
    s1 = sparse.csr_matrix(s1)#Convert it to sparse matrix
    
    return s1  

def data_selection(p1,p2,similarity,debugging):
    a = list(product(p1['p1.key'], p2['p2.key']))
    
    #Passing the data to a dataframe
    newdf = pd.DataFrame(data=a, columns=['p1.key','p2.key'])
    simdf = pd.DataFrame(data=similarity,columns=['cos_sim'])
    
    #Get the cosine similarity score
    newdf = pd.concat([newdf,simdf], axis=1)
    
    #Country codes are fetched separately to remove the rows which should not be considered
    p1_keys = ['p1.key','p1.country_code']
    p2_keys = ['p2.key','p2.country_code']
    
    newdf = pd.merge(newdf,p1[p1_keys],how='left',on = 'p1.key')
    newdf = pd.merge(newdf,p2[p2_keys],how='left',on = 'p2.key')
    
    newdf = newdf[newdf['p1.country_code'] == newdf['p2.country_code']]#Filter out combinations - This is an additional step to save memory
    
    if debugging == 1:
    
        ##Get all the P1 key and P2 keys
        p1_keys = ['p1.key','p1.hotel_name','p1.city_name','p1.hotel_address',\
          'p1.star_rating','p1.postal_code']
        p2_keys = ['p2.key','p2.hotel_name','p2.city_name','p2.hotel_address',\
          'p2.star_rating','p2.postal_code']
    
        newdf = pd.merge(newdf,p1[p1_keys],how='left',on = 'p1.key')
        newdf = pd.merge(newdf,p2[p2_keys],how='left',on = 'p2.key')
    
    #Choosing the top ranked key
    tmp = newdf.groupby('p1.key').cos_sim.nlargest(1).reset_index()
    merged = pd.merge(newdf, tmp, left_index=True, right_on='level_1',how='inner')
    merged.rename(columns={"p1.key_x": "p1.key"},inplace=True)
    print(merged.shape)
    merged.drop_duplicates(subset=['p1.key','p2.key'],inplace=True)
    print(merged.shape)
    
    return merged
    

def output_func(input_file):
    xlsx_file= pd.ExcelFile(input_file)
    #Process 3 different files
    p1 = pd.read_excel(xlsx_file, 'Partner1')
    p2 = pd.read_excel(xlsx_file, 'Partner2')
    match = pd.read_excel(xlsx_file, 'examples')
    
    #Clean the requisite datafile
    p1_proc = data_cleaning(p1,'p1.key',p1cols)
    p2_proc = data_cleaning(p2,'p2.key',p2cols)
    
    #TF-IDF vectorizer called here - n-grams are fixed to try 2,3 and 4. It can be changed also
    vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,4))
    vectorizer.fit(p1_proc['combined_str'].append(p2_proc['combined_str']))
    
    #We have treated star rating in a different way and hence, given in a separate way
    s1,s2 = tf_idf_vectorizer(p1_proc,vectorizer,'p1.star_rating'),tf_idf_vectorizer(p2_proc,vectorizer,'p2.star_rating')
    
    similarities = cosine_similarity(s1,s2)#Generate a matrix for similarities
    similarities = similarities.reshape(s1.shape[0]*s2.shape[0],1)#Reshaping the data
    
    df1 = data_selection(p1_proc,p2_proc,similarities,0)
    
    return df1   


final = output_func(input_loc) #Use this final dataset for processing

