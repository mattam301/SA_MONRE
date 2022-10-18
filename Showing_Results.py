import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from underthesea import word_tokenize
import pickle

stopwords_path = "Data/Data_ref/vietnamese-stopwords.txt"

### PREPROCESSING

def lower_df(df):
    df['text'] = df['text'].str.lower()
    return df

def segmentation(df):


    '''UNDER THE SEA'''
    
    list_text = df['text'].to_list()
    #print(list_text[0])
    for i in range(len(list_text)):
        list_text[i] = word_tokenize(list_text[i], format='text')
    
    df['text'] =  list_text
    return df

def get_stopwords_list(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

def remove_stopwords(df):
    stopwords = get_stopwords_list(stopwords_path)

    list_text = df['text'].tolist()

    results = []
    for text in list_text:
        tmp = text.split(' ')
        for stop_word in stopwords:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    df['text'] = results
    return df

def data_preprocessing(df):
    df = lower_df(df)
    df = segmentation(df)
    df = remove_stopwords(df)

    return df

def text_to_df(tmp_text):
    input = tmp_text
    list_input = [input]
    test_df = pd.DataFrame()
    test_df['text'] = list_input    
    return test_df

def predict_sample(test_df, selected, Tfidf_vect):
    # test_df = test_df[['text']]
    test_df = data_preprocessing(test_df)
    test_df = Tfidf_vect.transform(test_df['text'])
    test_df = test_df.toarray()
    prediction = selected.predict(test_df)   

    return prediction

def validate_sample(tmp_text, selected, Tfidf_vect):
    tmp_df = text_to_df(tmp_text)
    output = predict_sample(tmp_df, selected, Tfidf_vect)

    if (output[0] == -1):
        print("Bai viet co kha nang tieu cuc cao !!!")   
    else:
        print("Bai viet co kha nang tieu cuc thap :3")

if __name__ == '__main__':
    pickled_model = pickle.load(open('Data/model/lgbm.pkl', 'rb'))
    Tfidf_vect = pickle.load(open('Data/model/tfidf.pickle', 'rb'))

    # input  = """"""
    # validate_sample(input, pickled_model, Tfidf_vect)
    df = pd.read_csv("Data/Data_dat_dai/unlabelled.csv")
    print("Data reading completed")

    # df = df.sample(500)
    print("Data sampling completed")

    pred = predict_sample(df, pickled_model, Tfidf_vect)
    print("Output predicted")

    df['label'] = pred
    print("Start extracting file")
    df.to_csv('auto_labelling_full_lgbm.csv', encoding = 'utf-8-sig')