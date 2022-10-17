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



### MODELLING AND TRAINING

def read_splitted_data():

    df_train = pd.read_csv('Data/Data_dat_dai/df_train_after.csv')
    df_test = pd.read_csv('Data/Data_dat_dai/df_test_after.csv')


    X_train = df_train[['text']]
    Y_train = df_train[['label']]

    X_test = df_test[['text']]
    Y_test = df_test[['label']]   

    return X_train, Y_train, X_test, Y_test

def tf_idf_config(X_train, X_test):

    Tfidf_vect = TfidfVectorizer(max_features=1000)
    Tfidf_vect.fit(X_train['text'].to_list())
    tfidf_train = Tfidf_vect.transform(X_train['text'])
    tfidf_test = Tfidf_vect.transform(X_test['text'])

    tfidf_train = tfidf_train.toarray()
    tfidf_test = tfidf_test.toarray()

    return Tfidf_vect, tfidf_train, tfidf_test

def model_selection(model_name, tfidf_train, tfidf_test, Y_train, Y_test):
    if (model_name == "SVM"):
        model = svm.SVC(C=1.0, kernel='linear', gamma='auto', degree = 3)
        model.fit(tfidf_train,Y_train)
        # predict the labels on validation dataset
        predictions_SVM = model.predict(tfidf_test)
        # Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100)  
        print(classification_report(Y_test,predictions_SVM)) 
        return model    

    elif (model_name == "LGBM"):

        model = LGBMClassifier()
        model.fit(tfidf_train,Y_train)
        predictions_lgbm = model.predict(tfidf_test)
        # Use accuracy_score function to get the accuracy
        print("ligbm Accuracy Score -> ",accuracy_score(predictions_lgbm, Y_test)*100)
        print(classification_report(Y_test,predictions_lgbm))

        return model

    elif (model_name == "Random Forest"):

        model = RandomForestClassifier(n_estimators=1000,min_samples_leaf=2,
                                        min_samples_split= 10, max_features = 'sqrt', criterion = 'entropy', bootstrap= True,
                                      random_state=25)
        model.fit(tfidf_train,Y_train) 
        predictions_rf = model.predict(tfidf_test)
        # Use accuracy_score function to get the accuracy
        print("rf Accuracy Score -> ",accuracy_score(predictions_rf, Y_test)*100)
        print(classification_report(Y_test,predictions_rf))

        return model
    else:
        print("No invalid model selected, try again")
        return 0


if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = read_splitted_data()
    Tfidf_vect, tfidf_train, tfidf_test = tf_idf_config(X_train, X_test)
    selected = model_selection(model_name = "Random Forest", tfidf_train = tfidf_train, tfidf_test = tfidf_test, Y_train = Y_train, Y_test = Y_test)

    pickle.dump(selected, open('Data/model/rf.pkl', 'wb'))

    pickle.dump(Tfidf_vect, open("Data/model/tfidf.pickle", "wb"))