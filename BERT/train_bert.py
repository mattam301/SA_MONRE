import os
from BERT import BERT_FEATURES
from Preprocesser import Preprocesser
from constant import STOP_WORD_PATH
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle 


if __name__ == '__main__':
    df_train = pd.read_csv('data/df_train (3).csv')
    df_test = pd.read_csv('data/df_test (3).csv')
    print("Start Preprocessing....")
    pre = Preprocesser(STOP_WORD_PATH)
    df_train = pre.processing(df_train)
    df_test = pre.processing(df_test)

    bert = BERT_FEATURES()

    text_train = df_train['text'].to_list()
    label_train = df_train['label'].to_list()
    text_test = df_test['text'].to_list()
    label_test = df_test['label'].to_list()

    features_train = bert.make_bert_features(text_train)
    features_test = bert.make_bert_features(text_test)

    print("Start training....")
    rf = RandomForestClassifier(n_estimators=550, min_samples_leaf=2,
                                min_samples_split=10, max_features='sqrt', criterion='entropy', bootstrap=True,
                                random_state=42)
    rf.fit(features_train, label_train)
    predictions_rf = rf.predict(features_test)
    print("rf Accuracy Score -> ", accuracy_score(predictions_rf, label_test) * 100)
    print(classification_report(label_test, predictions_rf))

    pickle.dump(pre,open("models/preprocesser.pkl","wb"))
    pickle.dump(bert, open("models/bert_gen_feat.pkl", "wb"))
    pickle.dump(rf, open("models/Random_forest_bert.pkl", "wb"))

    




