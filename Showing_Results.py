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
    pickled_model = pickle.load(open('Data/model/svm.pkl', 'rb'))
    Tfidf_vect = pickle.load(open('Data/model/tfidf.pickle', 'rb'))

    df = pd.read_excel('Data/Data_dat_dai/100_sample_T.xlsx')

#     input  = """Ngày 6.8, tin từ UBND tỉnh Quảng Ngãi, Chủ tịch UBND tỉnh Quảng Ngãi có văn bản yêu cầu rà soát lại tất cả các dự án đã có quyết định chủ trương đầu tư ngoài ngân sách mà chưa triển khai thực hiện thì đề xuất UBND tỉnh có biện pháp xử lý ngay, không để xảy ra tình trạng có quyết định chủ trương đầu tư mà các chủ đầu tư cứ giữ đất không triển khai thực hiện. Khu du lịch sinh thái Thiên Đàng, xã Bình Thạnh, H.Bình Sơn, thuộc Khu kinh tế Dung Quất triển khai 17 năm vẫn chưa hoạt động HẢI PHONG Trong đó, Chủ tịch UBND tỉnh Quảng Ngãi yêu cầu Trưởng ban quản lý Khu kinh tế Dung Quất và các khu công nghiệp tỉnh Quảng Ngãi, chủ tịch UBND các huyện, thị xã, thành phố và cơ quan, đơn vị liên quan thực hiện đầy đủ trách nhiệm về quản lý đất đai, tuyệt đối không để xảy ra việc tổ chức, cá nhân tự ý triển khai san lấp mặt bằng, xây dựng công trình trên phần diện tích đất thực hiện dự án khi chưa có quyết định giao (cho thuê) đất của cấp có thẩm quyền. Trong quá trình theo dõi và kiểm tra, khi phát hiện những trường hợp vi phạm phải xử lý nghiêm khắc theo quy định của pháp luật về đất đai. UBND tỉnh Quảng Ngãi giao Sở TN-MT Quảng Ngãi kiểm tra, rà soát, không tham mưu việc giao (cho thuê) đất, đối với các tổ chức có hành vi vi phạm về đất đai. Ngoài ra, liên quan việc thu hút đầu tư và tiến độ triển khai các dự án trên địa bàn tỉnh, Chủ tịch UBND tỉnh Quảng Ngãi Đặng Văn Minh còn yêu cầu các đơn vị, địa phương tăng cường rà soát, tham mưu UBND tỉnh thu hồi các dự án triển khai chậm tiến độ, nhất là các dự án mà nhà đầu tư thiếu năng lực, xin dự án để xí phần, tránh lặp lại những thiếu sót trong công tác quản lý, thu hút đầu tư. \n Chủ tịch UBND tỉnh Quảng Ngãi giao Sở KHĐT tỉnh Quảng Ngãi rà soát lại tất cả các dự án đầu tư đã có quyết định chủ trương đầu tư ngoài ngân sách mà chưa triển khai thực hiện, thì phải đề xuất UBND tỉnh có biện pháp xử lý ngay. "Không để xảy ra tình trạng, quyết định chủ trương đầu tư nằm trên giấy rồi các chủ đầu tư cứ giữ đất không triển khai thực hiện. Rút kinh nghiệm từ Khu kinh tế Dung Quất là điển hình trong việc buông lỏng quản lý đối với các dự án đã cấp quyết định chủ trương đầu tư nhưng hiện nay không triển khai, không xử lý được”, ông Đặng Văn Minh cho biết. Tại Khu kinh tế Dung Quất và các Khu công nghiệp Quảng Ngãi, hiện có 348 dự án cấp phép còn hiệu lực (52 dự án vốn FDI), trong đó có 243 dự án đã xây dựng hoàn thành và đi vào hoạt động, còn 105 dự án đang trong quá trình triển khai, trong đó có rất nhiều dự án chây ì, chậm tiến độ trong thời gian dài. Tin liên quan Những công trình 'làm nghèo' đất nước: Trung tâm dạy nghề kiểu mẫu thành bãi hoang Bình Định: Chấm dứt hoạt động 2 dự án thuộc Tập đoàn FLC tại Quy Nhơn Bình Thuận: Vì sao thu hồi biên bản bàn giao khu đất đang bị điều tra? Tách hành vi của Tổng giám đốc Công ty Quốc Cường Gia Lai tiếp tục điều tra ông đặng văn minh, chủ tịch ubnd tỉnh quảng ngãi, yêu cầu các địa phương và ngành chức năng xem xét, xử lý các dự án chây ì triển khai do thiếu năng lực từ phía nhà đầu tư hoặc "xí đất" lấy phần.
# """
#     validate_sample(input, pickled_model, Tfidf_vect)
    df = pd.read_csv("Data/Data_dat_dai/unlabelled.csv")
    print("Data reading completed")

    df = df.sample(500)
    print("Data sampling completed")

    pred = predict_sample(df, pickled_model, Tfidf_vect)
    print("Output predicted")

    df['label'] = pred
    print("Start extracting file")
    df.to_csv('Data/Result/auto_labelling_full_lgbm.csv', encoding = 'utf-8-sig')