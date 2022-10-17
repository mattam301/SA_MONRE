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
    # X_train, Y_train, X_test, Y_test = read_splitted_data()
    # Tfidf_vect, tfidf_train, tfidf_test = tf_idf_config(X_train, X_test)
    # selected = model_selection(model_name = "LGBM", tfidf_train = tfidf_train, tfidf_test = tfidf_test, Y_train = Y_train, Y_test = Y_test)


    input  = """Gia đình người nước ngoài tìm hiểu dự án để mua nhà ở tại Khu đô thị Phú Mỹ Hưng (TP.HCM). Ảnh: Lê Toàn Dự thảo đang bỏ ngỏ Tuần qua, Dự thảo Luật Đất đai (sửa đổi) đã được đặt lên bàn nghị sự của Chính phủ trong phiên họp chuyên đề về xây dựng pháp luật do Thủ tướng Chính phủ Phạm Minh Chính chủ trì. Một trong những nội dung được các thành viên Chính phủ quan tâm thảo luận là quyền của người nước ngoài liên quan đến đất đai tại Việt Nam. Đây cũng chính là vấn đề được nhiều doanh nhân, chuyên gia quan tâm, góp ý, bởi đang có vướng mắc với một số luật khác. Cụ thể, Điều 159, Luật Nhà ở quy định tổ chức, cá nhân người nước ngoài là một trong những đối tượng được sở hữu nhà ở tại Việt Nam. Khoản 2, Điều 14, Luật Kinh doanh bất động sản quy định người Việt Nam định cư ở nước ngoài, tổ chức, cá nhân nước ngoài được thuê các loại bất động sản để sử dụng, được mua, thuê nhà ở theo quy định của pháp luật về nhà ở. “Nội dung về quyền sở hữu nhà ở tại Việt Nam của tổ chức, cá nhân nước ngoài đã được Bộ Xây dựng báo cáo và Bộ Chính trị thống nhất đưa vào Luật Nhà ở”, Bộ Tài nguyên và Môi trường nêu rõ trong hồ sơ Dự án Luật Đất đai (sửa đổi). Tuy nhiên, Điều 5, Luật Đất đai hiện hành quy định về người sử dụng đất chỉ quy định tổ chức nước ngoài được thực hiện dự án nhà ở tại Việt Nam, không quy định cá nhân là người nước ngoài được sử dụng đất ở tại Việt Nam. Để đảm bảo tính thống nhất, đồng bộ của hệ thống pháp luật, cần sửa đổi Luật Đất đai theo hướng bổ sung đối tượng cá nhân nước ngoài được sở hữu nhà ở tại Việt Nam theo pháp luật về nhà ở thì được sử dụng đất theo pháp luật về đất đai. Nhưng, theo Bộ Tài nguyên và Môi trường (cơ quan chủ trì soạn thảo Dự án Luật Đất đai sửa đổi), đây không phải là phương án duy nhất. “Hầu hết các nước trên thế giới đều có các quy định nhằm hạn chế hoặc kiểm soát các quyền và nghĩa vụ của người nước ngoài liên quan đến đất đai, bất động sản ở các mức độ khác nhau. Trong bối cảnh đất đai ở nước ta thuộc sở hữu toàn dân, nên việc trao quyền sử dụng đất cho người nước ngoài cần thận trọng và phải trên cơ sở quan điểm chỉ đạo của Trung ương. “Trong quá trình tổng kết Nghị quyết số 19-NQ/TW, vấn đề này đã được đưa vào báo cáo, tuy nhiên, sau đó đã có chỉ đạo đưa ra”, cơ quan chủ trì soạn thảo nêu lý do cần thận trọng. Từ giải thích này, Bộ Tài nguyên và Môi trường đề xuất hai phương án xử lý. Phương án 1: Bộ Xây dựng và Bộ Tài nguyên và Môi trường phối hợp trong quá trình dự thảo luật tiếp tục báo cáo Ban Cán sự đảng Chính phủ, Đảng đoàn Quốc hội xem xét báo cáo Bộ Chính trị cho phép người nước ngoài sở hữu nhà ở tại Việt Nam được sử dụng đất tại Việt Nam. Theo đó, Dự thảo sẽ sửa đổi quy định về người sử dụng đất tại Điều 5, Luật Đất đai và các quy định tại chương quy định về quyền và nghĩa vụ của người sử dụng đất để đảm bảo tương thích sau khi có ý kiến chỉ đạo của cấp có thẩm quyền. Phương án 2 được tính đến là, Nghị quyết số 18-NQ/TW đã có quy định về việc sử dụng không gian ngầm, không gian trên không. Vì vậy, có thể xem xét giải quyết vấn đề sử dụng đất của tổ chức, cá nhân nước ngoài thông qua bổ sung quy định mới liên quan đến người nước ngoài được sử dụng không gian xây dựng công trình gắn liền với đất giống như kinh nghiệm của một số nước hiện nay cho phép đất của một người nhưng tài sản trên đất thuộc sở hữu của người khác. Tuy nhiên, Dự thảo cũng chưa có quy định về vấn đề này, tức là vẫn đang bỏ ngỏ cả hai phương án. Theo quy định tại Điều 6, Dự thảo, “cá nhân người nước ngoài” không được xem là người sử dụng đất, không được Nhà nước giao đất, cho thuê đất, công nhận quyền sử dụng đất, nhận chuyển quyền sử dụng đất; thuê lại đất trong khu công nghiệp, cụm công nghiệp, khu công nghệ cao. Cần thống nhất Thông tin từ cuộc họp chuyên đề xây dựng pháp luật của Chính phủ tháng 8/2022 cho biết, đối với nội dung về tiếp cận đất đai của tổ chức kinh tế có vốn đầu tư nước ngoài, Thủ tướng yêu cầu bổ sung quy định về hạn chế và kiểm soát được việc tiếp cận các khu vực trọng yếu, nhạy cảm về quốc phòng, an ninh. Ngoài ra, Thủ tướng cũng chỉ đạo cân nhắc bổ sung quy định người nước ngoài thuộc đối tượng được phép sở hữu nhà ở theo pháp luật về nhà ở thì được quyền sử dụng đất ở để có căn cứ cấp giấy chứng nhận quyền sử dụng đất và tài sản gắn liền với đất nhằm thống nhất với Dự án Luật Nhà ở (sửa đổi) về đối tượng được sở hữu nhà ở là người nước ngoài. Xem xét lại quy định về người sử dụng đất đối với người nước ngoài để đảm bảo tính đồng bộ trong hệ thống pháp luật cũng là quan điểm của Liên đoàn Thương mại và Công nghiệp Việt Nam (VCCI) khi tham gia thẩm định Dự thảo Luật Đất đai (sửa đổi). Giữ như quy định hiện hành, nếu người nước ngoài bán nhà cho người Việt Nam thì vô hình trung, quyền của người mua là người Việt Nam sẽ không được đảm bảo. Đó là điều bất hợp lý mà theo VCCI, cần sửa đổi. Vướng mắc ở quy định về quyền sử dụng đất của người nước ngoài tại Việt Nam cũng được ông Nguyễn Văn Đỉnh, chuyên gia pháp lý đất đai, đầu tư xây dựng, kinh doanh bất động sản đề cập khá sâu trong hội thảo đầu tiên góp ý về Dự thảo. Ông Đỉnh phân tích, về mặt logic, người bán (người nước ngoài) không có quyền sử dụng đất thì đương nhiên người mua cũng không có quyền sử dụng đất (bởi không được nhận chuyển giao quyền này từ người bán). Như thế, người Việt Nam mua nhà ở của người nước ngoài chịu quy chế pháp lý như người nước ngoài (chỉ có quyền sở hữu nhà ở, mà không gắn với quyền sử dụng đất). Mặt khác, trong tờ trình đề nghị xây dựng Luật Nhà ở sửa đổi (dự kiến thông qua cùng thời điểm với Luật Đất đai sửa đổi), Bộ Xây dựng đề xuất tập trung giải quyết 8 nhóm chính sách lớn, đầu tiên là chính sách về sở hữu nhà ở: “Tiếp tục chính sách khuyến khích, tạo điều kiện thuận lợi cho cá nhân, tổ chức nước ngoài được mua và sở hữu nhà ở tại Việt Nam, phù hợp với thông lệ quốc tế, thu hút đầu tư, đồng thời vẫn bảo đảm an ninh, quốc phòng”. Nhưng chính sách khuyến khích cá nhân nước ngoài mua nhà ở sẽ không thể thực hiện trọn vẹn nếu không giải quyết được tận gốc vấn đề chứng nhận quyền sở hữu. “Để giải quyết tận gốc, Luật Đất đai (sửa đổi) cần thiết phải ghi nhận quyền sử dụng đất của cá nhân nước ngoài”, vị chuyên gia này góp ý. Sau khi được Ủy ban Kinh tế thẩm tra, Ủy ban Thường vụ Quốc hội cho ý kiến, xin ý kiến đại biểu chuyên trách, Dự thảo Luật Đất đai (sửa đổi) sẽ được trình Quốc hội tại Kỳ họp thứ tư (tháng 10/2022). Tránh lạm dụng trong thu hồi đất Sửa đổi Luật Đất đai, một trong các vấn đề được đặc biệt quan tâm là cơ chế thu hồi đất. Về vấn đề này, Thủ tướng yêu cầu, đối với các trường hợp Nhà nước thu hồi đất, cần cụ thể hóa trong Dự thảo các tiêu chí, điều kiện đối với trường hợp Nhà nước thu hồi đất, tránh tình trạng lạm dụng các trường hợp phát triển kinh tế để thu hồi đất, gây bức xúc trong nhân dân, nhất là thu hồi cho dự án nhà ở thương mại. Thủ tướng cũng lưu ý, đối với trường hợp dự án khu đô thị, dự án nhà ở thương mại vừa thuộc đối tượng thu hồi đất, vừa thuộc đối tượng nhận chuyển nhượng, thì cần có quy định cụ thể khi nào thực hiện thu hồi đất, khi nào thực hiện nhận chuyển nhượng.
    """
    pickled_model = pickle.load(open('Data/model/lgbm.pkl', 'rb'))
    Tfidf_vect = pickle.load(open('Data/model/tfidf.pickle', 'rb'))
    validate_sample(input, pickled_model, Tfidf_vect)