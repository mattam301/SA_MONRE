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
    pickled_model = pickle.load(open('Data/model/lgbm.pkl', 'rb'))
    Tfidf_vect = pickle.load(open('Data/model/tfidf.pickle', 'rb'))

    # input  = """Bài 1: Quy hoạch chưa sát thực tế Công tác lập quy hoạch, kế hoạch sử dụng đất thời kỳ đầu (giai đoạn 2016-2020) của một số địa phương thời gian qua chất lượng chưa cao, chưa sát với tình hình thực tế và quá trình phát triển kinh tế - xã hội. Việc một số quy hoạch đã công bố, thời gian kéo dài, không triển khai hoặc chỉ triển khai một phần, không điều chỉnh quy hoạch, làm ảnh hưởng đến quyền lợi người dân trong vùng dự án. Bên cạnh đó, theo quy định, các dự án, công trình đưa vào kế hoạch sử dụng đất hàng năm của địa phương nhưng nếu quá 3 năm không triển khai thực hiện thì người sử dụng đất được quyền sử dụng đất của mình. Tuy nhiên, có một số công trình, dự án tiếp tục đưa vào kỳ sử dụng đất tiếp theo, gây thiệt thòi cho người sử dụng đất trong 3 năm khi không khai thác, thu được lợi nhuận trên mảnh đất của mình. Định hướng và khái quát Cụ thể của việc đánh giá trên, tại huyện Đầm Dơi cho thấy, trong quy hoạch sử dụng đất kỳ trước qua các năm 2016-2020, trên địa bàn huyện đã thực hiện được 263/672 hạng mục công trình, dự án (trong đó có 1 hạng mục thực hiện 1 phần của dự án đầu tư xây dựng hệ thống thuỷ lợi phục vụ nuôi thuỷ sản), đạt 39,14% về mặt tổng số công trình, dự án đã đăng ký, với tổng diện tích quy hoạch đã thực hiện là 591,18 ha/1.360,91 ha. Qua rà soát, trên địa bàn huyện Đầm Dơi có 6 công trình, dự án quá 3 năm nhưng chưa triển khai, trình cấp thẩm quyền xem xét huỷ bỏ theo quy định, gồm đường số 7, Khóm 4, thị trấn Đầm Dơi; lộ Tô Văn Mười nối dài (đoạn từ đường Nguyễn Tạo đến đường 30/4), Khóm 4, thị trấn Đầm Dơi; đê bao mương Chung Kiết - Khâu Mét; cụm dân cư Khóm 1 (phía Đông), thị trấn Đầm Dơi; khu dân cư (tái định cư) ấp Tân Long A, xã Tân Tiến; khu dân cư (tái định cư) ấp Thuận Long, xã Tân Tiến. Sau nhiều năm quy hoạch treo, lộ Tô Văn Mười nối dài (đoạn từ đường Nguyễn Tạo đến đường 30/4), Khóm 4, thị trấn Đầm Dơi, vẫn chưa có nguồn lực để thực hiện, được địa phương đưa ra khỏi quy hoạch trong giai đoạn tiếp theo. Ông Huỳnh Trung Quang, Chủ tịch UBND thị trấn Đầm Dơi, cho hay, trên địa bàn có 7 dự án tuyến lộ được quy hoạch từ năm 2010, nhưng đến nay chỉ làm được 1 tuyến. Xa hơn nữa, quy hoạch trung tâm thương mại của huyện nằm trên địa bàn thị trấn có từ năm 2005, nhưng đến nay vẫn chưa được thực hiện. “Nhiều năm không thực hiện, đồng nghĩa là quy hoạch treo, phải huỷ bỏ. Tuy nhiên, có những dự án không thể huỷ, như trung tâm thương mại, vì nhận thấy thật sự cần thiết, nên vẫn phải tiếp tục thực hiện”, ông Quang nêu thực tế. Ông Nguyễn Phương Bình, Phó chủ tịch UBND huyện Đầm Dơi, cho biết, công tác dự báo, xác định nhu cầu sử dụng đất trong quy hoạch kỳ trước của các ngành, đơn vị trên địa bàn huyện còn chậm, chưa rõ ràng và thống nhất về vị trí thực hiện, tên dự án, quy mô dự án. Do đó, trong quá trình tổ chức thực hiện gặp rất nhiều khó khăn do có điều chỉnh tên, quy mô, vị trí thực hiện dự án. Tuy công tác mời gọi đầu tư được tổ chức thực hiện tốt, nhưng vẫn còn nhiều cơ hội đầu tư bị bỏ lỡ. “Nguyên nhân chính là do vị trí quy hoạch dự án chưa đáp ứng được mong muốn của nhà đầu tư về vị trí tiếp giáp, tiềm năng phát triển..., ngược lại, nhà đầu tư đề xuất vị trí lại có đơn giá bồi thường giải phóng mặt bằng khá cao do thuộc các khu vực có nhiều hộ dân sinh sống, giá đất cao...”, ông Bình nhìn nhận thực tế dẫn đến việc triển khai các công trình, dự án được quy hoạch sử dụng đất thời gian qua trên địa bàn đạt kết quả thấp. Một vấn đề được lãnh đạo UBND huyện Đầm Dơi nêu thực tế trên địa bàn là công tác quy hoạch phát triển cụm công nghiệp tiến triển chậm, chưa khai thác hết tiềm năng của địa phương. Cụ thể ở đây là quy hoạch cụm công nghiệp Tân Thuận 35 ha, cụm công nghiệp Nguyễn Huân 50 ha, cụm công nghiệp thị trấn Đầm Dơi 35 ha… “Vị trí quy hoạch cụm công nghiệp trong quy hoạch kỳ trước nằm trong khu vực đô thị, dẫn đến giá trị bồi thường, hỗ trợ tái định cư cao; quy mô còn nhỏ, tiềm năng mở rộng không lớn do tiếp giáp với khu đô thị trung tâm, hạ tầng giao thông đấu nối chưa đảm bảo khả năng phát triển công nghiệp. Cùng với đó, công tác khảo sát, nghiên cứu lập quy hoạch chi tiết 1/500, đồ án quy hoạch các tỷ lệ còn chậm, chưa thực hiện đồng bộ với thời điểm xây dựng quy hoạch sử dụng đất. Nhiều dự án được quy hoạch chỉ mang tính định hướng, khái quát nhằm mục tiêu mời gọi đầu tư”, ông Bình trần tình. Được quy hoạch từ rất lâu (năm 2005), đến nay Trung tâm Thương mại huyện Đầm Dơi vẫn chưa được triển khai thực hiện. Cùng chung thực trạng Cũng với thực trạng trên nhưng có thời gian dài hơn, suốt giai đoạn 2011-2020, huyện Năm Căn chỉ thực hiện được 234/712 công trình, dự án đã đăng ký thực hiện quy hoạch (đạt 32,87%), với diện tích 13.282,98/20.546,74 ha. Qua rà soát, địa phương loại bỏ 324 công trình, dự án với diện tích trên 6.346 ha cho giai đoạn đến năm 2030, vì xét thấy không còn phù hợp với điều kiện phát triển kinh tế - xã hội của huyện. Tại huyện Trần Văn Thời, giai đoạn 2016-2020 đã triển khai thực hiện được 243/796 hạng mục công trình, dự án thuộc trường hợp thực hiện thủ tục thu hồi đất, giao đất, cho thuê và chuyển mục đích sử dụng đất, đạt 30,53% về tổng số công trình, dự án đã đăng ký, với tổng diện tích đã thực hiện là 348,65 ha/ 2.946,06 ha, đạt 13,83% về diện tích thực hiện. Bên cạnh các dự án đã được phê duyệt, trong kỳ quy hoạch trên, huyện Trần Văn Thời đã thực hiện hoàn thành 17 dự án được phê duyệt bổ sung với tổng diện tích 1.390,22 ha. Cũng trong thời kỳ quy hoạch này, địa phương đã điều chỉnh tạm ngưng thực hiện 184/796 hạng mục công trình, dự án, tổng diện tích 673,88 ha. Với những con số trên, ông Hồ Song Toàn, Phó chủ tịch UBND huyện Trần Văn Thời, cho rằng, kết quả thực hiện các công trình, dự án trong quy hoạch sử dụng đất đến năm 2020 trên địa bàn đáp ứng tốt các mục tiêu phát triển kinh tế - xã hội theo từng giai đoạn trong 10 năm qua. Thừa nhận về kết quả thực hiện chưa đạt tỷ lệ cao, tuy nhiên, ông Toàn khẳng định thành quả đạt được là sự thay đổi lớn về kết cấu hạ tầng, chất lượng cuộc sống của người dân ngày càng được cải thiện; nhiều công trình, dự án đã thực hiện và được đưa vào khai thác đạt hiệu quả cao. Dự án đầu tư xây dựng khu dân cư bờ Nam Sông Đốc, với diện tích 287,07 ha, là 1 trong 4 dự án đã đưa vào kế hoạch sử dụng đất hàng năm, nhưng không triển khai thực hiện nên huyện đề nghị huỷ bỏ và được HĐND tỉnh ban hành Nghị quyết số 08/2021/NQ-HĐND, ngày 4/12/2021 chấp thuận. Tại thành phố trung tâm của tỉnh, trong quy hoạch sử dụng đất kỳ trước qua các năm 2016-2020, trên địa bàn TP Cà Mau đã đăng ký thực hiện 486 hạng mục công trình, dự án. Kết quả đã thực hiện được 124/486 hạng mục công trình, dự án (trong đó, có 5 dự án đã thực hiện một phần và chuyển tiếp sang kỳ quy hoạch mới để tiếp tục thực hiện), đạt 25,51% về mặt tổng số công trình, dự án đã đăng ký với tổng diện tích quy hoạch đã thực hiện là 314,98 ha/4.134,42 ha. Ngoài ra, giai đoạn 2018-2021, thành phố đã điều chỉnh tạm ngưng thực hiện 61 hạng mục công trình, dự án với tổng diện tích 796,34 ha. Chuyển tiếp sang kỳ quy hoạch 2021-2030 tiếp tục thực hiện với 306 hạng mục công trình, dự án. “TP Cà Mau trình cấp thẩm quyền xem xét huỷ bỏ công trình dự án quá 3 năm nhưng chưa triển khai, có thể kể đến Dự án nhà ở công nhân Phường 8 (đường Vành đai Tây Nam thuộc địa bàn Phường 8 và xã Lý Văn Lâm); Dự án khu nhà ở công nhân lao động Công ty Minh Phú; Dự án đầu tư xây dựng kết cấu hạ tầng đường Nguyễn Mai...”, ông Bùi Tứ Hải, Phó chủ tịch UBND TP Cà Mau, thông tin. Đối với cấp tỉnh, con số thống kê cho thấy, danh mục các công trình, dự án theo Nghị quyết 84/NQ-CP, ngày 7/10/2019 của Chính phủ có 498 công trình, dự án. Trong đó, tỉnh đã triển khai lập thủ tục đất đai 82 công trình, dự án; trình HĐND tỉnh ban hành nghị quyết huỷ bỏ 36 công trình, dự án, với diện tích 591,32 ha; điều chỉnh 17 công trình, dự án, với diện tích 430,57 ha./. Trần Nguyên BÀI 2: GIAO CHỈ TIÊU LỆCH SO VỚI THỰC TẾ
    # """
    # validate_sample(input, pickled_model, Tfidf_vect)
    df = pd.read_csv("Data/Data_dat_dai/unlabelled.csv")
    print("Data reading completed")

    df = df.sample(1000)
    print("Data sampling completed")

    pred = predict_sample(df, pickled_model, Tfidf_vect)
    print("Output predicted")

    df['label'] = pred
    print("Start extracting file")
    df.to_csv('auto_labelling.csv', encoding = 'utf-8-sig')