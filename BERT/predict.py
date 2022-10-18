import pickle
from constant import STOP_WORD_PATH
from Preprocesser import Preprocesser
import pandas as pd

def predict_news(summary, content):
    df = pd.DataFrame()
    df['summary'] = [summary]
    df['content'] = [content]
    pre = pickle.load(open("models/preprocesser.pkl","rb"))
    bert = pickle.load(open("models/bert_gen_feat.pkl","rb"))
    rf = pickle.load(open("models/Random_forest_bert.pkl","rb"))

    pre = Preprocesser(STOP_WORD_PATH)
    df = pre.processing(df)
    text = df['text'].to_list()
    features = bert.make_bert_features(text)
    prediction = rf.predict(features)

    return prediction[0]

if __name__ == '__main__':
    print(predict_news("""(NLĐO)- Bị cáo buộc cùng thuộc cấp gây thất thoát hàng ngàn tỉ đồng trong vụ án khu đất vàng tại Bình Dương, bị cáo Trần Văn Nam, nguyên bí thư Tỉnh ủy Bình Dương, ngày mai 15-8 hầu toà tại Hà Nội.
""", """Theo dự kiến ngày mai 15-8, TAND TP Hà Nội mở phiên tòa xét xử sơ thẩm 28 bị cáo trong vụ án xảy ra tại Tổng công ty Sản xuất - Xuất nhập khẩu Bình Dương TNHH MTV (Tổng công ty 3-2). Phiên toà dự kiến diễn ra trong khoảng 20 ngày, kể cả ngày thứ 7 và chủ nhật. Ông Trần Văn Nam thời điểm còn tại vị Trong vụ án này, bị cáo Trần Văn Nam, cựu bí thư Tỉnh ủy Bình Dương; Trần Thanh Liêm, cựu chủ tịch UBND tỉnh Bình Dương, và 20 bị cáo là cựu lãnh đạo Tỉnh ủy, UBND tỉnh và nhiều sở ngành của Bình Dương cùng bị truy tố tội "Vi phạm quy định về quản lý, sử dụng tài sản nhà nước gây thất thoát, lãng phí"và tội "Tham ô tài sản". Bị cáo Nguyễn Văn Minh, cựu chủ tịch Tổng công ty 3-2; Trần Nguyên Vũ, cựu tổng giám đốc tổng công ty SX-XNC Bình Dương-Công ty Cổ phần; và Huỳnh Thanh Hải, chủ tịch Công ty TNHH MTV Đầu tư và quản lý dự án Bình Dương, cùng bị truy tố tội "Vi phạm quy định về quản lý, sử dụng tài sản nhà nước gây thất thoát, lãng phí"và tội "Tham ô tài sản". Bị cáo Võ Hồng Cường, Nguyễn Thục Anh và Trần Đình Như Ý đều bị truy tố về tội "Tham ô tài sản". Phiên tòa do Thẩm phán Vũ Quang Huy làm chủ tọa. Trong vụ án này, TAND TP Hà Nội xác định bị hại là Tổng công ty 3-2. Những người có quyền lợi, nghĩa vụ liên quan, bao gồm Tỉnh ủy Bình Dương, Văn phòng Tỉnh ủy Bình Dương, UBND tỉnh Bình Dương, Sở Tài chính tỉnh Bình Dương, Sở Tài nguyên và Môi trường tỉnh Bình Dương… Theo cáo trạng, ngày 9-1-2012, bị can Trần Văn Nam, khi đó là Phó Chủ tịch UBND tỉnh Bình Dương, ký Công văn chấp thuận cho Tổng Công ty 3-2 được lập thủ tục giao đất Khu dịch vụ trong Khu liên hợp. Trên cơ sở Đơn xin giao đất của Tổng Công ty SX-XNK Bình Dương và Tờ trình của Sở Tài nguyên và Môi trường tỉnh Bình Dương, bị can Trần Văn Nam ký Quyết định giao đất có thu tiền sử dụng đất khu đất có diện tích 43 ha và 145 ha cho Tổng Công ty 3-2. Theo quy định của pháp luật về đất đai, giá đất để thu tiền sử dụng đất đối với Tổng Công ty 3-2 được áp dụng tại thời điểm giao đất nêu trên. Tuy nhiên, các bị can thuộc Cục thuế tỉnh Bình Dương và các bị can thuộc Văn phòng UBND tỉnh Bình Dương đã tham mưu, đề xuất cho áp dụng đơn giá là 51.914 đồng/m2 theo Quyết định của UBND tỉnh Bình Dương ban hành từ ngày 27-12-2006 để tính thu tiền sử dụng đất đối với Tổng Công ty 3-2. Bị can Trần Văn Nam, với chức trách, nhiệm vụ được phân công, biết rõ nội dung đề xuất áp đơn giá đất bình quân năm 2006 để thu tiền sử dụng đất theo các quyết định giao đất năm 2012 và 2013 là trái với quy định của pháp luật nhưng vẫn ký ban hành Công văn ngày 23-1-2012 để thu tiền sử dụng đất cho Tổng Công ty 3-2. Việc này đã gây thất thoát cho Nhà nước hơn 761 tỉ đồng. Theo cáo trạng, cơ quan điều tra còn xác định năm 2016, với động cơ cá nhân nhằm chiếm đoạt, hưởng lợi từ các khu đất xin giao làm Dự án, thông qua hình thức liên doanh với Công ty cổ phần bất động sản Âu Lạc do Nguyễn Đại Dương (con rể ông Minh) thành lập, điều hành hoạt động để thành lập Công ty TNHH Đầu tư Xây dựng Tân Phú, trong đó Tổng Công ty SX-XNK Bình Dương góp 30% vốn điều lệ. Sau đó, Nguyễn Văn Minh đại diện cho Công ty 3-2 chuyển nhượng khu đất 43 ha cho Công ty Tân Phú và chuyển nhượng 30% vốn góp tại Công ty Tân Phú cho cho Công ty Âu Lạc của Nguyễn Đại Dương. Các bị can Trần Văn Nam, Bí thư Tỉnh uỷ (thời điểm đó), cùng các lãnh đạo tỉnh Bình Dương biết việc chuyển nhượng Khu đất 43 ha của Tổng Công ty 3-2 đã làm trái quy định của pháp luật song đã không thực hiện các biện pháp để quản lý, bảo toàn tài sản của Nhà nước, không ngăn chặn, huỷ bỏ việc chuyển nhượng trái pháp luật để chuyển trả khu đất 43 ha về cho Công ty Impco theo đúng phê duyệt của Tỉnh ủy. Khi có dư luận về những sai phạm tại khu đất 43 ha, Trần Văn Nam tiếp tục chỉ đạo bị can cấp dưới ban hành các văn bản đính chính, điều chỉnh nội dung sai lệch bản chất phương án sử dụng đất đã phê duyệt trước đó, nhằm hợp thức hoá, che giấu những sai phạm của Nguyễn Văn Minh và Tổng Công ty 3-2. Các bị cáo trong vụ án - Ảnh: Bộ Công an Khi Nguyễn Văn Minh xin ý kiến về việc cho phép Tổng Công ty 3-2 chuyển nhượng 30% vốn góp tại Công ty Tân Phú cho Công ty Âu Lạc, Trần Văn Nam tiếp tục chỉ đạo ban hành văn bản chấp thuận cho Tổng Công ty 3-2 chuyển nhượng 30% vốn góp của Tổng Công ty 3-2 đang sở hữu tại Công ty Tân Phú, dẫn đến toàn bộ tài sản thuộc sở hữu Nhà nước gồm quyền sử dụng đất 43 ha và 30% vốn góp chuyển sang công ty tư nhân. Hành vi phạm tội của các bị can đã gây thất thoát số tiền gần 985 tỉ đồng. Đối với khu đất 145 ha, phương án sử dụng đất đã được Tỉnh ủy, UBND tỉnh Bình Dương phê duyệt cho phép Tổng Công ty SX-XNK Bình Dương được tiếp tục sử dụng, quản lý, kế thừa tính vào giá trị doanh nghiệp sau cổ phần hoá. Tuy nhiên, lợi dụng chức vụ, quyền hạn được giao, với động cơ vụ lợi và mục đích tạo điều kiện cho 2 Công ty "sân sau" có lợi ích của Nguyễn Văn Minh và Nguyễn Thục Anh (con gái Minh) trong việc liên doanh, góp vốn bằng quyền sử dụng đất khu đất 145 ha khi thành lập Công ty cổ phần đầu tư và phát triển Tân Thành, Nguyễn Văn Minh đã chỉ đạo, quyết định chủ trương cùng các bị can tại Tổng Công ty 3-2 cùng các bị can khác phân loại, sắp xếp khu đất 145 ha vào mục "Tài sản chờ thanh lý", không xác định lại giá trị quyền sử dụng đất theo quy định để đưa vào giá trị doanh nghiệp khi cổ phần hóa. Bị can Thanh Liêm, khi lamg Chủ tịch UBND tỉnh Bình Dương, Trưởng Ban chỉ đạo cổ phần hóa Tổng Công ty 3-2, biết rõ chủ trương của Tỉnh ủy và phê duyệt phương án sử dụng đất của UBND tỉnh Bình Dương đối với khu đất 145 ha phải được kế thừa tính vào giá trị doanh nghiệp khi cổ phần hóa theo quy định của pháp luật. Tuy nhiên, bị can này vẫn ký Quyết định ngày 8-12-2017 phê duyệt giá trị doanh nghiệp Tổng Công ty 3-2, trong đó không có giá trị quyền sử dụng đất khu đất 145 ha. Hành vi nêu trên Nguyễn Văn Minh, Trần Thanh Liêm và đồng phạm gây thất thoát hơn 4.030 tỉ đồng. Về hành vi tham ô tài sản, với mục đích tạo nguồn tiền để hoàn ứng đã sử dụng trước đó, Nguyễn Văn Minh đã tạo điều kiện cho Công ty Phát Triển do con gái là Nguyễn Thục Anh và Trần Đình Như Ý thành lập phải trả nợ ngân hàng khi tham gia góp vốn vào Công ty Tân Thành, đồng thời tạo điều kiện để Công ty Hưng Vượng có nguồn tài chính thanh toán các khoản tiền đang còn nợ Tổng Công ty 3-2. Lợi dụng chức vụ, quyền hạn là Chủ tịch HĐQT Tổng Công ty 3-2, Nguyễn Văn Minh đã ban hành chủ trương, quyết định và chỉ đạo thẩm định giá trị quyền sử dụng đất Khu đất 145 ha để tạo giá trị chênh lệch so với giá trị khu đất khi đưa vào góp vốn tại Công ty Tân Thành, tiến hành mua bán, chuyển nhượng 19% cổ phần để chiếm đoạt hơn 815 tỉ đồng của Tổng Công ty 3-2. Theo cáo trạng, Nguyễn Văn Minh chiếm hưởng hơn 154 tỉ đồng; Nguyễn Thục Anh chiếm hưởng hơn 209 tỉ đồng; Trần Đình Như Ý chiếm hưởng hơn 201 tỉ đồng; Võ Hồng Cường chiếm hưởng gần 39 tỉ đồng.
"""))